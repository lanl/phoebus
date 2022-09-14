// © 2021-2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include <cmath>

#include "fixup.hpp"

#include <bvals/bvals_interfaces.hpp>
#include <defs.hpp>
#include <singularity-eos/eos/eos.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "geometry/tetrads.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/radiation.hpp"

using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;
using singularity::RadiationType;
using singularity::neutrinos::Opacity;

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto fix = std::make_shared<StateDescriptor>("fixup");
  Params &params = fix->AllParams();

  bool enable_flux_fixup = pin->GetOrAddBoolean("fixup", "enable_flux_fixup", false);
  params.Add("enable_flux_fixup", enable_flux_fixup);
  bool enable_fixup = pin->GetOrAddBoolean("fixup", "enable_fixup", false);
  params.Add("enable_fixup", enable_fixup);
  bool enable_floors = pin->GetOrAddBoolean("fixup", "enable_floors", false);
  bool enable_c2p_fixup = pin->GetOrAddBoolean("fixup", "enable_c2p_fixup", false);
  params.Add("enable_c2p_fixup", enable_c2p_fixup);
  bool enable_source_fixup = pin->GetOrAddBoolean("fixup", "enable_source_fixup", false);
  params.Add("enable_source_fixup", enable_source_fixup);
  bool enable_ceilings = pin->GetOrAddBoolean("fixup", "enable_ceilings", false);
  params.Add("enable_ceilings", enable_ceilings);
  bool enable_radiation_ceilings =
      pin->GetOrAddBoolean("fixup", "enable_radiation_ceilings", false);
  params.Add("enable_radiation_ceilings", enable_radiation_ceilings);
  bool report_c2p_fails = pin->GetOrAddBoolean("fixup", "report_c2p_fails", false);
  params.Add("report_c2p_fails", report_c2p_fails);
  bool report_source_fails = pin->GetOrAddBoolean("fixup", "report_source_fails", false);
  params.Add("report_source_fails", report_source_fails);
  bool enable_mhd_floors = pin->GetOrAddBoolean("fixup", "enable_mhd_floors", false);
  bool enable_rad_floors = pin->GetOrAddBoolean("fixup", "enable_rad_floors", false);

  const bool enable_mhd = pin->GetOrAddBoolean("fluid", "mhd", false);
  const bool enable_rad = pin->GetOrAddBoolean("physics", "rad", false);

  if (enable_mhd_floors && !enable_mhd) {
    enable_mhd_floors = false;
    PARTHENON_WARN("WARNING Disabling MHD floors because MHD is disabled!");
  }
  params.Add("enable_mhd_floors", enable_mhd_floors);

  if (enable_rad_floors && !enable_rad) {
    enable_rad_floors = false;
    PARTHENON_WARN("WARNING Disabling radiation floors because radiation is disabled!");
  }
  params.Add("enable_rad_floors", enable_rad_floors);

  if (enable_c2p_fixup && !enable_floors) {
    enable_floors = true;
    PARTHENON_WARN(
        "WARNING Forcing enable_floors to \"true\" because enable_c2p_fixup = true");
  }
  if (enable_mhd_floors && !enable_floors) {
    enable_floors = true;
    PARTHENON_WARN(
        "WARNING Forcing enable_floors to \"true\" because enable_mhd_floors = true");
  }
  if (enable_rad_floors && !enable_floors) {
    enable_floors = true;
    PARTHENON_WARN(
        "WARNING Forcing enable_floors to \"true\" because enable_rad_floors = true");
  }
  params.Add("enable_floors", enable_floors);

  if (enable_floors) {
    const std::string floor_type = pin->GetString("fixup", "floor_type");
    if (floor_type == "ConstantRhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_floor", 0.0);
      params.Add("floor", Floors(constant_rho_sie_floor_tag, rho0, sie0));
    } else if (floor_type == "ExpX1RhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "sie_exp_floor", -1.0);
      params.Add("floor", Floors(exp_x1_rho_sie_floor_tag, rho0, sie0, rp, sp));
    } else if (floor_type == "ExpX1RhoU") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "u0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "u_exp_floor", -3.0);
      params.Add("floor", Floors(exp_x1_rho_u_floor_tag, rho0, sie0, rp, sp));
    } else if (floor_type == "X1RhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "u0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "u_exp_floor", -3.0);
      params.Add("floor", Floors(x1_rho_sie_floor_tag, rho0, sie0, rp, sp));
    } else if (floor_type == "RRhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "u0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "u_exp_floor", -3.0);
      params.Add("floor", Floors(r_rho_sie_floor_tag, rho0, sie0, rp, sp));
    } else {
      PARTHENON_FAIL("invalid <fixup>/floor_type input");
    }
  } else {
    params.Add("floor", Floors());
  }

  if (enable_ceilings) {
    const std::string ceiling_type =
        pin->GetOrAddString("fixup", "ceiling_type", "ConstantGamSie");
    if (ceiling_type == "ConstantGamSie") {
      Real gam0 = pin->GetOrAddReal("fixup", "gam0_ceiling", 1000.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_ceiling", 100.0);
      params.Add("ceiling", Ceilings(constant_gam_sie_ceiling_tag, gam0, sie0));
    } else {
      PARTHENON_FAIL("invalid <fixup>/ceiling_type input");
    }
  } else {
    params.Add("ceiling", Ceilings());
  }

  if (enable_radiation_ceilings) {
    const std::string radiation_ceiling_type =
        pin->GetOrAddString("fixup", "radiation_ceiling_type", "ConstantXi0");
    if (radiation_ceiling_type == "ConstantXi0") {
      Real xi0 = pin->GetOrAddReal("fixup", "xi0_ceiling", 0.99);
      params.Add("radiation_ceiling",
                 RadiationCeilings(constant_xi0_radiation_ceiling_tag, xi0));
    }
  } else {
    params.Add("radiation_ceiling", RadiationCeilings());
  }

  if (enable_mhd_floors) {
    Real bsqorho_max = pin->GetOrAddReal("fixup", "bsqorho_max", 50.);
    Real bsqou_max = pin->GetOrAddReal("fixup", "bsqou_max", 2500.);
    Real uorho_max = pin->GetOrAddReal("fixup", "uorho_max", 50.);
    params.Add("bsqorho_max", bsqorho_max);
    params.Add("bsqou_max", bsqou_max);
    params.Add("uorho_max", uorho_max);
  }

  params.Add("bounds",
             Bounds(params.Get<Floors>("floor"), params.Get<Ceilings>("ceiling"),
                    params.Get<RadiationCeilings>("radiation_ceiling")));

  return fix;
}

template <typename T, class CLOSURE>
TaskStatus ApplyFloorsImpl(T *rc, IndexDomain domain = IndexDomain::entire) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;
  namespace ir = radmoment_internal;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *fluid_pkg = pmb->packages.Get("fluid").get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();

  bool rad_active = rad_pkg->Param<bool>("active");

  bool enable_floors = fix_pkg->Param<bool>("enable_floors");
  bool enable_mhd_floors = fix_pkg->Param<bool>("enable_mhd_floors");
  bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");

  if (!enable_floors) return TaskStatus::complete;

  const std::vector<std::string> vars(
      {p::density, c::density, p::velocity, c::momentum, p::energy, c::energy, p::bfield,
       p::ye, c::ye, p::pressure, p::temperature, p::gamma1, pr::J, pr::H, cr::E, cr::F,
       impl::cell_signal_speed, impl::fail, ir::tilPi});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  auto idx_J = imap.GetFlatIdx(pr::J, false);
  auto idx_H = imap.GetFlatIdx(pr::H, false);
  auto idx_E = imap.GetFlatIdx(cr::E, false);
  auto idx_F = imap.GetFlatIdx(cr::F, false);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);

  const int num_species = enable_rad_floors ? rad_pkg->Param<int>("num_species") : 0;

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  const Real c2p_tol = fluid_pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = fluid_pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter);

  Coordinates_t coords = rc->GetParentPointer().get()->coords;
  Real bsqorho_max, bsqou_max, uorho_max;
  if (enable_mhd_floors) {
    bsqorho_max = fix_pkg->Param<Real>("bsqorho_max");
    bsqou_max = fix_pkg->Param<Real>("bsqou_max");
    uorho_max = fix_pkg->Param<Real>("uorho_max");
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyFloors", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        double eos_lambda[2]; // used for stellarcollapse eos and
                              // other EOS's that require root
                              // finding.
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i)); // use last temp as initial guess

        double rho_floor, sie_floor;
        bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
                         rho_floor, sie_floor);
        double gamma_max, e_max;
        bounds.GetCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
                           gamma_max, e_max);
        Real xi_max;
        bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                                    coords.x3v(k, j, i), xi_max);

        Real rho_floor_max = rho_floor;
        Real u_floor_max = rho_floor * sie_floor;

        bool floor_applied = false;
        bool rad_floor_applied = false;
        bool ceiling_applied = false;
        bool rad_ceiling_applied = false;

        Real gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        Real gammacon[3][3];
        geom.MetricInverse(CellLocation::Cent, k, j, i, gammacon);
        const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
        Real betacon[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, betacon);
        const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);

        if (enable_mhd_floors) {
          Real Bsq = 0.0;
          Real Bdotv = 0.0;
          const Real vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                              v(b, pvel_lo + 2, k, j, i)};
          const Real bp[3] = {v(b, pb_lo, k, j, i), v(pb_lo + 1, k, j, i),
                              v(pb_lo + 2, k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vp, gcov);
          const Real iW = 1.0 / W;
          SPACELOOP2(ii, jj) {
            Bsq += gcov[ii + 1][jj + 1] * bp[ii] * bp[jj];
            Bdotv += gcov[ii + 1][jj + 1] * bp[ii] * vp[jj];
          }
          Real bcon0 = W * Bdotv / alpha;
          const Real bsq = (Bsq + alpha * alpha * bcon0 * bcon0) * iW * iW;

          rho_floor_max = std::max<Real>(rho_floor_max, bsq / bsqorho_max);
          u_floor_max = std::max<Real>(u_floor_max, bsq / bsqou_max);

          rho_floor_max = std::max<Real>(
              rho_floor_max,
              std::max<Real>(v(b, peng, k, j, i), u_floor_max) / uorho_max);
        }

        Real drho = rho_floor_max - v(b, prho, k, j, i);
        Real du = u_floor_max - v(b, peng, k, j, i);
        if (drho > 0. || du > 0.) {
          floor_applied = true;
          drho = std::max<Real>(drho, 1.e-100); // > 0 so EOS calls are happy
          du = std::max<Real>(du, 1.e-100);
          // set min du to be drho*sie_floor?
        }

        //        if (enable_rad_floors) {
        //          const Real r = std::exp(coords.x1v(k, j, i));
        //          // TODO(BRR) use bounds class
        //          const Real Jfloor = 1.e-10;
        //          Real dJ[radiation::MaxNumSpecies];
        //          for (int ispec = 0; ispec < num_species; ++ispec) {
        //            dJ[ispec] = Jfloor - v(b, idx_J(ispec), k, j, i);
        //            if (dJ[ispec] > 0.) {
        //              rad_floor_applied = true;
        //            }
        //          }
        //
        //          if (rad_floor_applied) {
        //            for (int ispec = 0; ispec < num_species; ++ispec) {
        //              dJ[ispec] = std::max<Real>(dJ[ispec], 1.e-100);
        //            }
        //          }
        //        }

        Real dcrho, dS[3], dBcons[3], dtau, dyecons;
        Real bp[3] = {0};
        if (pb_hi > 0) {
          SPACELOOP(ii) { bp[ii] = v(b, pb_lo + ii, k, j, i); }
        }

        if (floor_applied) {
          Real vp_normalobs[3] = {0}; // Inject floors at rest in normal observer frame
          Real ye_prim_default = 0.5;
          eos_lambda[0] = ye_prim_default;
          Real dprs =
              eos.PressureFromDensityInternalEnergy(drho, ratio(du, drho), eos_lambda);
          Real dgm1 = ratio(
              eos.BulkModulusFromDensityInternalEnergy(drho, ratio(du, drho), eos_lambda),
              dprs);
          prim2con::p2c(drho, vp_normalobs, bp, du, ye_prim_default, dprs, dgm1, gcov,
                        gammacon, betacon, alpha, sdetgam, dcrho, dS, dBcons, dtau,
                        dyecons);

          // Update cons vars (not B field)
          v(b, crho, k, j, i) += dcrho;
          SPACELOOP(ii) { v(b, cmom_lo + ii, k, j, i) += dS[ii]; }
          v(b, ceng, k, j, i) += dtau;
          if (pye > 0) {
            v(b, cye, k, j, i) += dyecons;
          }

          // fluid c2p
          auto status = invert(geom, eos, coords, k, j, i);
          if (status == con2prim_robust::ConToPrimStatus::failure) {
            // If fluid c2p fails, set to floors
            v(b, prho, k, j, i) = drho;
            SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = vp_normalobs[ii]; }
            v(b, peng, k, j, i) = du;
            if (pye > 0) {
              v(b, pye, k, j, i) = ye_prim_default;
            }

            // Update auxiliary primitives
            eos_lambda[0] = v(b, pye, k, j, i);
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), v(b, peng, k, j, i) / v(b, prho, k, j, i),
                eos_lambda);

            // Update cons vars (not B field)
            prim2con::p2c(drho, vp_normalobs, bp, du, ye_prim_default, dprs, dgm1, gcov,
                          gammacon, betacon, alpha, sdetgam, v(b, crho, k, j, i), dS,
                          dBcons, v(b, ceng, k, j, i), dyecons);
            SPACELOOP(ii) { v(b, cmom_lo + ii, k, j, i) = dS[ii]; }
            if (pye > 0) {
              v(b, cye, k, j, i) = dyecons;
            }
          }
        }

        // Fluid ceilings?
        Real vpcon[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                         v(b, pvel_lo + 2, k, j, i)};
        const Real W = phoebus::GetLorentzFactor(vpcon, gcov);
        if (W > gamma_max || v(b, peng, k, j, i) / v(b, prho, k, j, i) > e_max) {
          ceiling_applied = true;
        }

        if (ceiling_applied) {
          const Real rescale = std::sqrt(gamma_max * gamma_max - 1.) / (W * W - 1.);
          SPACELOOP(ii) { vpcon[ii] *= rescale; }
          SPACELOOP(ii) {
            v(b, pvel_lo + ii, k, j, i) = vpcon[ii];
          }

          Real ye = 0.;
          if (pye > 0) {
            ye = v(b, pye, k, j, i);
          }
          prim2con::p2c(v(b, prho, k, j, i), vpcon, bp, v(b, peng, k, j, i), ye,
                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gammacon, betacon,
                        alpha, sdetgam, v(b, crho, k, j, i), dS, dBcons,
                        v(b, ceng, k, j, i), dyecons);
          SPACELOOP(ii) { v(b, cmom_lo + ii, k, j, i) = dS[ii]; }
          if (pye > 0) {
            v(b, cye, k, j, i) = dyecons;
          }
        }

        if (rad_active) {
          Vec con_vp{v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                    v(b, pvel_lo + 2, k, j, i)};
          const Real W = phoebus::GetLorentzFactor(con_vp.data, gcov);
          Vec con_v{v(b, pvel_lo, k, j, i) / W, v(b, pvel_lo + 1, k, j, i) / W,
                    v(b, pvel_lo + 2, k, j, i) / W};
          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);
          Real E;
          Real J;
          Vec cov_H;
          Vec cov_F;
          Tens2 con_TilPi;
          CLOSURE c(con_v, &g);

          // If we applied fluid floors or ceilings we modified v^i, so we need to call
          // rad c2p here before we can use radiation primitive variables
          if (floor_applied || ceiling_applied) {
            // rad c2p
            // Vector cov_H{0}; // No flux
            // Tens2 conTilPi{0}; // Isotropic
            // Real dE[radiation::MaxNumSpecies];
            // Real cov_dF[radiation::MaxNumSpecies];

            for (int ispec = 0; ispec < num_species; ++ispec) {
              E = v(b, idx_E(ispec), k, j, i) / sdetgam;
              SPACELOOP(ii) { cov_F(ii) = v(b, idx_F(ispec, ii), k, j, i) / sdetgam; }

              // We need the real conTilPi
              if (iTilPi.IsValid()) {
                SPACELOOP2(ii, jj) {
                  con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                }
              } else {
                // TODO(BRR) don't be lazy and actually retrieve these
                Real xi = 0.;
                Real phi = acos(-1.0) * 1.000001;
                c.GetCovTilPiFromCon(E, cov_F, xi, phi, &con_TilPi);
              }

              c.Con2Prim(E, cov_F, con_TilPi, &J, &cov_H);
              v(b, idx_J(ispec), k, j, i) = J;
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J; }
            }
          }

          // TODO(BRR) use Bounds class
          const Real Jfloor = 1.e-10;
          rad_floor_applied = false;
          rad_ceiling_applied = false;
          Vec con_v_normalobs{0};
          CLOSURE c_iso(con_v_normalobs, &g);
          Tens2 con_tilPi_iso{0};
          for (int ispec = 0; ispec < num_species; ++ispec) {
            Real dJ = Jfloor - v(b, idx_J(ispec), k, j, i);
            if (dJ > 0.) {
              rad_floor_applied = true;

              constexpr bool update_cons_vars = false;
              // Update cons vars, c2p
              if (update_cons_vars) {
                J = dJ;
                SPACELOOP(ii) {
                  cov_H(ii) = 0.; // No flux H^i = 0 (note H^0 = 0 so H_i = 0)
                }
                PARTHENON_DEBUG_REQUIRE(!std::isnan(J), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(0)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(1)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(2)), "bad");
                c_iso.Prim2Con(J, cov_H, con_tilPi_iso, &E, &cov_F);

                E += v(b, idx_E(ispec), k, j, i) / sdetgam;
                SPACELOOP(ii) { cov_F(ii) += v(b, idx_F(ispec, ii), k, j, i) / sdetgam; }

                v(b, idx_E(ispec), k, j, i) = E * sdetgam;
                SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = cov_F(ii) * sdetgam; }
                
                // We need the real conTilPi
                if (iTilPi.IsValid()) {
                  SPACELOOP2(ii, jj) {
                    con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                  }
                } else {
                  // TODO(BRR) don't be lazy and actually retrieve these
                  Real xi = 0.;
                  Real phi = acos(-1.0) * 1.000001;
                  c.GetCovTilPiFromCon(E, cov_F, xi, phi, &con_TilPi);
                }

                c.Con2Prim(E, cov_F, con_TilPi, &J, &cov_H);

                v(b, idx_J(ispec), k, j, i) = J;
                SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J; }
              } else {
                v(b, idx_J(ispec), k, j, i) += dJ;
                
                J = v(b, idx_J(ispec), k, j, i);
                SPACELOOP(ii) {
                  cov_H(ii) = v(b, idx_H(ispec, ii), k, j, i) * J;
                }
                
                // We need the real conTilPi
                if (iTilPi.IsValid()) {
                  SPACELOOP2(ii, jj) {
                    con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                  }
                } else {
                  c.GetCovTilPiFromPrim(J, cov_H, &con_TilPi);
                }

                PARTHENON_DEBUG_REQUIRE(!std::isnan(J), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(0)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(1)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(2)), "bad");

                c.Prim2Con(J, cov_H, con_TilPi, &E, &cov_F);
                v(b, idx_E(ispec), k, j, i) = E * sdetgam;
                SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = cov_F(ii) * sdetgam; }
              }
            }

            Real xi = 0.;
            SPACELOOP2(ii, jj) {
              xi += gammacon[ii][jj] * v(b, idx_H(ispec, ii), k, j, i) *
                    v(b, idx_H(ispec, jj), k, j, i);
            }
            xi = std::sqrt(xi);
            if (xi > xi_max) {
              rad_ceiling_applied = true;

              J = v(b, idx_J(ispec), k, j, i);
              SPACELOOP(ii) {
                cov_H(ii) = (xi_max / xi) * v(b, idx_H(ispec, ii), k, j, i) * J;
                v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J;
              }

              // if conTilPi not provided, need to recalculate from prims
              //if (!iTilPi.IsValid()) {
              //  c.GetCovTilPiFromPrim(J, cov_H, &con_TilPi);
              //}
              // We need the real conTilPi
              if (iTilPi.IsValid()) {
                SPACELOOP2(ii, jj) {
                  con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                }
              } else {
                c.GetCovTilPiFromPrim(J, cov_H, &con_TilPi);
              }

              //printf("Ceil xi: [%i %i %i] J = %e cov_H = %e %e %e\n", k, j, i, J, cov_H(0), cov_H(1), cov_H(2));
              PARTHENON_DEBUG_REQUIRE(!std::isnan(J), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(0)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(1)), "bad");
              PARTHENON_DEBUG_REQUIRE(!std::isnan(cov_H(2)), "bad");
              c.Prim2Con(J, cov_H, con_TilPi, &E, &cov_F);
              v(b, idx_E(ispec), k, j, i) = E * sdetgam;
              SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = cov_F(ii) * sdetgam; }
            }
          }

          // rad floor applied?

          // calculate dcons
          // update rad cons
          // rad c2p

          // rad ceiling applied?
        }
      });

  //        if (rad_floor_applied) {
  //          // Add radiation energy density in normal observer frame
  //          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k,
  //          j, i); Vector con_v{0}; // Normal observer frame Vector cov_H{0}; //
  //          No flux Tens2 conTilPi{0}; // Isotropic CLOSURE c(con_v, &g); Real
  //          dE[radiation::MaxNumSpecies]; Real cov_dF[radiation::MaxNumSpecies];
  //          for (int ispec = 0; ispec < num_species; ++ispec) {
  //            c.Prim2Con(dJ[ispec], cov_H, conTilPi, &(dE[ispec]),
  //            &(cov_dF[ispec])); v(b, idx_E(ispec), k, j, i) += sdetgam *
  //            dE[ispec]; SPACELOOP(ii) {
  //              v(b, idx_F(ispec, ii), k, j, i) += sdetgam * cov_dF[ispec](ii);
  //            }
  //
  //            // Convert back to updated primitive variables (reuse d* variables),
  //            // need real fluid velocity
  //            dE[ispec] = v(b, idx_E(ispec), k, j, i) / sdetgam;
  //            SPACELOOP(ii) {
  //              cov_dF[ispec](ii) = v(b, idx_F(ispec, ii), k, j, i) / sdetgam
  //            }
  //            // We need the real conTilPi
  //            if (idx_tilPi.IsValid()) {
  //              SPACELOOP2(ii, jj) {
  //                conTilPi(ii, jj) = v(b, idx_tilPi(ispec, ii, jj), k, j, i);
  //              }
  //            } else {
  //              // TODO(BRR) don't be lazy and actually retrieve these
  //              Real xi = 0.;
  //              Real phi = acos(-1.0) * 1.000001;
  //              c.GetCovTilPiFromCon(dE[ispec], cov_dF[ispec], xi, phi, conTilPi);
  //            }
  //            c.Con2Prim(dE[ispec], cov_dF[ispec], conTilPi, &(dJ[ispec]),
  //            &cov_H); v(b, idx_J(ispec), k, j, i) = dJ[ispec]; SPACELOOP(ii) {
  //              v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii);
  //            }
  //          }
  //        }
  //
  //        // Rad ceilings?
  //
  //        if (floor_applied || ceiling_applied || rad_floor_applied ||
  //        rad_ceiling_applied) {
  //          // Call rad p2c regardless because v^i presumably changed even if rad
  //          prims didnt
  //        }

  //        bool floor_applied = false;
  //        if (v(b, prho, k, j, i) < rho_floor) {
  //          floor_applied = true;
  //          v(b, prho, k, j, i) = rho_floor;
  //        }
  //        if (v(b, peng, k, j, i) / v(b, prho, k, j, i) < sie_floor) {
  //          floor_applied = true;
  //          v(b, peng, k, j, i) = sie_floor * v(b, prho, k, j, i);
  //        }
  //
  //        Real gcov[4][4];
  //        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
  //        const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
  //
  //        Real con_vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
  //                          v(b, pvel_lo + 2, k, j, i)};
  //        const Real W = phoebus::GetLorentzFactor(con_vp, gcov);
  //        if (W > gamma_max) {
  //          floor_applied = true;
  //          const Real rescale = std::sqrt((gamma_max * gamma_max - 1.) / (W * W
  //          - 1.)); SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) *= rescale; }
  //        }
  //
  //        if (enable_mhd_floors) {
  //          Real Bsq = 0.0;
  //          Real Bdotv = 0.0;
  //          const Real vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j,
  //          i),
  //                              v(b, pvel_lo + 2, k, j, i)};
  //          const Real bp[3] = {v(b, pb_lo, k, j, i), v(pb_lo + 1, k, j, i),
  //                              v(pb_lo + 2, k, j, i)};
  //          const Real W = phoebus::GetLorentzFactor(vp, gcov);
  //          const Real iW = 1.0 / W;
  //          SPACELOOP2(ii, jj) {
  //            Bsq += gcov[ii + 1][jj + 1] * bp[ii] * bp[jj];
  //            Bdotv += gcov[ii + 1][jj + 1] * bp[ii] * vp[jj];
  //          }
  //          Real bcon0 = W * Bdotv / alpha;
  //          const Real bsq = (Bsq + alpha * alpha * bcon0 * bcon0) * iW * iW;
  //
  //          if (bsq / v(b, prho, k, j, i) > bsqorho_max) {
  //            floor_applied = true;
  //            v(b, prho, k, j, i) = bsq / bsqorho_max;
  //          }
  //          if (bsq / v(b, peng, k, j, i) > bsqou_max) {
  //            floor_applied = true;
  //            v(b, peng, k, j, i) = bsq / bsqou_max;
  //          }
  //          if (v(b, peng, k, j, i) / v(b, prho, k, j, i) > uorho_max) {
  //            floor_applied = true;
  //            v(b, peng, k, j, i) = uorho_max * v(b, prho, k, j, i);
  //          }
  //        }

  //        if (enable_rad_floors) {
  //          const Real r = std::exp(coords.x1v(k, j, i));
  //          const Real Jmin = 1.e-10;
  //          for (int ispec = 0; ispec < num_species; ++ispec) {
  //            if (v(b, idx_J(ispec), k, j, i) < Jmin) {
  //              floor_applied = true;
  //              v(b, idx_J(ispec), k, j, i) = Jmin;
  //            }
  //
  //            Real con_gamma[3][3];
  //            geom.MetricInverse(CellLocation::Cent, k, j, i, con_gamma);
  //            Real xi = 0.;
  //            SPACELOOP2(ii, jj) {
  //              xi += con_gamma[ii][jj] * v(b, idx_H(ispec, ii), k, j, i) *
  //                    v(b, idx_H(ispec, jj), k, j, i);
  //            }
  //            xi = std::sqrt(xi);
  //            if (xi > xi_max) {
  //              floor_applied = true;
  //              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
  //            }
  //          }
  //        }
  //
  //        if (floor_applied) {
  //          // Update dependent primitives
  //          if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
  //          v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
  //              v(b, prho, k, j, i), v(b, peng, k, j, i) / v(b, prho, k, j, i),
  //              eos_lambda);
  //          v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
  //              v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
  //          v(b, gm1, k, j, i) =
  //              ratio(eos.BulkModulusFromDensityTemperature(v(b, prho, k, j, i),
  //                                                          v(b, tmp, k, j, i),
  //                                                          eos_lambda),
  //                    v(b, prs, k, j, i));
  //
  //          // Update fluid conserved variables
  //          const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
  //          Real gcon[3][3];
  //          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
  //          Real beta[3];
  //          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  //          Real S[3];
  //          const Real con_vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k,
  //          j, i),
  //                                  v(b, pvel_hi, k, j, i)};
  //          Real bcons[3];
  //          Real bp[3] = {0.0, 0.0, 0.0};
  //          if (pb_hi > 0) {
  //            bp[0] = v(b, pb_lo, k, j, i);
  //            bp[1] = v(b, pb_lo + 1, k, j, i);
  //            bp[2] = v(b, pb_hi, k, j, i);
  //          }
  //          Real ye_cons;
  //          Real ye_prim = 0.0;
  //          if (pye > 0) {
  //            ye_prim = v(b, pye, k, j, i);
  //          }
  //          Real sig[3];
  //          prim2con::p2c(v(b, prho, k, j, i), con_vp, bp, v(b, peng, k, j, i),
  //          ye_prim,
  //                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gcon,
  //                        beta, alpha, gdet, v(b, crho, k, j, i), S, bcons, v(b,
  //                        ceng, k, j, i), ye_cons, sig);
  //          v(b, cmom_lo, k, j, i) = S[0];
  //          v(b, cmom_lo + 1, k, j, i) = S[1];
  //          v(b, cmom_hi, k, j, i) = S[2];
  //          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
  //          for (int m = slo; m <= shi; m++) {
  //            v(b, m, k, j, i) = sig[m - slo];
  //          }
  //
  //          // Update radiation conserved variables
  //          Real cov_gamma[3][3];
  //          geom.Metric(CellLocation::Cent, k, j, i, cov_gamma);
  //          const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma);
  //          Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};
  //          for (int ispec = 0; ispec < num_species; ++ispec) {
  //            const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);
  //
  //            typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b,
  //            k, j, i); CLOSURE c(con_v, &g);
  //
  //            Real E;
  //            Vec covF;
  //            Tens2 conTilPi;
  //            Real J = v(b, idx_J(ispec), k, j, i);
  //            Vec covH = {{v(b, idx_H(ispec, 0), k, j, i) * J,
  //                         v(b, idx_H(ispec, 1), k, j, i) * J,
  //                         v(b, idx_H(ispec, 2), k, j, i) * J}};
  //
  //            if (iTilPi.IsValid()) {
  //              SPACELOOP2(ii, jj) {
  //                conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
  //              }
  //            } else {
  //              c.GetCovTilPiFromPrim(J, covH, &conTilPi);
  //            }
  //
  //            c.Prim2Con(J, covH, conTilPi, &E, &covF);
  //
  //            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
  //            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam *
  //            covF(ii); }
  //          }
  //        }
  //      });
  printf("Done with %s\n", __func__);

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ApplyFloors(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad_pkg = pm->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return ApplyFloorsImpl<T, radiation::ClosureM1<Vec, Tens2, settings>>(rc);
  } else if (method == "moment_eddington") {
    return ApplyFloorsImpl<T, radiation::ClosureEdd<Vec, Tens2, settings>>(rc);
  } else if (method == "mocmc") {
    return ApplyFloorsImpl<T, radiation::ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return ApplyFloorsImpl<T, radiation::ClosureEdd<Vec, Tens2, settings>>(rc);
  }
  return TaskStatus::fail;
}

template TaskStatus ApplyFloors<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

template <typename T, class CLOSURE>
TaskStatus RadConservedToPrimitiveFixupImpl(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace ir = radmoment_internal;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;

  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_rad = rad_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_rad) return TaskStatus::complete;

  const std::vector<std::string> vars({p::density,
                                       c::density,
                                       p::velocity,
                                       c::momentum,
                                       p::energy,
                                       c::energy,
                                       p::bfield,
                                       p::ye,
                                       c::ye,
                                       p::pressure,
                                       p::temperature,
                                       p::gamma1,
                                       pr::J,
                                       pr::H,
                                       cr::E,
                                       cr::F,
                                       impl::cell_signal_speed,
                                       ir::tilPi,
                                       ir::c2pfail,
                                       impl::fail});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  auto idx_pvel = imap.GetFlatIdx(p::velocity);
  auto idx_cmom = imap.GetFlatIdx(c::momentum);
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  auto idx_J = imap.GetFlatIdx(pr::J, false);
  auto idx_H = imap.GetFlatIdx(pr::H, false);
  auto idx_E = imap.GetFlatIdx(cr::E, false);
  auto idx_F = imap.GetFlatIdx(cr::F, false);
  int ifluidfail = imap[impl::fail].first;
  int iradfail = imap[ir::c2pfail].first;
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total rad nfail: %i\n", nfail_total);
    IndexRange ibi = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jbi = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    nfail_total = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kbi.s, kbi.e, jbi.s, jbi.e, ibi.s, ibi.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total rad interior nfail: %i\n", nfail_total);
  }

  // TODO(BRR) make this less ugly? Do this at all?
  // TODO(BRR) if (use_ghost_for_stencil_fixup) {} ?
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "C2P fail initialization", DevExecSpace(), 0, v.GetDim(5) - 1,
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ib.s || i > ib.e || j < jb.s || j > jb.e || k < kb.s || k > kb.e) {
          // Do not use ghost zones as data for averaging
          // TODO(BRR) need to allow ghost zones from neighboring blocks
          v(b, ifluidfail, k, j, i) = con2prim_robust::FailFlags::fail;
          v(b, iradfail, k, j, i) = radiation::FailFlags::fail;
        }
      });

  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  const int nspec = idx_E.DimSize(1);
  const int ndim = pmb->pmy_mesh->ndim;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadConToPrim::Solve fixup", DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s + 1, jb.e - 1, ib.s + 1, ib.e - 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // If fluid fail but rad success, recalculate rad c2p and set iradfail with
        // result, then still process if (fluid fail && rad fail) check.
        // Note that it is assumed that the fluid is already fixed up
        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = v(b, iradfail, k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              v(b, iradfail, k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v(b, iradfail, k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 v(b, iradfail, k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v(b, iradfail, k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   v(b, iradfail, k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };

        if (v(b, ifluidfail, k, j, i) == con2prim_robust::FailFlags::fail ||
            v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {

          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          const Real vel[] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                              v(b, idx_pvel(2), k, j, i)};

          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);

          Real num_valid = v(b, iradfail, k, j, i - 1) + v(b, iradfail, k, j, i + 1);
          if (ndim > 1)
            num_valid += v(b, iradfail, k, j - 1, i) + v(b, iradfail, k, j + 1, i);
          if (ndim == 3)
            num_valid += v(b, iradfail, k - 1, j, i) + v(b, iradfail, k + 1, j, i);
          if (num_valid > 0.5) {
            const Real norm = 1.0 / num_valid;
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = fixup(idx_J(ispec), norm);
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = fixup(idx_H(ispec, ii), norm);
              }
            }
          } else {
            //            printf("[%i %i %i] no valid rad c2p neighbors!\n", k, j, i);
            Real ucon[4] = {0};
            Real vpcon[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                             v(b, idx_pvel(2), k, j, i)};
            GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);
            Geometry::Tetrads tetrads(ucon, gcov);
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = 1.e-10;
              Real Mcon_fluid[4] = {v(b, idx_J(ispec), k, j, i), 0., 0., 0.};
              Real Mcon_coord[4] = {0};
              tetrads.TetradToCoordCon(Mcon_fluid, Mcon_coord);
              Vec Hcon = {Mcon_coord[1] - ucon[1] * v(b, idx_J(ispec), k, j, i),
                          Mcon_coord[2] - ucon[2] * v(b, idx_J(ispec), k, j, i),
                          Mcon_coord[3] - ucon[3] * v(b, idx_J(ispec), k, j, i)};
              Vec Hcov;
              g.lower3Vector(Hcon, &Hcov);

              // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = Hcov(ii) / v(b,
              // idx_J(ispec), k, j, i); }
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = 0.; }
            }
          }

          Real xi_max;
          bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                                      coords.x3v(k, j, i), xi_max);

          // Clamp variables
          for (int ispec = 0; ispec < nspec; ispec++) {
            v(b, idx_J(ispec), k, j, i) =
                std::max<Real>(v(b, idx_J(ispec), k, j, i), 1.e-10);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                         v(b, idx_H(ispec, 2), k, j, i)};
            Real xi = std::sqrt(g.contractCov3Vectors(cov_H, cov_H));
            if (xi > xi_max) {
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
            }
          }

          const Real W = phoebus::GetLorentzFactor(vel, gcov);
          Vec con_v({vel[0] / W, vel[1] / W, vel[2] / W});
          CLOSURE c(con_v, &g);
          for (int ispec = 0; ispec < nspec; ispec++) {
            Real E;
            Vec cov_F;
            Tens2 con_tilPi;
            Real J = v(b, idx_J(ispec), k, j, i);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i) * J,
                         v(b, idx_H(ispec, 1), k, j, i) * J,
                         v(b, idx_H(ispec, 2), k, j, i) * J};
            if (iTilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                con_tilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);
            }
            c.Prim2Con(J, cov_H, con_tilPi, &E, &cov_F);

            Real xi = std::sqrt(g.contractCov3Vectors(cov_H, cov_H)) / J;
            //            printf("Rad fixup [%i %i %i] J = %e cov_H = %e %e %e (xi = %e) E
            //            = %e cov_F = %e %e %e\n",
            //              k,j,i,J,cov_H(0),cov_H(1),cov_H(2),xi,E,cov_F(0),cov_F(1),cov_F(2));

            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * cov_F(ii); }
          }
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus RadConservedToPrimitiveFixup(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad_pkg = pm->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureM1<Vec, Tens2, settings>>(
        rc);
  } else if (method == "moment_eddington") {
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureEdd<Vec, Tens2, settings>>(
        rc);
  } else if (method == "mocmc") {
    return RadConservedToPrimitiveFixupImpl<
        T, radiation::ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureEdd<Vec, Tens2, settings>>(
        rc);
  }
  return TaskStatus::fail;
}

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *fluid_pkg = pmb->packages.Get("fluid").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();

  const std::vector<std::string> vars({p::density, c::density, p::velocity, c::momentum,
                                       p::energy, c::energy, p::bfield, p::ye, c::ye,
                                       p::pressure, p::temperature, p::gamma1,
                                       impl::cell_signal_speed, impl::fail});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  int ifail = imap[impl::fail].first;

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_fluid = fluid_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_fluid) return TaskStatus::complete;

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total nfail: %i\n", nfail_total);
  }

  const int ndim = pmb->pmy_mesh->ndim;

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  // TODO(BRR) make this less ugly (or do this at all?)
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "C2P fail initialization", DevExecSpace(), 0, v.GetDim(5) - 1,
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ib.s || i > ib.e || j < jb.s || j > jb.e || k < kb.s || k > kb.e) {
          // Do not use ghost zones as data for averaging
          // TODO(BRR) need to allow ghost zones from neighboring blocks
          v(b, ifail, k, j, i) = con2prim_robust::FailFlags::fail;
        }
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real eos_lambda[2]; // use last temp as initial guess
        eos_lambda[0] = 0.5;
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i));

        // Need to account for not stenciling outside of ghost zones
        bool is_outer_ghost_layer =
            (i == ib.s || i == ib.e || j == jb.s || j == jb.e || k == kb.s || k == kb.e);

        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = v(b, ifail, k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              v(b, ifail, k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v(b, ifail, k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 v(b, ifail, k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v(b, ifail, k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   v(b, ifail, k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };
        if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
          Real num_valid = 0;
          if (!is_outer_ghost_layer) {
            num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
            if (ndim > 1)
              num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
            if (ndim == 3)
              num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);
          }
          if (num_valid > 0.5) {
            const Real norm = 1.0 / num_valid;
            v(b, prho, k, j, i) = fixup(prho, norm);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b, pv, k, j, i) = fixup(pv, norm);
            }
            v(b, peng, k, j, i) = fixup(peng, norm);

            if (pye > 0) v(b, pye, k, j, i) = fixup(pye, norm);
            if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                eos_lambda);
            v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
                v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
            v(b, gm1, k, j, i) =
                ratio(eos.BulkModulusFromDensityTemperature(
                          v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda),
                      v(b, prs, k, j, i));
          } else {
            printf("[%i %i %i] no valid c2p neighbors!\n", k, j, i);
            // No valid neighbors; set fluid mass/energy to near-zero and set primitive
            // velocities to zero

            v(b, prho, k, j, i) = 1.e-20;
            v(b, peng, k, j, i) = 1.e-20;

            // Safe value for ye
            if (pye > 0) {
              v(b, pye, k, j, i) = 0.5;
            }

            if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                eos_lambda);
            v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
                v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
            v(b, gm1, k, j, i) =
                ratio(eos.BulkModulusFromDensityTemperature(
                          v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda),
                      v(b, prs, k, j, i));

            // Zero primitive velocities
            SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = 0.; }
          }
          // Update conserved variables
          const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real S[3];
          const Real vel[] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                              v(b, pvel_hi, k, j, i)};
          Real bcons[3];
          Real bp[3] = {0.0, 0.0, 0.0};
          if (pb_hi > 0) {
            bp[0] = v(b, pb_lo, k, j, i);
            bp[1] = v(b, pb_lo + 1, k, j, i);
            bp[2] = v(b, pb_hi, k, j, i);
          }
          Real ye_cons;
          Real ye_prim = 0.0;
          if (pye > 0) {
            ye_prim = v(b, pye, k, j, i);
          }
          Real sig[3];
          prim2con::p2c(v(b, prho, k, j, i), vel, bp, v(b, peng, k, j, i), ye_prim,
                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gcon, beta, alpha,
                        gdet, v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i), ye_cons,
                        sig);
          v(b, cmom_lo, k, j, i) = S[0];
          v(b, cmom_lo + 1, k, j, i) = S[1];
          v(b, cmom_hi, k, j, i) = S[2];
          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
          for (int m = slo; m <= shi; m++) {
            v(b, m, k, j, i) = sig[m - slo];
          }
        }
      });

  return TaskStatus::complete;
}

// If radiation source terms fail (probably due to a rootfind failing to converge) average
// the fluid and radiation variables over good neighbors. Or if there are no good
// neighbors, set everything to the floors
template <typename T, class CLOSURE>
TaskStatus SourceFixupImpl(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;
  namespace ir = radmoment_internal;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();
  if (!rad_pkg->Param<bool>("active")) {
    return TaskStatus::complete;
  }
  bool enable_source_fixup = fix_pkg->Param<bool>("enable_source_fixup");
  if (!enable_source_fixup) {
    return TaskStatus::complete;
  }

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  const std::vector<std::string> vars(
      {p::density, c::density, p::velocity, c::momentum, p::energy, c::energy, p::bfield,
       p::ye, c::ye, p::pressure, p::temperature, p::gamma1, pr::J, pr::H, cr::E, cr::F,
       impl::cell_signal_speed, ir::srcfail, ir::tilPi});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  auto idx_pvel = imap.GetFlatIdx(p::velocity);
  auto idx_cmom = imap.GetFlatIdx(c::momentum);
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  int ifail = imap[ir::srcfail].first;
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);
  // TODO(BRR) get iTilPi for MOCMC

  bool report_source_fails = fix_pkg->Param<bool>("report_source_fails");
  if (report_source_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Source fixup failures", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, ifail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total source nfail: %i\n", nfail_total);
  }

  const int ndim = pmb->pmy_mesh->ndim;

  auto geom = Geometry::GetCoordinateSystem(rc);
  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  auto num_species = rad_pkg->Param<int>("num_species");

  // TODO(BRR) make this less ugly
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Source fail initialization", DevExecSpace(), 0,
      v.GetDim(5) - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ib.s || i > ib.e || j < jb.s || j > jb.e || k < kb.s || k > kb.e) {
          // Do not use ghost zones as data for averaging
          // TODO(BRR) need to allow ghost zones from neighboring blocks
          v(b, ifail, k, j, i) = radiation::FailFlags::fail;
        }
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Source fixup", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        double eos_lambda[2]; // used for stellarcollapse eos and
                              // other EOS's that require root
                              // finding.
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i)); // use last temp as initial guess

        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = v(b, ifail, k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              v(b, ifail, k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v(b, ifail, k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 v(b, ifail, k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v(b, ifail, k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   v(b, ifail, k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };
        if (v(b, ifail, k, j, i) == radiation::FailFlags::fail) {

          Real num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
          if (ndim > 1) num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
          if (ndim == 3) num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);

          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);

          if (num_valid > 0.5) {
            const Real norm = 1.0 / num_valid;

            v(b, prho, k, j, i) = fixup(prho, norm);
            v(b, peng, k, j, i) = fixup(peng, norm);
            SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = fixup(idx_pvel(ii), norm); }
            if (pye > 0) {
              v(b, pye, k, j, i) = fixup(pye, norm);
            }

            for (int ispec = 0; ispec < num_species; ispec++) {
              v(b, idx_J(ispec), k, j, i) = fixup(idx_J(ispec), norm);
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = fixup(idx_H(ispec, ii), norm);
              }
            }
          } else {
            // No valid neighbors; set to floors with zero spatial velocity
            printf("No valid neighbors! %i %i %i\n", k, j, i);

            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i),
                             coords.x3v(k, j, i), rho_floor, sie_floor);
            v(b, prho, k, j, i) = rho_floor;
            v(b, peng, k, j, i) = rho_floor * sie_floor;

            // Zero primitive velocities
            SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = 0.; }

            // Auxiliary primitives
            // Safe value for ye
            if (pye > 0) {
              v(b, pye, k, j, i) = 0.5;
            }
            if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                eos_lambda);

            Real ucon[4] = {0};
            Real vpcon[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                             v(b, idx_pvel(2), k, j, i)};
            GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);
            Geometry::Tetrads tetrads(ucon, gcov);

            for (int ispec = 0; ispec < num_species; ispec++) {
              v(b, idx_J(ispec), k, j, i) = 1.e-10;

              Real Mcon_fluid[4] = {v(b, idx_J(ispec), k, j, i), 0., 0., 0.};
              Real Mcon_coord[4] = {0};
              tetrads.TetradToCoordCon(Mcon_fluid, Mcon_coord);
              Real Hcon[3] = {Mcon_coord[1] - ucon[1] * v(b, idx_J(ispec), k, j, i),
                              Mcon_coord[2] - ucon[2] * v(b, idx_J(ispec), k, j, i),
                              Mcon_coord[3] - ucon[3] * v(b, idx_J(ispec), k, j, i)};
              Real Hcov[3] = {0};
              SPACELOOP2(ii, jj) { Hcov[ii] += gcov[ii + 1][jj + 1] * Hcon[jj]; }

              // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = Hcov[ii] / v(b,
              // idx_J(ispec), k, j, i);
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = 0.;
                printf("Fail [%i %i %i] H[%i] = %e\n", k, j, i, ii,
                       v(b, idx_H(ispec, ii), k, j, i));
              }

              // TODO(BRR) is this wrong? Want zero flux in fluid frame, this isn't it
              // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = 0.; }
              // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = Mcon_coord[ii + 1] /
              // v(b, idx_J(ispec), k, j, i);
              //  printf("Fail [%i %i %i] H[%i] = %e\n", k,j,i,ii,v(b, idx_H(ispec, ii),
              //  k, j, i));
              //}
            }
          }

          // Apply floors (redundant with case of no valid neighbors)
          double rho_floor, sie_floor;
          bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
                           rho_floor, sie_floor);
          v(b, prho, k, j, i) = std::max<Real>(v(b, prho, k, j, i), rho_floor);
          v(b, peng, k, j, i) =
              std::max<Real>(v(b, peng, k, j, i), rho_floor * sie_floor);

          if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
          v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
              v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
              eos_lambda);
          v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
              v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
          v(b, gm1, k, j, i) =
              ratio(eos.BulkModulusFromDensityTemperature(v(b, prho, k, j, i),
                                                          v(b, tmp, k, j, i), eos_lambda),
                    v(b, prs, k, j, i));

          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);

          Real gamma_max, e_max;
          bounds.GetCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                             coords.x3v(k, j, i), gamma_max, e_max);

          // Clamp fluid velocity
          Real con_vp[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                            v(b, idx_pvel(2), k, j, i)};
          Real W = phoebus::GetLorentzFactor(con_vp, gcov);
          if (W > gamma_max) {
            const Real rescale = std::sqrt((gamma_max * gamma_max - 1.) / (W * W - 1.));
            SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) *= rescale; }
          }
          const Real r = std::exp(coords.x1v(k, j, i));
          Real xi_max;
          bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                                      coords.x3v(k, j, i), xi_max);
          const Real Jmin = 1.e-10;
          for (int ispec = 0; ispec < num_species; ispec++) {
            v(b, idx_J(ispec), k, j, i) =
                std::max<Real>(v(b, idx_J(ispec), k, j, i), Jmin);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                         v(b, idx_H(ispec, 2), k, j, i)};
            Vec con_H;
            g.raise3Vector(cov_H, &con_H);
            Real xi = 0.;
            SPACELOOP(ii) { xi += cov_H(ii) * con_H(ii); }
            xi = std::sqrt(xi);
            if (xi > xi_max) {
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
            }
          }

          // Update MHD conserved variables
          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real S[3];
          const Real vel[] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                              v(b, idx_pvel(2), k, j, i)};
          Real bcons[3];
          Real bp[3] = {0.0, 0.0, 0.0};
          if (pb_hi > 0) {
            bp[0] = v(b, pb_lo, k, j, i);
            bp[1] = v(b, pb_lo + 1, k, j, i);
            bp[2] = v(b, pb_hi, k, j, i);
          }
          Real ye_cons;
          Real ye_prim = 0.0;
          if (pye > 0) {
            ye_prim = v(b, pye, k, j, i);
          }
          Real sig[3];
          prim2con::p2c(v(b, prho, k, j, i), vel, bp, v(b, peng, k, j, i), ye_prim,
                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gcon, beta, alpha,
                        sdetgam, v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i),
                        ye_cons, sig);
          v(b, idx_cmom(0), k, j, i) = S[0];
          v(b, idx_cmom(1), k, j, i) = S[1];
          v(b, idx_cmom(2), k, j, i) = S[2];
          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
          for (int m = slo; m <= shi; m++) {
            v(b, m, k, j, i) = sig[m - slo];
          }

          // Update radiation conserved variables
          W = phoebus::GetLorentzFactor(vel, gcov);
          Vec con_v({vel[0] / W, vel[1] / W, vel[2] / W});
          CLOSURE c(con_v, &g);
          for (int ispec = 0; ispec < num_species; ispec++) {
            Real E;
            Vec cov_F;
            Tens2 con_TilPi = {0};
            Real J = v(b, idx_J(ispec), k, j, i);
            Vec cov_H = {J * v(b, idx_H(ispec, 0), k, j, i),
                         J * v(b, idx_H(ispec, 1), k, j, i),
                         J * v(b, idx_H(ispec, 2), k, j, i)};
            if (iTilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetCovTilPiFromPrim(J, cov_H, &con_TilPi);
            }
            c.Prim2Con(J, cov_H, con_TilPi, &E, &cov_F);
            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * cov_F(ii); }
          }
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus SourceFixup(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad_pkg = pm->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return SourceFixupImpl<T, radiation::ClosureM1<Vec, Tens2, settings>>(rc);
  } else if (method == "moment_eddington") {
    return SourceFixupImpl<T, radiation::ClosureEdd<Vec, Tens2, settings>>(rc);
  } else if (method == "mocmc") {
    return SourceFixupImpl<T, radiation::ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return SourceFixupImpl<T, radiation::ClosureEdd<Vec, Tens2, settings>>(rc);
  }
  return TaskStatus::fail;
}

TaskStatus FixFluxes(MeshBlockData<Real> *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  using parthenon::BoundaryFace;
  using parthenon::BoundaryFlag;
  auto *pmb = rc->GetParentPointer().get();
  auto &fixup_pkg = pmb->packages.Get("fixup");
  if (!fixup_pkg->Param<bool>("enable_flux_fixup")) return TaskStatus::complete;

  auto fluid = pmb->packages.Get("fluid");
  const std::string ix1_bc = fluid->Param<std::string>("ix1_bc");
  const std::string ox1_bc = fluid->Param<std::string>("ox1_bc");
  const std::string ix2_bc = fluid->Param<std::string>("ix2_bc");
  const std::string ox2_bc = fluid->Param<std::string>("ox2_bc");

  auto rad = pmb->packages.Get("radiation");
  int num_species = 0;
  if (rad->Param<bool>("active")) {
    num_species = rad->Param<int>("num_species");
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int ndim = pmb->pmy_mesh->ndim;

  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace cr = radmoment_cons;

  // x1-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
    if (ix1_bc == "outflow") {
      PackIndexMap imap;
      auto v =
          rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                     std::vector<std::string>({c::density, cr::E}), imap);
      const auto crho = imap[c::density].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = std::min(v.flux(X1DIR, crho, k, j, i), 0.0);
            // TODO(BRR) This seems to be unstable (at the outer boundary)
            //            if (idx_E.IsValid()) {
            //              for (int ispec = 0; ispec < num_species; ispec++) {
            //                v.flux(X1DIR, idx_E(ispec), k, j, i) =
            //                    std::min(v.flux(X1DIR, idx_E(ispec), k, j, i), 0.0);
            //              }
            //            }
          });
    } else if (ix1_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density, c::energy, cr::E}),
          std::vector<std::string>({c::density, c::energy, cr::E}), imap);
      const auto crho = imap[c::density].first;
      const auto cener = imap[c::energy].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = 0.0;
            v.flux(X1DIR, cener, k, j, i) = 0.0;
            if (idx_E.IsValid()) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X1DIR, idx_E(ispec), k, j, i) = 0.0;
              }
            }
          });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    if (ox1_bc == "outflow") {
      PackIndexMap imap;
      auto v =
          rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                     std::vector<std::string>({c::density, cr::E}), imap);
      const auto crho = imap[c::density].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.e + 1, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = std::max(v.flux(X1DIR, crho, k, j, i), 0.0);
            // TODO(BRR) Unstable (see above)
            //            if (idx_E.IsValid()) {
            //              for (int ispec = 0; ispec < num_species; ispec++) {
            //                v.flux(X1DIR, idx_E(ispec), k, j, i) =
            //                    std::max(v.flux(X1DIR, idx_E(ispec), k, j, i), 0.0);
            //              }
            //            }
          });
    } else if (ox1_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density, c::energy, cr::E}),
          std::vector<std::string>({c::density, c::energy, cr::E}), imap);
      const auto crho = imap[c::density].first;
      const auto cener = imap[c::energy].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.e + 1, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = 0.0;
            v.flux(X1DIR, cener, k, j, i) = 0.0;
            if (idx_E.IsValid()) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X1DIR, idx_E(ispec), k, j, i) = 0.0;
              }
            }
          });
    }
  }
  if (ndim == 1) return TaskStatus::complete;

  // x2-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
    if (ix2_bc == "outflow") {
      PackIndexMap imap;
      auto v =
          rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                     std::vector<std::string>({c::density, cr::E}), imap);
      const auto crho = imap[c::density].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = std::min(v.flux(X2DIR, crho, k, j, i), 0.0);
            //            if (idx_E.IsValid()) {
            //              for (int ispec = 0; ispec < num_species; ispec++) {
            //                v.flux(X2DIR, idx_E(ispec), k, j, i) =
            //                    std::min(v.flux(X2DIR, idx_E(ispec), k, j, i), 0.0);
            //              }
            //            }
          });
    } else if (ix2_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density, c::energy, c::momentum, cr::E, cr::F}),
          std::vector<std::string>({c::density, c::energy, c::momentum, cr::E, cr::F}),
          imap);
      const auto crho = imap[c::density].first;
      const auto cener = imap[c::energy].first;
      auto idx_cmom = imap.GetFlatIdx(c::momentum);
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      auto idx_F = imap.GetFlatIdx(cr::F, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = 0.0;
            v.flux(X2DIR, cener, k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(0), k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(2), k, j, i) = 0.0;
            for (int ispec = 0; ispec < num_species; ispec++) {
              v.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
              v.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
              v.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
            }
          });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    if (ox2_bc == "outflow") {
      PackIndexMap imap;
      auto v =
          rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                     std::vector<std::string>({c::density, cr::E}), imap);
      const auto crho = imap[c::density].first;
      auto idx_E = imap.GetFlatIdx(cr::E, false);

      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = std::max(v.flux(X2DIR, crho, k, j, i), 0.0);
            //            if (idx_E.IsValid()) {
            //              for (int ispec = 0; ispec < num_species; ispec++) {
            //                v.flux(X2DIR, idx_E(ispec), k, j, i) =
            //                    std::max(v.flux(X2DIR, idx_E(ispec), k, j, i), 0.0);
            //              }
            //            }
          });
    } else if (ox2_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density, c::energy, c::momentum, cr::E, cr::F}),
          std::vector<std::string>({c::density, c::energy, c::momentum, cr::E, cr::F}),
          imap);
      const auto crho = imap[c::density].first;
      const auto cener = imap[c::energy].first;
      auto idx_cmom = imap.GetFlatIdx(c::momentum);
      auto idx_E = imap.GetFlatIdx(cr::E, false);
      auto idx_F = imap.GetFlatIdx(cr::F, false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = 0.0;
            v.flux(X2DIR, cener, k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(0), k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(2), k, j, i) = 0.0;
            if (idx_E.IsValid()) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
              }
            }
          });
    }
  }

  if (ndim == 2) return TaskStatus::complete;

  // x3-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::outflow) {
    PackIndexMap imap;
    auto v =
        rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                   std::vector<std::string>({c::density, cr::E}), imap);
    const auto crho = imap[c::density].first;
    auto idx_E = imap.GetFlatIdx(cr::E, false);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.s, kb.s, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = std::min(v.flux(X3DIR, crho, k, j, i), 0.0);
          //          if (idx_E.IsValid()) {
          //            for (int ispec = 0; ispec < num_species; ispec++) {
          //              v.flux(X3DIR, idx_E(ispec), k, j, i) =
          //                  std::min(v.flux(X3DIR, idx_E(ispec), k, j, i), 0.0);
          //            }
          //          }
        });
  } else if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto v = rc->PackVariablesAndFluxes(
        std::vector<std::string>({c::density, c::energy, cr::E}),
        std::vector<std::string>({c::density, c::energy, cr::E}), imap);
    const auto crho = imap[c::density].first;
    const auto cener = imap[c::energy].first;
    auto idx_E = imap.GetFlatIdx(cr::E, false);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.s, kb.s, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = 0.0;
          v.flux(X3DIR, cener, k, j, i) = 0.0;
          if (idx_E.IsValid()) {
            for (int ispec = 0; ispec < num_species; ispec++) {
              v.flux(X3DIR, idx_E(ispec), k, j, i) = 0.;
            }
          }
        });
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::outflow) {
    PackIndexMap imap;
    auto v =
        rc->PackVariablesAndFluxes(std::vector<std::string>({c::density, cr::E}),
                                   std::vector<std::string>({c::density, cr::E}), imap);
    const auto crho = imap[c::density].first;
    auto idx_E = imap.GetFlatIdx(cr::E, false);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.e + 1, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = std::max(v.flux(X3DIR, crho, k, j, i), 0.0);
          //          if (idx_E.IsValid()) {
          //            for (int ispec = 0; ispec < num_species; ispec++) {
          //              v.flux(X3DIR, idx_E(ispec), k, j, i) =
          //                  std::max(v.flux(X3DIR, idx_E(ispec), k, j, i), 0.0);
          //            }
          //          }
        });
  } else if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto v = rc->PackVariablesAndFluxes(
        std::vector<std::string>({c::density, c::energy, cr::E}),
        std::vector<std::string>({c::density, c::energy, cr::E}), imap);
    const auto crho = imap[c::density].first;
    const auto cener = imap[c::energy].first;
    auto idx_E = imap.GetFlatIdx(cr::E, false);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.e + 1, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = 0.0;
          v.flux(X3DIR, cener, k, j, i) = 0.0;
          if (idx_E.IsValid()) {
            for (int ispec = 0; ispec < num_species; ispec++) {
              v.flux(X3DIR, idx_E(ispec), k, j, i) = 0.;
            }
          }
        });
  }

  return TaskStatus::complete;
}

template TaskStatus SourceFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

template TaskStatus
RadConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

template TaskStatus
ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace fixup
