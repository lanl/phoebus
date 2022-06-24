// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
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

#define FIXUP_PRINT_TOTAL_NFAIL 0

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
  bool enable_ceilings = pin->GetOrAddBoolean("fixup", "enable_ceilings", false);
  params.Add("enable_ceilings", enable_ceilings);
  bool report_c2p_fails = pin->GetOrAddBoolean("fixup", "report_c2p_fails", false);
  params.Add("report_c2p_fails", report_c2p_fails);
  bool enable_mhd_floors = pin->GetOrAddBoolean("fixup", "enable_mhd_floors", false);
  params.Add("enable_mhd_floors", enable_mhd_floors);
  bool enable_rad_floors = pin->GetOrAddBoolean("fixup", "enable_rad_floors", false);
  params.Add("enable_rad_floors", enable_rad_floors);

  if (!pin->GetOrAddBoolean("fluid", "mhd", false)) {
    PARTHENON_REQUIRE(enable_mhd_floors == false,
                      "MHD floors enabled but MHD is disabled!");
  }
  if (!pin->GetOrAddBoolean("physics", "rad", false)) {
    PARTHENON_REQUIRE(enable_rad_floors == false,
                      "Radiation floors enabled but radiation is disabled!");
  }

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

  if (enable_mhd_floors) {
    Real bsqorho_max = pin->GetOrAddReal("fixup", "bsqorho_max", 50.);
    Real bsqou_max = pin->GetOrAddReal("fixup", "bsqou_max", 2500.);
    Real uorho_max = pin->GetOrAddReal("fixup", "uorho_max", 50.);
    params.Add("bsqorho_max", bsqorho_max);
    params.Add("bsqou_max", bsqou_max);
    params.Add("uorho_max", uorho_max);
  }

  params.Add("bounds",
             Bounds(params.Get<Floors>("floor"), params.Get<Ceilings>("ceiling")));

  return fix;
}

template <typename T, class CLOSURE>
TaskStatus ApplyFloorsImpl(T *rc, IndexDomain domain = IndexDomain::entire) {
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

  bool enable_floors = fix_pkg->Param<bool>("enable_floors");
  bool enable_mhd_floors = fix_pkg->Param<bool>("enable_mhd_floors");
  bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");

  if (!enable_floors) return TaskStatus::complete;

  const std::vector<std::string> vars(
      {p::density, c::density, p::velocity, c::momentum, p::energy, c::energy, p::bfield,
       p::ye, c::ye, p::pressure, p::temperature, p::gamma1, pr::J, pr::H, cr::E, cr::F,
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
  vpack_types::FlatIdx idx_J({-1}, -1);
  vpack_types::FlatIdx idx_H({-1}, -1);
  vpack_types::FlatIdx idx_E({-1}, -1);
  vpack_types::FlatIdx idx_F({-1}, -1);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (enable_rad_floors) {
    idx_J = imap.GetFlatIdx(pr::J);
    idx_H = imap.GetFlatIdx(pr::H);
    idx_E = imap.GetFlatIdx(cr::E);
    idx_F = imap.GetFlatIdx(cr::F);
    if (programming::is_specialization_of<CLOSURE, radiation::ClosureMOCMC>::value) {
      iTilPi = imap.GetFlatIdx(ir::tilPi);
    }
  }
  const int nspec = idx_J.DimSize(1);

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

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

        bool floor_applied = false;
        if (v(b, prho, k, j, i) < rho_floor) {
          floor_applied = true;
          v(b, prho, k, j, i) = rho_floor;
        }
        if (v(b, peng, k, j, i) / v(b, prho, k, j, i) < sie_floor) {
          floor_applied = true;
          v(b, peng, k, j, i) = sie_floor * v(b, prho, k, j, i);
        }

        Real gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);

        Real con_vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                          v(b, pvel_lo + 2, k, j, i)};
        const Real W = phoebus::GetLorentzFactor(con_vp, gcov);
        // TODO(BRR) use ceilings
        const Real Wmax = 50.;
        if (W > Wmax) {
          floor_applied = true;
          const Real rescale = std::sqrt((Wmax * Wmax - 1.) / (W * W - 1.));
          SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) *= rescale; }
        }

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

          if (bsq / v(b, prho, k, j, i) > bsqorho_max) {
            floor_applied = true;
            v(b, prho, k, j, i) = bsq / bsqorho_max;
          }
          if (bsq / v(b, peng, k, j, i) > bsqou_max) {
            floor_applied = true;
            v(b, peng, k, j, i) = bsq / bsqou_max;
          }
          if (v(b, peng, k, j, i) / v(b, prho, k, j, i) > uorho_max) {
            floor_applied = true;
            v(b, peng, k, j, i) = uorho_max * v(b, prho, k, j, i);
          }
        }

        if (enable_rad_floors) {
          const Real r = std::exp(coords.x1v(k, j, i));
          const Real Jmin = 1.e-4 * std::pow(r, -4);
          for (int ispec = 0; ispec < nspec; ++ispec) {
            // if (v(b, idx_J(ispec), k, j, i) < 1.e-5 * v(b, peng, k, j, i)) {
            if (v(b, idx_J(ispec), k, j, i) < Jmin) {
              floor_applied = true;
              // printf("Applying rad floor to [%i %i %i]\n", k, j, i);
              // v(b, idx_J(ispec), k, j, i) = 1.e-5 * v(b, peng, k, j, i);
              v(b, idx_J(ispec), k, j, i) = Jmin;
            }

            // Limit magnitude of flux
            const Real Hmag = std::sqrt(std::pow(v(b, idx_H(0, ispec), k, j, i), 2) +
                                        std::pow(v(b, idx_H(1, ispec), k, j, i), 2) +
                                        std::pow(v(b, idx_H(2, ispec), k, j, i), 2));
            if (Hmag > 1.) {
              //              printf("Hmag = %e! [%i %i %i]\n", Hmag, k,j,i);
              //              PARTHENON_FAIL("Hmag");
              floor_applied = true;

              const Real rescale = 0.99 / Hmag;
              SPACELOOP(ii) { v(b, idx_H(ii, ispec), k, j, i) *= rescale; }
            }
          }
        }

        if (floor_applied) {
          // Update dependent primitives
          if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
          v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
              v(b, prho, k, j, i), v(b, peng, k, j, i) / v(b, prho, k, j, i), eos_lambda);
          v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
              v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
          v(b, gm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                                   v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda) /
                               v(b, prs, k, j, i);

          // Update fluid conserved variables
          const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real S[3];
          const Real con_vp[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
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
          prim2con::p2c(v(b, prho, k, j, i), con_vp, bp, v(b, peng, k, j, i), ye_prim,
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

          // Update radiation conserved variables
          Real cov_gamma[3][3];
          geom.Metric(CellLocation::Cent, k, j, i, cov_gamma);
          const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma);
          Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};
          for (int ispec = 0; ispec < nspec; ++ispec) {
            const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);

            typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, 0, b, j, i);
            CLOSURE c(con_v, &g);

            Real E;
            Vec covF;
            Tens2 conTilPi;
            Real J = v(b, idx_J(ispec), k, j, i);
            Vec covH = {{v(b, idx_H(ispec, 0), k, j, i) * J,
                         v(b, idx_H(ispec, 1), k, j, i) * J,
                         v(b, idx_H(ispec, 2), k, j, i) * J}};

            if (programming::is_specialization_of<CLOSURE,
                                                  radiation::ClosureMOCMC>::value) {
              SPACELOOP2(ii, jj) {
                conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetCovTilPiFromPrim(J, covH, &conTilPi);
            }

            c.Prim2Con(J, covH, conTilPi, &E, &covF);

            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * covF(ii); }
          }
        }
      });

  return TaskStatus::complete;
}

template TaskStatus ApplyFloors<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

template <typename T>
TaskStatus ApplyFloors(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad->Param<std::string>("method");
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

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
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

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  if (!enable_c2p_fixup) return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real eos_lambda[2]; // use last temp as initial guess
        eos_lambda[0] = 0.5;
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i));

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
          Real num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
          if (ndim > 1) num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
          if (ndim == 3) num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);
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
            v(b, gm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                v(b, prho, k, j, i), ratio(v(b, tmp, k, j, i), v(b, prs, k, j, i)),
                eos_lambda);
          } else {
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
            v(b, gm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                v(b, prho, k, j, i), ratio(v(b, tmp, k, j, i), v(b, prs, k, j, i)),
                eos_lambda);

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

TaskStatus FixFluxes(MeshBlockData<Real> *rc) {
  using parthenon::BoundaryFace;
  using parthenon::BoundaryFlag;
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fixup")->Param<bool>("enable_flux_fixup"))
    return TaskStatus::complete;

  auto fluid = pmb->packages.Get("fluid");
  const std::string ix1_bc = fluid->Param<std::string>("ix1_bc");
  const std::string ox1_bc = fluid->Param<std::string>("ox1_bc");
  const std::string ix2_bc = fluid->Param<std::string>("ix2_bc");
  const std::string ox2_bc = fluid->Param<std::string>("ox2_bc");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int ndim = pmb->pmy_mesh->ndim;

  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace cr = radmoment_cons;

  // x1-direction
  printf("bflags: %i %i %i %i\n",
         static_cast<int>(pmb->boundary_flag[BoundaryFace::inner_x1]),
         static_cast<int>(pmb->boundary_flag[BoundaryFace::outer_x1]),
         static_cast<int>(pmb->boundary_flag[BoundaryFace::inner_x2]),
         static_cast<int>(pmb->boundary_flag[BoundaryFace::outer_x2]));
  printf("ix1_bc: %s ox1_bc: %s\n", ix1_bc.c_str(), ox1_bc.c_str());
  if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
    if (ix1_bc == "outflow") {
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, radmoment_cons::E}),
          std::vector<std::string>({fluid_cons::density, radmoment_cons::E}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X1DIR, 0, k, j, i) = std::min(flux.flux(X1DIR, 0, k, j, i), 0.0);
            // TODO(BRR) fix flux for radiation energy?
            flux.flux(X1DIR, 1, k, j, i) = std::max(flux.flux(X1DIR, 0, k, j, i), 0.0);
          });
    } else if (ix1_bc == "reflect") {
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy, cr::E}),
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy, cr::E}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X1DIR, 0, k, j, i) = 0.0;
            flux.flux(X1DIR, 1, k, j, i) = 0.0;
            flux.flux(X1DIR, 2, k, j, i) = 0.0;
          });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    if (ox1_bc == "outflow") {
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, cr::E}),
          std::vector<std::string>({fluid_cons::density, cr::E}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.e + 1, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X1DIR, 0, k, j, i) = std::max(flux.flux(X1DIR, 0, k, j, i), 0.0);
            flux.flux(X1DIR, 1, k, j, i) = std::max(flux.flux(X1DIR, 0, k, j, i), 0.0);
          });
    } else if (ox1_bc == "reflect") {
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy, cr::E}),
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy, cr::E}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.e + 1, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X1DIR, 0, k, j, i) = 0.0;
            flux.flux(X1DIR, 1, k, j, i) = 0.0;
            flux.flux(X1DIR, 2, k, j, i) = 0.0;
          });
    }
  }
  if (ndim == 1) return TaskStatus::complete;

  // x2-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
    if (ix2_bc == "outflow") {
      auto flux =
          rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                     std::vector<std::string>({fluid_cons::density}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X2DIR, 0, k, j, i) = std::min(flux.flux(X2DIR, 0, k, j, i), 0.0);
          });
    } else if (ix2_bc == "reflect") {
      PackIndexMap imap;
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy,
                                    fluid_cons::momentum, radmoment_cons::E,
                                    radmoment_cons::F}),
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy,
                                    fluid_cons::momentum, radmoment_cons::E,
                                    radmoment_cons::F}),
          imap);
      const int cmom_lo = imap[c::momentum].first;
      const int cmom_hi = imap[c::momentum].second;
      auto idx_E = imap.GetFlatIdx(cr::E);
      auto idx_F = imap.GetFlatIdx(cr::F);
      int nspec = idx_E.DimSize(1);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X2DIR, 0, k, j, i) = 0.0;
            flux.flux(X2DIR, 1, k, j, i) = 0.0;
            flux.flux(X2DIR, cmom_lo, k, j, i) = 0.0;
            flux.flux(X2DIR, cmom_lo + 2, k, j, i) = 0.0;
            for (int ispec = 0; ispec < nspec; ispec++) {
              flux.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
              flux.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
              flux.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
            }
          });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    if (ox2_bc == "outflow") {
      auto flux =
          rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                     std::vector<std::string>({fluid_cons::density}));
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X2DIR, 0, k, j, i) = std::max(flux.flux(X2DIR, 0, k, j, i), 0.0);
          });
    } else if (ox2_bc == "reflect") {
      PackIndexMap imap;
      auto flux = rc->PackVariablesAndFluxes(
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy,
                                    fluid_cons::momentum, cr::E, cr::F}),
          std::vector<std::string>({fluid_cons::density, fluid_cons::energy,
                                    fluid_cons::momentum, cr::E, cr::F}),
          imap);
      const int cmom_lo = imap[c::momentum].first;
      const int cmom_hi = imap[c::momentum].second;
      auto idx_E = imap.GetFlatIdx(cr::E);
      auto idx_F = imap.GetFlatIdx(cr::F);
      int nspec = idx_E.DimSize(1);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            flux.flux(X2DIR, 0, k, j, i) = 0.0;
            flux.flux(X2DIR, 1, k, j, i) = 0.0;
            flux.flux(X2DIR, cmom_lo, k, j, i) = 0.0;
            flux.flux(X2DIR, cmom_lo + 2, k, j, i) = 0.0;
            for (int ispec = 0; ispec < nspec; ispec++) {
              flux.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
              flux.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
              flux.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
            }
          });
    }
  }

  if (ndim == 2) return TaskStatus::complete;

  // x3-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::outflow) {
    auto flux =
        rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                   std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.s, kb.s, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X3DIR, 0, k, j, i) = std::min(flux.flux(X3DIR, 0, k, j, i), 0.0);
        });
  } else if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::reflect) {
    auto flux = rc->PackVariablesAndFluxes(
        std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
        std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.s, kb.s, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X3DIR, 0, k, j, i) = 0.0;
          flux.flux(X3DIR, 1, k, j, i) = 0.0;
        });
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::outflow) {
    auto flux =
        rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                   std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.e + 1, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X3DIR, 0, k, j, i) = std::max(flux.flux(X3DIR, 0, k, j, i), 0.0);
        });
  } else if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    auto flux = rc->PackVariablesAndFluxes(
        std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
        std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.e + 1, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X3DIR, 0, k, j, i) = 0.0;
          flux.flux(X3DIR, 1, k, j, i) = 0.0;
        });
  }

  return TaskStatus::complete;
}

template TaskStatus
ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace fixup
