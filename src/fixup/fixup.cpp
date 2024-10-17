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

#include <defs.hpp>

#include "analysis/history.hpp"
#include "fluid/con2prim_robust.hpp"
#include "fluid/fluid.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/reduction.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/radiation.hpp"
#include <parthenon/package.hpp>

using Microphysics::RadiationType;
using Microphysics::EOS::EOS;
using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto fix = std::make_shared<StateDescriptor>("fixup");
  Params &params = fix->AllParams();
  auto mutable_param = parthenon::Params::Mutability::Mutable;

  const bool enable_mhd = pin->GetOrAddBoolean("fluid", "mhd", false);
  const bool enable_rad = pin->GetOrAddBoolean("physics", "rad", false);

  bool enable_flux_fixup = pin->GetOrAddBoolean("fixup", "enable_flux_fixup", false);
  params.Add("enable_flux_fixup", enable_flux_fixup);
  bool enable_ox1_fmks_inflow_check =
      pin->GetOrAddBoolean("fixup", "enable_ox1_fmks_inflow_check", false);
  params.Add("enable_ox1_fmks_inflow_check", enable_ox1_fmks_inflow_check);
  bool enable_fixup = pin->GetOrAddBoolean("fixup", "enable_fixup", false);
  params.Add("enable_fixup", enable_fixup);
  bool enable_floors = pin->GetOrAddBoolean("fixup", "enable_floors", false);
  bool enable_c2p_fixup = pin->GetOrAddBoolean("fixup", "enable_c2p_fixup", false);
  params.Add("enable_c2p_fixup", enable_c2p_fixup);
  bool enable_source_fixup = pin->GetOrAddBoolean("fixup", "enable_source_fixup", false);
  params.Add("enable_source_fixup", enable_source_fixup);
  bool enable_ceilings = pin->GetOrAddBoolean("fixup", "enable_ceilings", false);
  params.Add("enable_ceilings", enable_ceilings);
  bool report_c2p_fails = pin->GetOrAddBoolean("fixup", "report_c2p_fails", false);
  params.Add("report_c2p_fails", report_c2p_fails);
  bool report_source_fails = pin->GetOrAddBoolean("fixup", "report_source_fails", false);
  params.Add("report_source_fails", report_source_fails);
  bool enable_mhd_floors = pin->GetOrAddBoolean("fixup", "enable_mhd_floors", false);
  bool enable_rad_floors = pin->GetOrAddBoolean("fixup", "enable_rad_floors", false);
  bool enable_rad_ceilings = pin->GetOrAddBoolean("fixup", "enable_rad_ceilings", false);
  bool enable_phi_enforcement =
      pin->GetOrAddBoolean("fixup", "enable_phi_enforcement", false);
  params.Add("enable_phi_enforcement", enable_phi_enforcement);
  if (enable_phi_enforcement) {
    PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
                      "Phi enforcement only supported for BH geometry!");
    Real enforced_phi = pin->GetReal("fixup", "enforced_phi");
    params.Add("enforced_phi", enforced_phi);
    Real enforced_phi_timescale =
        pin->GetOrAddReal("fixup", "enforced_phi_timescale", 1.e3);
    params.Add("enforced_phi_timescale", enforced_phi_timescale);
    Real enforced_phi_cadence = pin->GetOrAddReal("fixup", "enforced_phi_cadence", 10.);
    params.Add("enforced_phi_cadence", enforced_phi_cadence);
    Real enforced_phi_start_time =
        pin->GetOrAddReal("fixup", "enforced_phi_start_time", 175.);
    PARTHENON_REQUIRE(enforced_phi_start_time >= 0.,
                      "Phi enforcement start time must be non-negative!");
    params.Add("enforced_phi_start_time", enforced_phi_start_time);

    // Mutable parameters for following sanity maintenance
    params.Add("dphi_dt", 0., true);
    params.Add("next_dphi_dt_update_time", enforced_phi_start_time, true);
    params.Add("phi_factor", 0., true);
  }

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

  if (enable_rad_ceilings && !enable_rad) {
    enable_rad_ceilings = false;
    PARTHENON_WARN("WARNING Disabling radiation ceilings because radiation is disabled!");
  }
  params.Add("enable_rad_ceilings", enable_rad_ceilings);

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

  if (enable_rad_floors) {
    const std::string floor_type = pin->GetString("fixup", "rad_floor_type");
    if (floor_type == "ConstantJ") {
      Real J0 = pin->GetOrAddReal("fixup", "J0_floor", 0.0);
      params.Add("rad_floor", RadiationFloors(constant_j_floor_tag, J0));
    } else if (floor_type == "ExpX1J") {
      Real J0 = pin->GetOrAddReal("fixup", "J0_floor", 0.0);
      Real Jp = pin->GetOrAddReal("fixup", "J_exp_floor", -2.0);
      params.Add("rad_floor", RadiationFloors(exp_x1_j_floor_tag, J0, Jp));
    } else {
      PARTHENON_FAIL("invalid <fixup>/rad_floor_type input");
    }
  } else {
    params.Add("rad_floor", RadiationFloors());
  }

  if (enable_rad_ceilings) {
    const std::string radiation_ceiling_type =
        pin->GetOrAddString("fixup", "radiation_ceiling_type", "ConstantXi0");
    if (radiation_ceiling_type == "ConstantXi0") {
      Real xi0 = pin->GetOrAddReal("fixup", "xi0_ceiling", 0.99);
      Real tau0 = pin->GetOrAddReal("fixup", "tau0_ceiling", 1.e5);
      params.Add("rad_ceiling",
                 RadiationCeilings(constant_xi0_radiation_ceiling_tag, xi0, tau0));
    }
  } else {
    params.Add("rad_ceiling", RadiationCeilings());
  }

  if (enable_mhd_floors) {
    const std::string mhd_ceiling_type =
        pin->GetOrAddString("fixup", "mhd_ceiling_type", "ConstantBsqRat");
    if (mhd_ceiling_type == "ConstantBsqRat") {
      Real bsqorho0 = pin->GetOrAddReal("fixup", "bsqorho0_ceiling", 50.);
      Real bsqou0 = pin->GetOrAddReal("fixup", "bsqou0_ceiling", 2500.);
      params.Add("mhd_ceiling",
                 MHDCeilings(constant_bsq_rat_ceiling_tag, bsqorho0, bsqou0));
    } else {
      PARTHENON_FAIL("invalid <fixup>/mhd_ceiling_type input");
    }
  } else {
    params.Add("mhd_ceiling", MHDCeilings());
  }

  FAILURE_STRATEGY fluid_c2p_failure_strategy;
  std::string fluid_c2p_failure_strategy_str =
      pin->GetOrAddString("fixup", "fluid_c2p_failure_strategy", "interpolate");
  if (fluid_c2p_failure_strategy_str == "interpolate") {
    fluid_c2p_failure_strategy = FAILURE_STRATEGY::interpolate;
  } else if (fluid_c2p_failure_strategy_str == "floors") {
    fluid_c2p_failure_strategy = FAILURE_STRATEGY::floors;
  } else {
    PARTHENON_FAIL("fixup/fluid_c2p_failure_strategy not supported!");
  }
  params.Add("fluid_c2p_failure_strategy", fluid_c2p_failure_strategy);

  FAILURE_STRATEGY rad_c2p_failure_strategy;
  std::string rad_c2p_failure_strategy_str =
      pin->GetOrAddString("fixup", "rad_c2p_failure_strategy", "interpolate");
  if (rad_c2p_failure_strategy_str == "interpolate") {
    rad_c2p_failure_strategy = FAILURE_STRATEGY::interpolate;
  } else if (rad_c2p_failure_strategy_str == "floors") {
    rad_c2p_failure_strategy = FAILURE_STRATEGY::floors;
  } else {
    PARTHENON_FAIL("fixup/rad_c2p_failure_strategy not supported!");
  }
  params.Add("rad_c2p_failure_strategy", rad_c2p_failure_strategy);

  FAILURE_STRATEGY src_failure_strategy;
  std::string src_failure_strategy_str =
      pin->GetOrAddString("fixup", "src_failure_strategy", "interpolate");
  if (src_failure_strategy_str == "interpolate") {
    src_failure_strategy = FAILURE_STRATEGY::interpolate;
  } else if (src_failure_strategy_str == "floors") {
    src_failure_strategy = FAILURE_STRATEGY::floors;
  } else {
    PARTHENON_FAIL("fixup/src_c2p_failure_strategy not supported!");
  }
  params.Add("src_failure_strategy", src_failure_strategy);

  const bool c2p_failure_force_fixup_both =
      pin->GetOrAddBoolean("fixup", "c2p_failure_force_fixup_both", false);
  params.Add("c2p_failure_force_fixup_both", c2p_failure_force_fixup_both);

  params.Add("bounds",
             Bounds(params.Get<Floors>("floor"), params.Get<Ceilings>("ceiling"),
                    params.Get<MHDCeilings>("mhd_ceiling"),
                    params.Get<RadiationFloors>("rad_floor"),
                    params.Get<RadiationCeilings>("rad_ceiling")),
             mutable_param);

  return fix;
}

/// Given a valid state (including consistency between prim and cons variables),
/// this function returns another valid state that is within the bounds of specified
/// floors and ceilings.
template <typename T, class CLOSURE>
TaskStatus ApplyFloorsImpl(T *rc, IndexDomain domain = IndexDomain::entire) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;
  namespace ir = radmoment_internal;
  namespace impl = internal_variables;
  auto *pmesh = rc->GetMeshPointer();
  IndexRange ib = rc->GetBoundsI(domain);
  IndexRange jb = rc->GetBoundsJ(domain);
  IndexRange kb = rc->GetBoundsK(domain);

  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmesh->packages.Get("eos").get();
  StateDescriptor *fluid_pkg = pmesh->packages.Get("fluid").get();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();

  bool rad_active = rad_pkg->Param<bool>("active");

  bool enable_floors = fix_pkg->Param<bool>("enable_floors");
  bool enable_mhd_floors = fix_pkg->Param<bool>("enable_mhd_floors");
  bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");

  if (!enable_floors) return TaskStatus::complete;

  const Real ye_min = eos_pkg->Param<Real>("ye_min");
  const Real ye_max = eos_pkg->Param<Real>("ye_max");

  const std::vector<std::string> vars(
      {p::density::name(), c::density::name(), p::velocity::name(), c::momentum::name(),
       p::energy::name(), c::energy::name(), p::bfield::name(), p::ye::name(),
       c::ye::name(), p::pressure::name(), p::temperature::name(), p::gamma1::name(),
       pr::J::name(), pr::H::name(), cr::E::name(), cr::F::name(),
       impl::cell_signal_speed::name(), impl::fail::name(), ir::tilPi::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density::name()].first;
  const int crho = imap[c::density::name()].first;
  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int cmom_lo = imap[c::momentum::name()].first;
  const int cmom_hi = imap[c::momentum::name()].second;
  const int peng = imap[p::energy::name()].first;
  const int ceng = imap[c::energy::name()].first;
  const int prs = imap[p::pressure::name()].first;
  const int tmp = imap[p::temperature::name()].first;
  const int gm1 = imap[p::gamma1::name()].first;
  const int slo = imap[impl::cell_signal_speed::name()].first;
  const int shi = imap[impl::cell_signal_speed::name()].second;
  const int pb_lo = imap[p::bfield::name()].first;
  const int pb_hi = imap[p::bfield::name()].second;
  int pye = imap[p::ye::name()].second; // negative if not present
  int cye = imap[c::ye::name()].second;
  auto idx_J = imap.GetFlatIdx(pr::J::name(), false);
  auto idx_H = imap.GetFlatIdx(pr::H::name(), false);
  auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
  auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name(), false);

  const int num_species = enable_rad_floors ? rad_pkg->Param<int>("num_species") : 0;

  auto eos = eos_pkg->Param<EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  Bounds *pbounds = fix_pkg->MutableParam<Bounds>("bounds");

  // BLB: Setting EOS bnds for Ceilings/Floors here.
  pbounds->SetEOSBnds(eos_pkg);

  // copy bounds by value for kernel
  Bounds bounds = *pbounds;

  const Real c2p_tol = fluid_pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = fluid_pkg->Param<int>("c2p_max_iter");
  const Real c2p_floor_scale_fac = fluid_pkg->Param<Real>("c2p_floor_scale_fac");
  const bool c2p_fail_on_floors = fluid_pkg->Param<bool>("c2p_fail_on_floors");
  const bool c2p_fail_on_ceilings = fluid_pkg->Param<bool>("c2p_fail_on_ceilings");
  auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter,
                                                c2p_floor_scale_fac, c2p_fail_on_floors,
                                                c2p_fail_on_ceilings);

  Coordinates_t coords = rc->GetParentPointer()->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyFloors", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        double eos_lambda[2]; // used for stellarcollapse eos and
                              // other EOS's that require root
                              // finding.
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i)); // use last temp as initial guess

        double rho_floor, sie_floor;
        bounds.GetFloors(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                         coords.Xc<3>(k, j, i), rho_floor, sie_floor);
        double gamma_max, e_max;
        bounds.GetCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                           coords.Xc<3>(k, j, i), gamma_max, e_max);
        Real bsqorho_max, bsqou_max;
        bounds.GetMHDCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                              coords.Xc<3>(k, j, i), bsqorho_max, bsqou_max);
        Real J_floor;
        bounds.GetRadiationFloors(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                                  coords.Xc<3>(k, j, i), J_floor);
        Real xi_max;
        Real garbage;
        bounds.GetRadiationCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                                    coords.Xc<3>(k, j, i), xi_max, garbage);

        Real rho_floor_max = rho_floor;
        Real u_floor_max = rho_floor * sie_floor;

        bool floor_applied = false;
        bool ceiling_applied = false;

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
              rho_floor_max, std::max<Real>(v(b, peng, k, j, i), u_floor_max) / e_max);
        }

        Real drho = rho_floor_max - v(b, prho, k, j, i);
        Real du = u_floor_max - v(b, peng, k, j, i);
        if (drho > 0. || du > 0.) {
          floor_applied = true;
          drho = std::max<Real>(drho, du / sie_floor);
          du = std::max<Real>(du, sie_floor * drho);
        }

        Real dcrho, dS[3], dBcons[3], dtau, dyecons;
        Real bp[3] = {0};
        if (pb_hi > 0) {
          SPACELOOP(ii) { bp[ii] = v(b, pb_lo + ii, k, j, i); }
        }

        if (floor_applied) {
          Real vp_normalobs[3] = {0}; // Inject floors at rest in normal observer frame
          Real ye = 0.5;
          if (pye > 0) {
            ye = v(b, cye, k, j, i);
          }
          eos_lambda[0] = ye;
          Real dprs =
              eos.PressureFromDensityInternalEnergy(drho, ratio(du, drho), eos_lambda);
          Real dgm1 = ratio(
              eos.BulkModulusFromDensityInternalEnergy(drho, ratio(du, drho), eos_lambda),
              dprs);
          prim2con::p2c(drho, vp_normalobs, bp, du, ye, dprs, dgm1, gcov, gammacon,
                        betacon, alpha, sdetgam, dcrho, dS, dBcons, dtau, dyecons);

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
              v(b, pye, k, j, i) = ye_min;
            }

            // Update auxiliary primitives
            if (pye > 0) {
              eos_lambda[0] = v(b, pye, k, j, i);
            }
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), v(b, peng, k, j, i) / v(b, prho, k, j, i),
                eos_lambda);

            // Update cons vars (not B field)
            prim2con::p2c(drho, vp_normalobs, bp, du, ye, dprs, dgm1, gcov, gammacon,
                          betacon, alpha, sdetgam, v(b, crho, k, j, i), dS, dBcons,
                          v(b, ceng, k, j, i), dyecons);
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
          SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = vpcon[ii]; }

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

        if (floor_applied || ceiling_applied) {
          // Update derived prims
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
                c.GetConTilPiFromCon(E, cov_F, xi, phi, &con_TilPi);
              }

              c.Con2Prim(E, cov_F, con_TilPi, &J, &cov_H);
              v(b, idx_J(ispec), k, j, i) = J;
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J; }
            }
          }

          // TODO(BRR) use Bounds class
          Vec con_v_normalobs{0};
          CLOSURE c_iso(con_v_normalobs, &g);
          Tens2 con_tilPi_iso{0};
          for (int ispec = 0; ispec < num_species; ++ispec) {
            Real dJ = J_floor - v(b, idx_J(ispec), k, j, i);
            if (dJ > 0.) {

              constexpr bool update_cons_vars = true; // false;
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
                  c.GetConTilPiFromCon(E, cov_F, xi, phi, &con_TilPi);
                }

                c.Con2Prim(E, cov_F, con_TilPi, &J, &cov_H);

                v(b, idx_J(ispec), k, j, i) = J;
                SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J; }
              } else {
                v(b, idx_J(ispec), k, j, i) += dJ;

                J = v(b, idx_J(ispec), k, j, i);
                SPACELOOP(ii) { cov_H(ii) = v(b, idx_H(ispec, ii), k, j, i) * J; }

                // We need the real conTilPi
                if (iTilPi.IsValid()) {
                  SPACELOOP2(ii, jj) {
                    con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                  }
                } else {
                  c.GetConTilPiFromPrim(J, cov_H, &con_TilPi);
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

            Vec cov_H{v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                      v(b, idx_H(ispec, 2), k, j, i)};
            const Real xi =
                std::sqrt(g.contractCov3Vectors(cov_H, cov_H) -
                          std::pow(g.contractConCov3Vectors(con_v, cov_H), 2));

            if (xi > xi_max) {

              J = v(b, idx_J(ispec), k, j, i);
              SPACELOOP(ii) {
                cov_H(ii) = (xi_max / xi) * v(b, idx_H(ispec, ii), k, j, i) * J;
                v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J;
              }

              // We need the real conTilPi
              if (iTilPi.IsValid()) {
                SPACELOOP2(ii, jj) {
                  con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                }
              } else {
                c.GetConTilPiFromPrim(J, cov_H, &con_TilPi);
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
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ApplyFloors(T *rc) {
  auto *pm = rc->GetMeshPointer();
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
    return ApplyFloorsImpl<T, radiation::ClosureM1<settings>>(rc);
  } else if (method == "moment_eddington") {
    return ApplyFloorsImpl<T, radiation::ClosureEdd<settings>>(rc);
  } else if (method == "mocmc") {
    return ApplyFloorsImpl<T, radiation::ClosureMOCMC<settings>>(rc);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return ApplyFloorsImpl<T, radiation::ClosureEdd<settings>>(rc);
  }
  return TaskStatus::complete;
}

template TaskStatus ApplyFloors<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

TaskStatus FixFluxes(MeshBlockData<Real> *rc) {
  using parthenon::BoundaryFace;
  using parthenon::BoundaryFlag;
  auto *pmb = rc->GetParentPointer();
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
  auto moments_active = rad->Param<bool>("moments_active");

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  const int ndim = pmb->pmy_mesh->ndim;
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace cr = radmoment_cons;

  // x1-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
    if (ix1_bc == "outflow") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                          std::vector<std::string>({c::density::name()}),
                                          imap);
      const auto crho = imap[c::density::name()].first;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = std::min(v.flux(X1DIR, crho, k, j, i), 0.0);
          });
    } else if (ix1_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>(
              {c::density::name(), c::energy::name(), cr::E::name()}),
          std::vector<std::string>(
              {c::density::name(), c::energy::name(), cr::E::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
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
      auto v = rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                          std::vector<std::string>({c::density::name()}),
                                          imap);
      const auto crho = imap[c::density::name()].first;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
          ib.e + 1, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X1DIR, crho, k, j, i) = std::max(v.flux(X1DIR, crho, k, j, i), 0.0);
          });
    } else if (ox1_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>(
              {c::density::name(), c::energy::name(), cr::E::name()}),
          std::vector<std::string>(
              {c::density::name(), c::energy::name(), cr::E::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
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
      auto v = rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                          std::vector<std::string>({c::density::name()}),
                                          imap);
      const auto crho = imap[c::density::name()].first;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = std::min(v.flux(X2DIR, crho, k, j, i), 0.0);
          });
    } else if (ix2_bc == "polar") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), c::bfield::name(), cr::E::name(),
                                    cr::F::name()}),
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), c::bfield::name(), cr::E::name(),
                                    cr::F::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_cb = imap.GetFlatIdx(c::bfield::name(), false);
      auto idx_cmom = imap.GetFlatIdx(c::momentum::name());
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
      auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = 0.0;
            v.flux(X2DIR, cener, k, j, i) = 0.0;
            SPACELOOP(ii) {
              v.flux(X2DIR, idx_cmom(ii), k, j, i) = 0.0;
              v.flux(X2DIR, idx_cb(ii), k, j, i) = 0.0;
            }
            v.flux(X1DIR, idx_cb(1), k, j - 1, i) = -v.flux(X1DIR, idx_cb(1), k, j, i);
            if (ndim == 3) {
              v.flux(X3DIR, idx_cb(1), k, j - 1, i) = -v.flux(X3DIR, idx_cb(1), k, j, i);
            }

            if (idx_E.IsValid()) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
              }
            }
          });
    } else if (ix2_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), cr::E::name(), cr::F::name()}),
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), cr::E::name(), cr::F::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_cb = imap.GetFlatIdx(c::bfield::name(), false);
      auto idx_cmom = imap.GetFlatIdx(c::momentum::name());
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
      auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.s, jb.s,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = 0.0;
            v.flux(X2DIR, cener, k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(0), k, j, i) = 0.0;
            v.flux(X2DIR, idx_cmom(2), k, j, i) = 0.0;

            if (moments_active) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
              }
            }
          });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    if (ox2_bc == "outflow") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                          std::vector<std::string>({c::density::name()}),
                                          imap);
      const auto crho = imap[c::density::name()].first;

      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = std::max(v.flux(X2DIR, crho, k, j, i), 0.0);
          });
    } else if (ox2_bc == "polar") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), c::bfield::name(), cr::E::name(),
                                    cr::F::name()}),
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), c::bfield::name(), cr::E::name(),
                                    cr::F::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_cb = imap.GetFlatIdx(c::bfield::name(), false);
      auto idx_cmom = imap.GetFlatIdx(c::momentum::name());
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
      auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(), kb.s, kb.e, jb.e + 1,
          jb.e + 1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v.flux(X2DIR, crho, k, j, i) = 0.0;
            v.flux(X2DIR, cener, k, j, i) = 0.0;
            SPACELOOP(ii) {
              v.flux(X2DIR, idx_cmom(ii), k, j, i) = 0.0;
              v.flux(X2DIR, idx_cb(ii), k, j, i) = 0.0;
            }
            v.flux(X1DIR, idx_cb(1), k, j, i) = -v.flux(X1DIR, idx_cb(1), k, j - 1, i);
            if (ndim == 3) {
              v.flux(X3DIR, idx_cb(1), k, j, i) = -v.flux(X3DIR, idx_cb(1), k, j - 1, i);
            }

            if (idx_E.IsValid()) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                v.flux(X2DIR, idx_E(ispec), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 0), k, j, i) = 0.0;
                v.flux(X2DIR, idx_F(ispec, 2), k, j, i) = 0.0;
              }
            }
          });
    } else if (ox2_bc == "reflect") {
      PackIndexMap imap;
      auto v = rc->PackVariablesAndFluxes(
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), cr::E::name(), cr::F::name()}),
          std::vector<std::string>({c::density::name(), c::energy::name(),
                                    c::momentum::name(), cr::E::name(), cr::F::name()}),
          imap);
      const auto crho = imap[c::density::name()].first;
      const auto cener = imap[c::energy::name()].first;
      auto idx_cmom = imap.GetFlatIdx(c::momentum::name());
      auto idx_cb = imap.GetFlatIdx(c::bfield::name(), false);
      auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
      auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
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
        rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                   std::vector<std::string>({c::density::name()}), imap);
    const auto crho = imap[c::density::name()].first;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.s, kb.s, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = std::min(v.flux(X3DIR, crho, k, j, i), 0.0);
        });
  } else if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto v = rc->PackVariablesAndFluxes(
        std::vector<std::string>({c::density::name(), c::energy::name(), cr::E::name()}),
        std::vector<std::string>({c::density::name(), c::energy::name(), cr::E::name()}),
        imap);
    const auto crho = imap[c::density::name()].first;
    const auto cener = imap[c::energy::name()].first;
    auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
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
        rc->PackVariablesAndFluxes(std::vector<std::string>({c::density::name()}),
                                   std::vector<std::string>({c::density::name()}), imap);
    const auto crho = imap[c::density::name()].first;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(), kb.e + 1, kb.e + 1, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v.flux(X3DIR, crho, k, j, i) = std::max(v.flux(X3DIR, crho, k, j, i), 0.0);
        });
  } else if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto v = rc->PackVariablesAndFluxes(
        std::vector<std::string>({c::density::name(), c::energy::name(), cr::E::name()}),
        std::vector<std::string>({c::density::name(), c::energy::name(), cr::E::name()}),
        imap);
    const auto crho = imap[c::density::name()].first;
    const auto cener = imap[c::energy::name()].first;
    auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
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

} // namespace fixup
