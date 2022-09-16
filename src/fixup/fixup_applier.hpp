// © 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef FIXUP_FIXUP_APPLIER_HPP_
#define FIXUP_FIXUP_APPLIER_HPP_

#include <cmath>
#include <limits>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;
using namespace parthenon;

#include <singularity-eos/eos/eos.hpp>

#include "fixup.hpp"
#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

namespace fixup {

const std::vector<std::string> FLUID_VARS = {
    fluid_prim::density,  fluid_prim::energy,      fluid_prim::velocity,
    fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1,
    fluid_prim::bfield,   fluid_prim::ye,          fluid_cons::density,
    fluid_cons::energy,   fluid_cons::momentum,    fluid_cons::bfield,
    fluid_cons::ye,       internal_variables::fail};

/// Convenient access to all fluid quantities
template <typename T>
class FluidAccessor {

 public:
  FluidAccessor(T *rc) : FluidAccessor(rc, PackIndexMap()) {}

  KOKKOS_INLINE_FUNCTION
  Real &prho(const int b, const int k, const int j, const int i) const {
    return v_(b, prho_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &pener(const int b, const int k, const int j, const int i) const {
    return v_(b, pener_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &pvel(const int ii, const int b, const int k, const int j, const int i) const {
    return v_(b, pvel_(ii), k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &prs(const int b, const int k, const int j, const int i) const {
    return v_(b, prs_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &tmp(const int b, const int k, const int j, const int i) const {
    return v_(b, tmp_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &gm1(const int b, const int k, const int j, const int i) const {
    return v_(b, gm1_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &pb(const int ii, const int b, const int k, const int j, const int i) const {
    return v_(b, pb_(ii), k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  bool b_valid() { return pb_.IsValid(); }

  KOKKOS_INLINE_FUNCTION
  bool ye_valid() { return (pye_ >= 0); }

  KOKKOS_INLINE_FUNCTION
  Real &pye(const int b, const int k, const int j, const int i) const {
    return v_(b, pye_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &crho(const int b, const int k, const int j, const int i) const {
    return v_(b, crho_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &cener(const int b, const int k, const int j, const int i) const {
    return v_(b, cener_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &cmom(const int ii, const int b, const int k, const int j, const int i) const {
    return v_(b, cmom_(ii), k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &cb(const int ii, const int b, const int k, const int j, const int i) const {
    return v_(b, cb_(ii), k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &cye(const int b, const int k, const int j, const int i) const {
    return v_(b, cye_, k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &fail(const int b, const int k, const int j, const int i) const {
    return v_(b, fail_, k, j, i);
  }

 private:
  FluidAccessor(T *rc, PackIndexMap imap)
      : v_(rc->PackVariables(FLUID_VARS, imap)), prho_(imap[fluid_prim::density].first),
        pener_(imap[fluid_prim::energy].first),
        pvel_(imap.GetFlatIdx(fluid_prim::velocity)),
        prs_(imap[fluid_prim::pressure].first), tmp_(imap[fluid_prim::temperature].first),
        gm1_(imap[fluid_prim::gamma1].first), pb_(imap.GetFlatIdx(fluid_prim::bfield)),
        pye_(imap[fluid_prim::ye].first), crho_(imap[fluid_cons::density].first),
        cener_(imap[fluid_cons::energy].first),
        cmom_(imap.GetFlatIdx(fluid_cons::momentum)),
        cb_(imap.GetFlatIdx(fluid_cons::bfield)), cye_(imap[fluid_cons::ye].first),
        fail_(imap[internal_variables::fail].first) {}

 protected:
  const VariablePack<Real> v_;
  const int prho_;
  const int pener_;
  const vpack_types::FlatIdx pvel_;
  const int prs_;
  const int tmp_;
  const int gm1_;
  const vpack_types::FlatIdx pb_;
  const int pye_;
  const int crho_;
  const int cener_;
  const vpack_types::FlatIdx cmom_;
  const vpack_types::FlatIdx cb_;
  const int cye_;
  const int fail_;
};

template <typename T>
class LocalFluidAccessor {
 public:
  LocalFluidAccessor(FluidAccessor<T> fld, const int b, const int k, const int j,
                     const int i)
      : fld_(fld), b_(b), k_(k), j_(j), i_(i) {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real &prho() const { return fld_.prho(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &pener() const { return fld_.pener(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &pvel(const int ii) const { return fld_.pvel(ii, b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &prs() const { return fld_.prs(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &tmp() const { return fld_.tmp(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &gm1() const { return fld_.gm1(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &pb(const int ii) const { return fld_.pb(ii, b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &pye() const { return fld_.pye(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &crho() const { return fld_.crho(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &cener() const { return fld_.cener(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &cmom(const int ii) const { return fld_.cmom(ii, b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &cb(const int ii) const { return fld_.cb(ii, b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &cye() const { return fld_.cye(b_, k_, j_, i_); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &fail() const { return fld_.fail(b_, k_, j_, i_); }

  KOKKOS_INLINE_FUNCTION
  bool b_valid() { return fld_.b_valid(); }

  KOKKOS_INLINE_FUNCTION
  bool ye_valid() { return fld_.ye_valid(); }

 private:
  FluidAccessor<T> &fld_;
  const int b_;
  const int k_;
  const int j_;
  const int i_;
};

// TODO(BRR) This won't work for reconstructed vars at faces
template <typename T, typename GEOM, typename C2P>
class BoundsApplier {
 public:
  BoundsApplier(T *rc, Bounds bounds, Coordinates_t coords, GEOM geom,
                singularity::EOS eos, Real c2p_tol, int c2p_maxiter,
                bool enable_mhd_floors, bool enable_rad_floors)
      : bounds_(bounds), coords_(coords), geom_(geom), eos_(eos),
        fld_(FluidAccessor<T>(rc)),
        c2p_(con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_maxiter)),
        enable_mhd_floors_(enable_mhd_floors), enable_rad_floors_(enable_rad_floors) {}
  // BoundsApplier(T *rc, Bounds bounds, FluidAccessor<T> fld)
  //    : bounds_(bounds), fld_(fld) {}

  // TODO(BRR) another class that takes a struct of local geometry to avoid re-evals
  template <class... Args>
  KOKKOS_INLINE_FUNCTION bool ApplyMHDFloors(const int b, const int k, const int j,
                                             const int i) const {

    const Real X1 = coords_.x1v(k, j, i);
    const Real X2 = coords_.x2v(k, j, i);
    const Real X3 = coords_.x3v(k, j, i);

    Real rho_floor, sie_floor, u_floor;
    bounds_.GetFloors(X1, X2, X3, rho_floor, sie_floor);

    Real gamma_ceiling, sie_ceiling;
    bounds_.GetCeilings(X1, X2, X3, gamma_ceiling, sie_ceiling);

    Real bsqorho_ceiling, bsqou_ceiling;
    bounds_.GetMHDCeilings(X1, X2, X3, bsqorho_ceiling, bsqou_ceiling);

    Real rho_floor_max = rho_floor;
    Real u_floor_max = rho_floor * sie_floor;

    Real gcov[4][4];
    geom_.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
    Real gammacon[3][3];
    geom_.MetricInverse(CellLocation::Cent, k, j, i, gammacon);
    const Real alpha = geom_.Lapse(CellLocation::Cent, k, j, i);
    Real betacon[3];
    geom_.ContravariantShift(CellLocation::Cent, k, j, i, betacon);
    const Real sdetgam = geom_.DetGamma(CellLocation::Cent, b, k, j, i);

    LocalFluidAccessor<T> mhd(fld_, b, k, j, i);

    bool mhd_active = mhd.b_valid();
    bool ye_active = mhd.ye_valid();
    // bool rad_active = rad.E_valid();

    bool floor_applied = false;
    bool ceiling_applied = false;

    if (enable_mhd_floors_) {
      Real Bsq = 0.0;
      Real Bdotv = 0.0;
      const Real vp[3] = {mhd.pvel(0), mhd.pvel(1), mhd.pvel(2)};
      const Real bp[3] = {mhd.pb(0), mhd.pb(1), mhd.pb(2)};
      const Real W = phoebus::GetLorentzFactor(vp, gcov);
      const Real iW = 1.0 / W;
      SPACELOOP2(ii, jj) {
        Bsq += gcov[ii + 1][jj + 1] * bp[ii] * bp[jj];
        Bdotv += gcov[ii + 1][jj + 1] * bp[ii] * vp[jj];
      }
      Real bcon0 = W * Bdotv / alpha;
      const Real bsq = (Bsq + alpha * alpha * bcon0 * bcon0) * iW * iW;

      rho_floor_max = std::max<Real>(rho_floor_max, bsq / bsqorho_ceiling);
      u_floor_max = std::max<Real>(u_floor_max, bsq / bsqou_ceiling);

      rho_floor_max = std::max<Real>(
          rho_floor_max, std::max<Real>(mhd.pener(), u_floor_max) / sie_ceiling);
    }

    Real drho = rho_floor_max - mhd.prho();
    Real du = u_floor_max - mhd.pener();

    if (drho > 0. || du > 0.) {
      floor_applied = true;
      if (drho < robust::SMALL()) {
        drho = du / sie_ceiling;
      }
      if (du < robust::SMALL()) {
        du = sie_floor * drho;
      }
    }

    Real dcrho, dS[3], dBcons[3], dtau, dyecons;
    Real bp[3] = {0};
    Real eos_lambda[2] = {0};
    if (mhd.b_valid()) {
      SPACELOOP(ii) { bp[ii] = mhd.pb(ii); }
    }

    if (floor_applied) {
      Real vp_normalobs[3] = {0}; // Inject floors at rest in normal observer frame
      Real ye_prim_default = 0.5;
      eos_lambda[0] = ye_prim_default;
      Real dprs = eos_.PressureFromDensityInternalEnergy(drho, robust::ratio(du, drho),
                                                         eos_lambda);
      Real dgm1 = robust::ratio(eos_.BulkModulusFromDensityInternalEnergy(
                                    drho, robust::ratio(du, drho), eos_lambda),
                                dprs);
      prim2con::p2c(drho, vp_normalobs, bp, du, ye_prim_default, dprs, dgm1, gcov,
                    gammacon, betacon, alpha, sdetgam, dcrho, dS, dBcons, dtau, dyecons);

      // Update cons vars (not B field)
      mhd.crho() += dcrho;
      SPACELOOP(ii) { mhd.cmom(ii) += dS[ii]; }
      mhd.cener() += dtau;
      if (mhd.ye_valid()) {
        mhd.cye() += dyecons;
      }

      // Fluid c2p
      auto status = c2p_(geom_, eos_, coords_, k, j, i);
      if (status == con2prim_robust::ConToPrimStatus::failure) {
        // If fluid c2p fails, set to floors

        mhd.prho() = rho_floor_max;
        SPACELOOP(ii) { mhd.pvel(ii) = vp_normalobs[ii]; }
        mhd.pener() = u_floor_max;
        if (mhd.ye_valid()) {
          mhd.pye() = ye_prim_default;
          eos_lambda[0] = mhd.pye();
        }

        // Update auxiliary primitives
        mhd.tmp() = eos_.TemperatureFromDensityInternalEnergy(
            mhd.prho(), mhd.pener() / mhd.prho(), eos_lambda);
        mhd.prs() = eos_.PressureFromDensityInternalEnergy(
            mhd.prho(), robust::ratio(mhd.pener(), mhd.prho()), eos_lambda);
        mhd.gm1() = robust::ratio(
            eos_.BulkModulusFromDensityInternalEnergy(
                mhd.prho(), robust::ratio(mhd.pener(), mhd.prho()), eos_lambda),
            mhd.prs());
        prim2con::p2c(mhd.prho(), vp_normalobs, bp, mhd.pener(), ye_prim_default,
                      mhd.prs(), mhd.gm1(), gcov, gammacon, betacon, alpha, sdetgam,
                      mhd.crho(), dS, dBcons, mhd.cener(), dyecons);
        SPACELOOP(ii) { mhd.cmom(ii) = dS[ii]; }
        if (mhd.ye_valid()) {
          mhd.cye() = dyecons;
        }
      }
    }

    if (floor_applied || ceiling_applied) {

      // Just update rad here if using?
      return true;
    } else {
      return false;
    }

    return false;
  }

 private:
  const Bounds bounds_;
  const Coordinates_t coords_;
  const GEOM geom_;
  const singularity::EOS eos_;
  const FluidAccessor<T> fld_;
  const C2P c2p_;
  const bool enable_mhd_floors_;
  const bool enable_rad_floors_;
};

} // namespace fixup

#endif // FIXUP_FIXUP_APPLIER_HPP_
