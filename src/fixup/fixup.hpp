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

#ifndef FIXUP_FIXUP_HPP_
#define FIXUP_FIXUP_HPP_

#include <cmath>
#include <limits>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus FixFluxes(MeshBlockData<Real> *rc);
template <typename T>
TaskStatus ApplyFloors(T *rc);
template <typename T>
TaskStatus RadConservedToPrimitiveFixup(T *rc);
template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc);
template <typename T>
TaskStatus SourceFixup(T *rc);

// TODO(BRR) finish this
enum class RadiationFloorFlag { ConstantJ, ExpX1J };
class RadiationFloors {
 public:
  RadiationFloors() {}
};

static struct ConstantRhoSieFloor {
} constant_rho_sie_floor_tag;
static struct ExpX1RhoSieFloor {
} exp_x1_rho_sie_floor_tag;
static struct ExpX1RhoUFloor {
} exp_x1_rho_u_floor_tag;
static struct X1RhoSieFloor {
} x1_rho_sie_floor_tag;
static struct RRhoSieFloor {
} r_rho_sie_floor_tag;

enum class FloorFlag { ConstantRhoSie, ExpX1RhoSie, ExpX1RhoU, X1RhoSie, RRhoSie };

class Floors {
 public:
  Floors()
      : Floors(constant_rho_sie_floor_tag, -std::numeric_limits<Real>::max(),
               -std::numeric_limits<Real>::max()) {}
  Floors(ConstantRhoSieFloor, const Real rho0, const Real sie0)
      : r0_(rho0), s0_(sie0), floor_flag_(FloorFlag::ConstantRhoSie) {}
  Floors(ExpX1RhoSieFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
      : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp),
        floor_flag_(FloorFlag::ExpX1RhoSie) {}
  Floors(ExpX1RhoUFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
      : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp),
        floor_flag_(FloorFlag::ExpX1RhoU) {}
  Floors(X1RhoSieFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
      : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp), floor_flag_(FloorFlag::X1RhoSie) {
  }
  Floors(RRhoSieFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
      : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp), floor_flag_(FloorFlag::RRhoSie) {}

  KOKKOS_INLINE_FUNCTION
  void GetFloors(const Real x1, const Real x2, const Real x3, Real &rflr,
                 Real &sflr) const {
    switch (floor_flag_) {
    case FloorFlag::ConstantRhoSie:
      rflr = r0_;
      sflr = s0_;
      break;
    case FloorFlag::ExpX1RhoSie:
      rflr = r0_ * std::exp(ralpha_ * x1);
      sflr = s0_ * std::exp(salpha_ * x1);
      break;
    case FloorFlag::ExpX1RhoU: {
      Real scratch = r0_ * std::exp(ralpha_ * x1);
      sflr = s0_ * std::exp(salpha_ * x1) / std::max(rflr, scratch);
      rflr = scratch;
    } break;
    case FloorFlag::X1RhoSie:
      rflr = r0_ * std::min(1.0, std::pow(x1, ralpha_));
      sflr = s0_ * std::min(1.0, std::pow(x1, salpha_));
      break;
    case FloorFlag::RRhoSie: {
      Real r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
      rflr = r0_ * std::min(1.0, std::pow(r, ralpha_));
      sflr = s0_ * std::min(1.0, std::pow(r, salpha_));
    } break;
    default:
      PARTHENON_FAIL("No valid floor set.");
    }
  }

 private:
  Real r0_, s0_, ralpha_, salpha_;
  const FloorFlag floor_flag_;
};

static struct ConstantGamSieCeiling {
} constant_gam_sie_ceiling_tag;

class Ceilings {
 public:
  Ceilings()
      : Ceilings(constant_gam_sie_ceiling_tag, std::numeric_limits<Real>::max(),
                 std::numeric_limits<Real>::max()) {}
  Ceilings(ConstantGamSieCeiling, const Real gam0, const Real sie0)
      : g0_(gam0), s0_(sie0), ceiling_flag_(1) {}

  KOKKOS_INLINE_FUNCTION
  void GetCeilings(const Real x1, const Real x2, const Real x3, Real &gmax,
                   Real &smax) const {
    switch (ceiling_flag_) {
    case 1:
      gmax = g0_;
      smax = s0_;
      break;
    default:
      PARTHENON_FAIL("No valid ceiling set.");
    }
  }

 private:
  Real g0_, s0_;
  const int ceiling_flag_;
};

static struct ConstantXi0RadiationCeiling {
} constant_xi0_radiation_ceiling_tag;

class RadiationCeilings {
 public:
  RadiationCeilings()
      : RadiationCeilings(constant_xi0_radiation_ceiling_tag,
                          std::numeric_limits<Real>::max()) {}
  RadiationCeilings(ConstantXi0RadiationCeiling, const Real xi0)
      : xi0_(xi0), radiation_ceiling_flag_(1) {}

  KOKKOS_INLINE_FUNCTION
  void GetRadiationCeilings(const Real x1, const Real x2, const Real x3,
                            Real &ximax) const {
    switch (radiation_ceiling_flag_) {
    case 1:
      ximax = xi0_;
      break;
    default:
      PARTHENON_FAIL("No valid radiation ceiling set.");
    }
  }

 private:
  Real xi0_;
  const int radiation_ceiling_flag_;
};

class Bounds {
 public:
  Bounds()
      : floors_(Floors()), ceilings_(Ceilings()),
        radiation_ceilings_(RadiationCeilings()) {}
  Bounds(const Floors &fl, const Ceilings &cl, const RadiationCeilings &rcl)
      : floors_(fl), ceilings_(cl), radiation_ceilings_(rcl) {}
  explicit Bounds(const Floors &fl)
      : floors_(fl), ceilings_(Ceilings()), radiation_ceilings_(RadiationCeilings()) {}
  explicit Bounds(const Ceilings &cl)
      : floors_(Floors()), ceilings_(cl), radiation_ceilings_(RadiationCeilings()) {}

  template <class... Args>
  KOKKOS_INLINE_FUNCTION void GetFloors(Args &&...args) const {
    floors_.GetFloors(std::forward<Args>(args)...);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION void GetCeilings(Args &&...args) const {
    ceilings_.GetCeilings(std::forward<Args>(args)...);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION void GetRadiationCeilings(Args &&...args) const {
    radiation_ceilings_.GetRadiationCeilings(std::forward<Args>(args)...);
  }

 private:
  const Floors floors_;
  const Ceilings ceilings_;
  const RadiationCeilings radiation_ceilings_;
};

} // namespace fixup

#endif // FIXUP_FIXUP_HPP_
