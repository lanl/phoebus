#ifndef FIXUP_FIXUP_HPP_
#define FIXUP_FIXUP_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include <limits>

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus NothingEscapes(MeshBlockData<Real> *rc);
TaskStatus FixFluxes(MeshBlockData<Real> *rc);
TaskStatus FixFailures(MeshBlockData<Real> *rc);
template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc);

static struct ConstantRhoSieFloor {
} constant_rho_sie_floor_tag;
static struct ExpX1RhoSieFloor {
} exp_x1_rho_sie_floor_tag;
static struct ExpX1RhoUFloor { 
} exp_x1_rho_u_floor_tag;

class Floors {
 public:
  Floors() : Floors(constant_rho_sie_floor_tag, -std::numeric_limits<Real>::max(),
                                                -std::numeric_limits<Real>::max()) {}
  Floors(ConstantRhoSieFloor, const Real rho0, const Real sie0)
    : r0_(rho0), s0_(sie0), floor_flag_(1) {}
  Floors(ExpX1RhoSieFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
    : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp), floor_flag_(2) {}
  Floors(ExpX1RhoUFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
    : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp), floor_flag_(3) {
      std::cout << "Initializing floor object!!!" << std::endl;
    }

  KOKKOS_INLINE_FUNCTION
  void GetFloors(const Real x1, const Real x2, const Real x3, Real &rflr, Real &sflr) const {
    Real scratch;
    switch(floor_flag_) {
      case 1:
        rflr = r0_;
        sflr = s0_;
        break;
      case 2:
        rflr = r0_*exp(ralpha_*x1);
        sflr = s0_*exp(salpha_*x1);
        break;
      case 3:
        scratch = r0_*exp(ralpha_*x1);
        sflr = s0_*exp(salpha_*x1)/std::max(rflr,scratch);
        rflr = scratch;
        break;
      default:
        PARTHENON_FAIL("No valid floor set.");
    }
  }

 private:
  Real r0_, s0_, ralpha_, salpha_;
  const int floor_flag_;
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
  void GetCeilings(const Real x1, const Real x2, const Real x3, Real &gmax, Real &smax) const {
    switch(ceiling_flag_) {
      case 1:
        gmax = g0_;
        smax = s0_;
        break;
      default:
        PARTHENON_FAIL("No valid floor set.");
    }
  }

 private:
  Real g0_, s0_;
  const int ceiling_flag_;
};

class Bounds {
 public:
  Bounds() : floors_(Floors()), ceilings_(Ceilings()) {}
  Bounds(const Floors &fl, const Ceilings &cl) : floors_(fl), ceilings_(cl) {}
  explicit Bounds(const Floors &fl) : floors_(fl), ceilings_(Ceilings()) {}
  explicit Bounds(const Ceilings &cl) : floors_(Floors()), ceilings_(cl) {}

  template <class... Args>
  KOKKOS_INLINE_FUNCTION
  void GetFloors(Args &&... args) const {
    floors_.GetFloors(std::forward<Args>(args)...);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION
  void GetCeilings(Args &&... args) const {
    ceilings_.GetCeilings(std::forward<Args>(args)...);
  }
 private:
  const Floors floors_;
  const Ceilings ceilings_;
};


} // namespace fixup

#endif // FIXUP_FIXUP_HPP_