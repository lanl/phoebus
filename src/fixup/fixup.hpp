#ifndef FIXUP_FIXUP_HPP_
#define FIXUP_FIXUP_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "phoebus_utils/variables.hpp"

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus NothingEscapes(MeshBlockData<Real> *rc);
TaskStatus FixFailures(MeshBlockData<Real> *rc);

static struct ConstantRhoSieFloor {
} constant_rho_sie_floor_tag;
static struct ExpX1RhoSieFloor {
} exp_x1_rho_sie_floor_tag;

class Floors {
 public:
  Floors() : Floors(constant_rho_sie_floor_tag, -1.e300, -1.e300) {}
  Floors(ConstantRhoSieFloor, const Real rho0, const Real sie0)
    : r0_(rho0), s0_(sie0), floor_flag_(1) {}
  Floors(ExpX1RhoSieFloor, const Real rho0, const Real sie0, const Real rp, const Real sp)
    : r0_(rho0), s0_(sie0), ralpha_(rp), salpha_(sp), floor_flag_(2) {}

  KOKKOS_INLINE_FUNCTION
  void GetFloors(const Real x1, const Real x2, const Real x3, Real &rflr, Real &sflr) const {
    switch(floor_flag_) {
      case 1:
        rflr = r0_;
        sflr = s0_;
        break;
      case 2:
        rflr = r0_*exp(ralpha_*x1);
        sflr = s0_*exp(salpha_*x1);
        break;
      default:
        PARTHENON_FAIL("No valid floor set.");
    }
  }

 private:
  Real r0_, s0_, ralpha_, salpha_;
  const int floor_flag_;
};

} // namespace fixup

#endif // FIXUP_FIXUP_HPP_