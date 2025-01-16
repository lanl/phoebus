// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#ifndef _PGEN_H_
#define _PGEN_H_

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// internal includes
#include "fluid/fluid.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/radiation.hpp"
#include "tracers/tracers.hpp"

// add the name of a namespace that contains your new ProblemGenerator
#define FOREACH_PROBLEM                                                                  \
  PROBLEM(phoebus)                                                                       \
  PROBLEM(advection)                                                                     \
  PROBLEM(check_cached_geom)                                                             \
  PROBLEM(shock_tube)                                                                    \
  PROBLEM(friedmann)                                                                     \
  PROBLEM(linear_modes)                                                                  \
  PROBLEM(thin_cooling)                                                                  \
  PROBLEM(leptoneq)                                                                      \
  PROBLEM(kelvin_helmholtz)                                                              \
  PROBLEM(rhs_tester)                                                                    \
  PROBLEM(sedov)                                                                         \
  PROBLEM(blandford_mckee)                                                               \
  PROBLEM(bondi)                                                                         \
  PROBLEM(radiation_advection)                                                           \
  PROBLEM(radiation_equilibration)                                                       \
  PROBLEM(rotor)                                                                         \
  PROBLEM(homogeneous_sphere)                                                            \
  PROBLEM(torus)                                                                         \
  PROBLEM(p2c2p)                                                                         \
  PROBLEM(tov)                                                                           \
  PROBLEM(homologous)

// if you need problem-specific modifications to inputs, add the name here
#define FOREACH_MODIFIER                                                                 \
  MODIFIER(phoebus)                                                                      \
  MODIFIER(torus)                                                                        \
  MODIFIER(radiation_advection)

// if you need problem-specific post-initialization modifiers to initial conditions, add
// the name here
#define FOREACH_POSTINIT_MODIFIER                                                        \
  POSTINIT_MODIFIER(phoebus)                                                             \
  POSTINIT_MODIFIER(torus)

/*
// DO NOT TOUCH THE MACROS BELOW
*/

// declare all the problem generators
#define PROBLEM(name)                                                                    \
  namespace name {                                                                       \
  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);                            \
  }
FOREACH_PROBLEM
#undef PROBLEM

// declare all the input modifiers
#define MODIFIER(name)                                                                   \
  namespace name {                                                                       \
  void ProblemModifier(ParameterInput *pin);                                             \
  }
FOREACH_MODIFIER
#undef MODIFIER

// declare all the initial condition modifiers
#define POSTINIT_MODIFIER(name)                                                          \
  namespace name {                                                                       \
  void PostInitializationModifier(ParameterInput *pin, Mesh *pmesh);                     \
  }
FOREACH_POSTINIT_MODIFIER
#undef POSTINIT_MODIFIER

namespace phoebus {

// make a map so we get all the function pointers available for lookup by name
static std::map<std::string, std::function<void(MeshBlock *pmb, ParameterInput *pin)>>
    pgen_dict({
#define PROBLEM(name) {#name, name::ProblemGenerator},
        FOREACH_PROBLEM
#undef PROBLEM
    });

static std::map<std::string, std::function<void(ParameterInput *pin)>> pmod_dict({
#define MODIFIER(name) {#name, name::ProblemModifier},
    FOREACH_MODIFIER
#undef MODIFIER
});

static std::map<std::string, std::function<void(ParameterInput *pin, Mesh *pmesh)>>
    pinitmod_dict({
#define POSTINIT_MODIFIER(name) {#name, name::PostInitializationModifier},
        FOREACH_POSTINIT_MODIFIER
#undef POSTINITMODIFIER
    });

/*
// END OF UNTOUCHABLE MACRO SECTION
*/

class PressResidual {
 public:
  KOKKOS_INLINE_FUNCTION
  PressResidual(const EOS &eos, const Real rho, const Real P, const Real Ye)
      : eos_(eos), rho_(rho), P_(P) {
    lambda_[0] = Ye;
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real e) {
    return eos_.PressureFromDensityInternalEnergy(rho_, e, lambda_) - P_;
  }

 private:
  const EOS &eos_;
  Real rho_, P_;
  Real lambda_[2];
};

template <typename T>
KOKKOS_INLINE_FUNCTION Real energy_from_rho_P(T &eos, const Real rho, const Real P,
                                              const Real emin, const Real emax,
                                              const Real Ye = 0.0) {
  PARTHENON_REQUIRE(P >= 0, "Pressure is negative!");
  PressResidual res(eos, rho, P, Ye);
  root_find::RootFind root;
  Real eroot = root.regula_falsi(res, emin, emax, 1.e-10 * P, emin - 1.e10);
  return rho * eroot;
}

} // namespace phoebus

#endif
