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

#ifndef CCSN_CCSN_HPP_
#define CCSN_CCSN_HPP_

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace CCSN {
constexpr int NCCSN = 7;

using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

// cef: additional vars for ccsn problems?
constexpr int R = 0;
constexpr int RHO = 1;
constexpr int V = 2;
constexpr int EPS = 3;
constexpr int YE = 4;
constexpr int P = 5;
constexpr int TEMP = 6;


KOKKOS_INLINE_FUNCTION
Real GetMRHS(const Real r, const Real rho_adm) { return 4 * M_PI * r * r * rho_adm; }

KOKKOS_INLINE_FUNCTION
Real GetPhiRHS(const Real r, const Real rho_adm, const Real m, const Real P) {
  if (r == 0) return 0;
  return (m + 4 * M_PI * r * r * r * P) / (r * (r - 2 * m));
}

KOKKOS_INLINE_FUNCTION
Real GetPRHS(const Real r, const Real rho_adm, const Real m, const Real P,
             const Real Pmin) {
  if (P < Pmin) return 0;
  Real phirhs = GetPhiRHS(r, rho_adm, m, P);
  return -(rho_adm + P) * phirhs;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//this task will change to something that inits the CCSN problem
TaskStatus InitializeCCSN(StateDescriptor *ccsnpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg);

} // namespace CCSN

#endif // CCSN_CCSN_HPP_
