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

#ifndef TOV_TOV_HPP_
#define TOV_TOV_HPP_

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace TOV {
constexpr int NTOV = 3;

using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int M = 0;
constexpr int P = 1;
constexpr int PHI = 2;

constexpr int NINTRINSIC = 2;

constexpr int RHO0 = 0;
constexpr int EPS = 1;

/*
 * Only valid for polytropic EOS's
 */
KOKKOS_INLINE_FUNCTION
void PolytropeThermoFromP(const Real P, const Real K, const Real Gamma, Real &rho,
                          Real &eps) {
  rho = std::pow(P / K, 1 / Gamma);
  eps = (K / (Gamma - 1)) * std::pow(P / K, (Gamma - 1) / Gamma);
}

KOKKOS_INLINE_FUNCTION
Real PolytropeK(const Real s, const Real Gamma) { return std::exp(s * (Gamma - 1)); }

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

KOKKOS_INLINE_FUNCTION
void TovRHS(Real r, const Real in[NTOV], const Real K, const Real Gamma, const Real Pmin,
            Real out[NTOV]) {
  Real m = in[TOV::M];
  Real P = in[TOV::P];
  // Real phi = in[TOV::PHI];
  if (P < Pmin) P = 0;
  
  Real rho, eps;
  // TODO(JMM): THIS ASSUMES A POLYTROPIC EOS.
  // More generality requires exposing entropy in singularity-eos
  PolytropeThermoFromP(P, K, Gamma, rho, eps);
  Real rho_adm = rho * (1 + eps);
  out[TOV::M] = GetMRHS(r, rho_adm);
  out[TOV::P] = GetPRHS(r, rho_adm, m, P, Pmin);
  out[TOV::PHI] = GetPhiRHS(r, rho_adm, m, P);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus IntegrateTov(StateDescriptor *tovpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg);

} // namespace TOV

#endif // TOV_TOV_HPP_
