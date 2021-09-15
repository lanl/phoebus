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

#ifndef MONOPOLE_GR_MONOPOLE_GR_UTILS_HPP_
#define MONOPOLE_GR_MONOPOLE_GR_UTILS_HPP_

#include "microphysics/eos_phoebus/eos_phoebus.hpp"

#include "monopole_gr.hpp"

namespace MonopoleGR {

constexpr unsigned int log2(unsigned int n) { return (n > 1) ? 1 + log2(n >> 1) : 0; }

// ======================================================================

namespace ShootingMethod {

KOKKOS_INLINE_FUNCTION
Real GetARHS(Real a, Real K, Real r, Real rho) {
  return (r <= 0)
             ? 0
             : (a / (2. * r)) * (r * r * (8 * M_PI * rho - (3. / 2.) * K * K) - a + 1);
}

KOKKOS_INLINE_FUNCTION
Real GetKRHS(Real a, Real K, Real r, Real j) {
  return (r <= 0) ? 0 : 8 * M_PI * a * a * j - (3. / r) * K;
}

KOKKOS_INLINE_FUNCTION
void HypersurfaceRHS(Real r, const Real in[NHYPER], const Real matter[NMAT],
                     Real out[NHYPER]) {
  Real a = in[Hypersurface::A];
  Real K = in[Hypersurface::K];
  Real rho = matter[Matter::RHO];
  Real j = matter[Matter::J_R];
  out[Hypersurface::A] = GetARHS(a, K, r, rho);
  out[Hypersurface::K] = GetKRHS(a, K, r, j);
}

template <typename H, typename M>
KOKKOS_INLINE_FUNCTION void GetResidual(const H &h, const M &m, Real r, int npoints,
                                        Real R[NHYPER]) {
  Real a = h(Hypersurface::A, npoints - 1);
  Real K = h(Hypersurface::K, npoints - 1);
  Real rho = m(Matter::RHO, npoints - 1);
  Real dadr = GetARHS(a, K, r, rho);
  R[Hypersurface::A] = (a - a * a * a) / (2 * r) - dadr;
  R[Hypersurface::K] = K;
}
} // namespace ShootingMethod

namespace TOV {

KOKKOS_INLINE_FUNCTION
Real GetMRHS(Real r, Real rho_adm) { return r == 0 ? 0 : 4 * M_PI * r * r * rho_adm; }

Real GetPRHS(Real r, Real rho_adm, Real m, Real P) {
  return r == 0 ? 0 : -(rho_adm + P) * (m + 4 * M_PI * r * r * r * P) / (r * (r - 2 * m));
}

KOKKOS_INLINE_FUNCTION
void TovRHS(Real r, const Real in[NTOV], singularity::EOS &eos, const Real T,
            Real out[NTOV]) {
  Real m = in[0];
  Real P = in[1];
  Real rho, eps;
  // TODO(JMM): Use lambdas
  eos.DensityEnergyFromPressureTemperature(P, T, nullptr, rho, eps);
  Real rho_adm = rho * (1 + eps);
  out[0] = GetMRHS(r, rho_adm);
  out[1] = GetPRHS(r, rho_adm, m, P);
}

} // namespace TOV

} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_UTILS_HPP_
