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
#include "monopole_gr_base.hpp"
#include "phoebus_utils/robust.hpp"

namespace MonopoleGR {
namespace ShootingMethod {

KOKKOS_INLINE_FUNCTION
Real GetARHS(const Real a, const Real K, const Real r, const Real rho) {
  if (r <= 0) return 0;
  return a * (4 + a * a * (-4 + r * r * (-3 * K * K + 32 * M_PI * rho))) / (8 * r);
}

KOKKOS_INLINE_FUNCTION
Real GetKRHS(Real a, Real K, Real r, Real j) {
  return (r <= 0) ? 0 : 8 * M_PI * a * a * j - robust::ratio(3.*K, r);
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
} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_UTILS_HPP_
