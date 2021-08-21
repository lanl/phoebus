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

#ifndef GR1D_GRD1D_UTILS_HPP_
#define GR1D_GRD1D_UTILS_HPP_

#include "gr1d.hpp"

namespace GR1D {

constexpr unsigned int log2(unsigned int n) { return (n > 1) ? 1 + log2(n >> 1) : 0; }

// ======================================================================

namespace ShootingMethod {

KOKKOS_INLINE_FUNCTION
Real GetARHS(Real a, Real K, Real r, Real rho) {
  return (r <= 0)
             ? 0
             : (a / (2. * r)) * (r * r * (16 * M_PI * rho - (3. / 2.) * K * K) - a + 1);
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

// ======================================================================

namespace Multigrid {

constexpr unsigned int MIN_NPOINTS = 5;
constexpr unsigned int LEVEL_OFFSET = log2(MIN_NPOINTS);

KOKKOS_INLINE_FUNCTION unsigned int NPointsPLevel(const unsigned int level,
                                                  const unsigned int npoints_fine) {
  return level == 0 ? npoints_fine : (npoints_fine >> level) + 1;
}

template <typename V>
KOKKOS_INLINE_FUNCTION void RestrictVar(const V &v, const int lcoarse, const int icoarse,
                                        const int nvars, const int npoints_coarse) {
  PARTHENON_DEBUG_REQUIRE(lcoarse > 1, "We must be restricting to a level > 0");
  PARTHENON_DEBUG_REQUIRE(npoints_coarse >= MIN_NPOINTS,
                          "We must not be at the coarsest level");
  const int lfine = lcoarse - 1;
  const int npoints_fine = (npoints_coarse - 1) * 2;
  const int ifine = 2 * icoarse;
  for (int ivar = 0; ivar < nvars; ++ivar) {
    if (icoarse == 0) { // use injection at boundaries
      v(lcoarse, ivar, icoarse) = v(lfine, ivar, ifine);
    } else if (icoarse == npoints_coarse - 1) {
      v(lcoarse, ivar, npoints_coarse - 1) = v(lfine, ivar, npoints_fine - 1);
    } else { // use full weighting
      v(lcoarse, ivar, icoarse) =
          0.25 * (v(lfine, ivar, ifine - 1) + 2 * v(lfine, ivar, ifine) +
                  v(lfine, ivar, ifine + 1));
    }
  }
}

} // namespace Multigrid

} // namespace GR1D

#endif // GR1D_GRD1D_UTILS_HPP_
