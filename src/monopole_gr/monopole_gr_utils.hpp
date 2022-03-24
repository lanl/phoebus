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

// System includes
#include <cmath>

// Parthenon includes
#include <parthenon/package.hpp>

// Phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr_base.hpp"
#include "phoebus_utils/robust.hpp"

namespace MonopoleGR {
namespace ShootingMethod {

KOKKOS_INLINE_FUNCTION
Real GetARHS(const Real a, const Real K, const Real r, const Real rho) {
  bool mask = (r > 0);
  // return mask*robust::ratio(a*(4 - 4*a*a + 3*r*r*a*a*K*K + 32*M_PI*a*rho),8*r);
  return mask *
         robust::ratio(a * (4 + a * a * (-4 + r * r * (3 * K * K + 32 * M_PI * rho))),
                       8 * r);
}

KOKKOS_INLINE_FUNCTION
Real GetKRHS(Real a, Real K, Real r, Real j) {
  bool mask = (r > 0);
  return mask * (8 * M_PI * a * a * j - robust::ratio(3. * K, r));
  // return (r <= 0) ? 0 : 8 * M_PI * j - robust::ratio(3.*K, r);
  // if (r <= 0) return 0;
  // return robust::ratio(robust::ratio(8*M_PI*j, a) - 3*r*K,r*r);
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

namespace Interp3DTo1D {

// True if a in a box centered at r with width dr.
// False otherwise.
KOKKOS_INLINE_FUNCTION
bool InBoundsHelper(const Real a, const Real r, const Real dr) {
  return ((a >= (r - 0.5 * dr)) && (a < (r + 0.5 * dr)));
}

KOKKOS_INLINE_FUNCTION
Real GetVolIntersectHelper(const Real rsmall, const Real drsmall, const Real rbig,
                           const Real drbig) {
  // TODO(JMM): If drsmall > drbig, we can probably just swap all the
  // arguments, but I haven't tested the monopole solver in this regime,
  // so I'm forbidding it for now.
  PARTHENON_DEBUG_REQUIRE(drsmall <= drbig, "drsmall <= drbig required.");
  // TODO(JMM): Is this masking too cute?
  // cast this way, there's no branching, but there are more flops maybe?
  bool left_in_bnds = InBoundsHelper(rsmall - 0.5 * drsmall, rbig, drbig);
  bool right_in_bnds = InBoundsHelper(rsmall + 0.5 * drsmall, rbig, drbig);
  bool interior_mask = left_in_bnds && right_in_bnds;
  bool left_mask = right_in_bnds && !left_in_bnds;
  bool right_mask = left_in_bnds && !right_in_bnds;
  Real vol_intersect = interior_mask * (drsmall / drbig);

  // divide by zero impossible because these are grid deltas
  vol_intersect += left_mask * (rsmall + 0.5 * drsmall - (rbig - 0.5 * drbig)) / drbig;
  vol_intersect += right_mask * (rbig + 0.5 * drbig - (rsmall - 0.5 * drsmall)) / drbig;
  return vol_intersect;
}

KOKKOS_INLINE_FUNCTION
void GetCoordsAndDerivsSph(const int k, const int j, const int i,
                           const parthenon::Coordinates_t &coords, Real &r, Real &th,
                           Real &ph, Real &dr, Real &dth, Real &dph, Real &dv) {
  r = coords.x1v(k, j, i);
  th = coords.x2v(k, j, i);
  ph = coords.x3v(k, j, i);
  dr = coords.Dx(1, k, j, i);
  dth = coords.Dx(2, k, j, i);
  dph = coords.Dx(3, k, j, i);
  const Real idr = (1. / 12.) * (dr * dr * dr) + dr * (r * r);
  dv = std::sin(th) * dth * dph * idr;
}

} // namespace Interp3DTo1D
} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_UTILS_HPP_
