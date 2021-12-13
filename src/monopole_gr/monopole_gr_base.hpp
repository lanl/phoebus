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

#ifndef MONOPOLE_GR_MONOPOLE_GR_BASE_HPP_
#define MONOPOLE_GR_MONOPOLE_GR_BASE_HPP_

// Spiner
#include <spiner/interpolation.hpp>

// Parthenon
#include <kokkos_abstraction.hpp>

namespace MonopoleGR {

constexpr int MIN_NPOINTS = 5;

/*
  Array structure:
  Slowest moving index is refinement level. 0 is finest.
  Next index is variable (if appropriate)
  Next index is grid point
 */
using Matter_t = parthenon::ParArray2D<Real>;
using Matter_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int NMAT = 4;
constexpr int NMAT_H = NMAT - 2; // used for integration on CPU
enum Matter {
  RHO = 0,  // Primitive density... (0,0)-component of Tmunu
  J_R = 1,  // Radial momentum, (r,t)-component of Tmunu
  trcS = 2, // Trace of the stress tensor: S = S^i_i
  Srr = 3   // The r-r component of the stress tensor
};
using Volumes_t = parthenon::ParArray1D<Real>;
using Volumes_host_t = typename parthenon::ParArray1D<Real>::HostMirror;

using Hypersurface_t = parthenon::ParArray2D<Real>;
using Hypersurface_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int NHYPER = 2;
enum Hypersurface {
  A = 0, // (r,r)-component of metric
  K = 1  // K^r_r, extrinsic curvature
};

using Alpha_t = parthenon::ParArray1D<Real>;
using Alpha_host_t = typename parthenon::ParArray1D<Real>::HostMirror;

using Beta_t = parthenon::ParArray1D<Real>;

constexpr int NGRAD = 8;
using Gradients_t = parthenon::ParArray2D<Real>;
enum Gradients {
  DADR = 0,     // dadr
  DKDR = 1,     // dKdr
  DALPHADR = 2, // dalphadr
  DBETADR = 3,  // dbetadr
  DADT = 4,     // dadt
  DALPHADT = 5, // dalphadt
  DKDT = 6,     // dKdt
  DBETADT = 7   // dbetadt
};

// TODO(JMM): Do we want this?
using Radius = Spiner::RegularGrid1D;

template <typename Array_t>
PORTABLE_INLINE_FUNCTION Real Interpolate(const Real r, const Array_t &A,
                                          const Radius &rgrid) {
  const int ix = rgrid.index(r);
  const int npoints = rgrid.nPoints();
  const Real min = rgrid.min();
  const Real dx = rgrid.dx();
  const Real floor = ix * dx + min;
  Real c[3];
  c[0] = A(ix);
  if (ix == 0) {
    c[1] = -(3 * A(ix) - 4. * A(ix + 1) + A(ix + 2)) / (2 * dx);
    c[2] = (A(ix) - 2 * A(ix + 1) + A(ix + 2)) / (2 * dx * dx);
  } else if (ix == npoints - 1) {
    c[1] = (3 * A(ix - 2) - 4 * A(ix - 1) + 3 * A(ix)) / (2 * dx);
    c[2] = (A(ix - 2) - 2 * A(ix - 1) + A(ix)) / (2 * dx * dx);
  } else {
    c[1] = (A(ix + 1) - A(ix - 1)) / (2 * dx);
    c[2] = (A(ix + 1) - 2 * A(ix) + A(ix - 1)) / (2 * dx * dx);
  }
  Real xoffset = (r - floor);
  return c[0] + c[1] * xoffset + c[2] * xoffset * xoffset;
}

template <typename Array_t>
PORTABLE_INLINE_FUNCTION Real Interpolate(const Real r, const Array_t &A,
                                          const Radius &rgrid, const int ivar) {
  const int ix = rgrid.index(r);
  const int npoints = rgrid.nPoints();
  const Real min = rgrid.min();
  const Real dx = rgrid.dx();
  const Real floor = ix * dx + min;
  Real c[3];
  c[0] = A(ivar, ix);
  if (ix == 0) {
    c[1] = -(3 * A(ivar, ix) - 4. * A(ivar, ix + 1) + A(ivar, ix + 2)) / (2 * dx);
    c[2] = (A(ivar, ix) - 2 * A(ivar, ix + 1) + A(ivar, ix + 2)) / (2 * dx * dx);
  } else if (ix == npoints - 1) {
    c[1] = (3 * A(ivar, ix - 2) - 4 * A(ivar, ix - 1) + 3 * A(ivar, ix)) / (2 * dx);
    c[2] = (A(ivar, ix - 2) - 2 * A(ivar, ix - 1) + A(ivar, ix)) / (2 * dx * dx);
  } else {
    c[1] = (A(ivar, ix + 1) - A(ivar, ix - 1)) / (2 * dx);
    c[2] = (A(ivar, ix + 1) - 2 * A(ivar, ix) + A(ivar, ix - 1)) / (2 * dx * dx);
  }
  Real xoffset = (r - floor);
  return c[0] + c[1] * xoffset + c[2] * xoffset * xoffset;
}

} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_BASE_HPP_
