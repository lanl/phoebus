//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef RECONSTRUCTION_HPP_
#define RECONSTRUCTION_HPP_

#include <algorithm>
#include <basic_types.hpp>
#include <kokkos_abstraction.hpp>

namespace PhoebusReconstruction {

KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 0.0);
  return std::max(0.0, std::min(2.0,std::min(2*r,0.5*(1+r))));
}
KOKKOS_FORCEINLINE_FUNCTION
Real minmod(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 0.0);
  //return std::max(0.0, std::min(2.0,std::min(2*r,0.5*(1+r))));
  const Real t = 1.5;
  return std::max(0.0, std::min(t, std::min(t*r,0.5*(1.0+r))));
}

KOKKOS_FORCEINLINE_FUNCTION
Real ltanh(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.0) 
    return 2.0*std::tanh(dm/dp);
  return 2.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real barth_jespersen(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 3.0); 
  return std::min(std::min(1.0, 4.0/(1.0+r)), 4.0*r/(1.0+r));
}

KOKKOS_FORCEINLINE_FUNCTION
Real overbee_alpha(const Real dm, const Real dp, const Real alpha) {
  if (alpha < 1.0 || alpha > 2.0) {
    PARTHENON_FAIL("alpha not between 1 and 2");
  }
  Real dc = std::abs(0.5*(dp - dm));
  Real dcent = std::min(dc,2.0*std::min(std::abs(dm),std::abs(dp)));
  Real dmin = alpha*std::min(std::abs(dm), std::abs(dp));
  dmin = std::max(dcent,dmin);
  if (std::abs(dp) > 0.)
    return dmin/std::abs(dp);
  return alpha;
}

KOKKOS_FORCEINLINE_FUNCTION
Real overbee(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.)
    return 2.0*std::min(std::abs(dm), std::abs(dp))/std::abs(dp);
  return 2.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real superbee(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 2.0);
  // superbee
  return std::max(std::max(0.0,std::min(2*r,1.0)), std::min(r,2.0));
}

KOKKOS_FORCEINLINE_FUNCTION
Real nolim(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.)
    return 0.5*(dp - dm)/dp;
  return 1.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real phifunc(const Real mind, const Real maxd, const Real gx, const Real gy, const Real dx, const Real dy) {
  Real delta = gx*dx + gy*dy;
  if(delta > 0.) {
    return std::min(1.0,maxd/delta);
  } else if (delta < 0.) {
    return std::min(1.0,mind/delta);
  }
  return 1.0;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void PiecewiseLinear(const int d, const int nlo, const int nhi,
                     const int k, const int j, const int i,
                     const T &v, const ParArrayND<Real> &ql, const ParArrayND<Real> &qr) {
  const int dir = d-1;
  int di = (d == X1DIR ? 1 : 0);
  int dj = (d == X2DIR ? 1 : 0);
  int dk = (d == X3DIR ? 1 : 0);
  for (int n=nlo; n<=nhi; n++) {
    Real dql = v(n,k,j,i) - v(n,k-dk,j-dj,i-di);
    Real dqr = v(n,k+dk,j+dj,i+di) - v(n,k,j,i);
    Real dq = mc(dql,dqr)*dqr;
    ql(dir,n,k+dk,j+dj,i+di) = v(n,k,j,i) + 0.5*dq;
    qr(dir,n,k,j,i) = v(n,k,j,i) - 0.5*dq;
  }
}

} // PhoebusReconstruction

#endif // RECONSTRUCTION_HPP_