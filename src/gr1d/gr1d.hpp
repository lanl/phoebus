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

#ifndef GR1D_GR1D_HPP_
#define GR1D_GR1D_HPP_

// stdlib
#include <memory>

// Spiner
#include <spiner/interpolation.hpp>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

// TODO(JMM): Clean up notation? Or is it fine?
/*
  ASSUMPTIONS:
  -------------
  ADM split:

  ds^2 = (-alpha^2 + beta_i beta^i) dt^2 + 2 beta_i dt dx^i + gamma_{ij} dx^i dx^j

  where alpha is lapse, beta is shift, gamma is 3-metric

  with spherically symmetric ansatz:

  ^(3)ds^2 = gamma_{ij} dx^i dx^j
           = a^2 dr^2 + b^2 r^2 dOmega^2

  Areal coordinates, meaning surface area of sphere = pi r^2
  => b = 1

  Extrinsic curvature defines time-evolution of metric:

  (partial_t - Lie_beta) gamma_{ij} = -2 alpha K_{ij}

  We assume trace of extrinsic curvature vanishes:

  K = K^i_i = 0

  This is so-called "maximal slicing" and provides
  singularity-avoiding coordinates.

  Note that:

  Lie_beta gamma_{ij} = gamma_{kj} \partial_i beta^k
                        + beta^k \partial_k gamma_{ij}
                        + gamma_{ki} \partial_j beta^k
			
  Trace-free + spherical symmetry implies that
  only non-vanishing components of extrinsic curvature are:

  K^th_th = K^ph_ph = -(1/2) K^r_r

  so we work with K^r_r

  GOVERNING EQUATIONS, SUMMARY:
  ------------------------------
  gamm_{ij} = diag(a^2, r^2, r^2 sin^2(th))
  K^i_j = K^r_r diag(1, -1/2, -1/2)

  da/dr = (a/2) (r^2 (16 pi rho - (3/2) (K^r_r)^2) - a + 1)
  r d K^r_r/dr = r k pi a^2 j^r - 3 K^r_r
  d alpha/dr = aleph
  a d aleph/dr = a^3 alpha ((3/2) (K^r_r)^2 + 4 pi (rho + S)) - (da/dr) aleph
  beta^r = -(1/2) alpha r K^r_r

  BOUNDARY CONDITIONS
  -------------------
  at r = 0:
     da/dr = 0
     d K^r_r /dr = 0
     aleph = 0

  at outer boundary (idealy r = infinity):
     da/dr = (a - a^3) / 2 r OR a = 1
     K^r_r = 0
     a aleph = (1 - a)/r     OR alpha = 1

  MATTER COMPONENTS:
  ------------------

  In ADM formulation:

  rho    = TOTAL mass-energy.
  j^i    = CONSERVED 3-momentum vector
  S_{ij} = 3-stress tensor
  ---------------------------------------
  rho    = n^mu n^nu T_{mu nu} = tau + D
  j^i    = - P^{i mu} n^nu T_{mu nu} = S^i 
  S_{ij} = P^mu_i P^nu_i T_{munu}

  where n^mu is unit normal of hypersurface, P^{i mu} is projector

  Translates to (in Valencia variables):

  rho    = tau + D
  j^i    = S^i
  Trc(S) = (rho_0 h + b^2)W^2 + 3(P + b^2/2) - b^i b_i

  In case of no B fields,

  trc(S) = (tau + D) + 4 P - rho_0 h

  where rho_0 here is primitive density and h is enthalpy.
 */

namespace GR1D {

struct Grids {
  using Grid_t = parthenon::ParArray1D<Real>;
  Grid_t a;      // sqrt(g_{11}). rr-component of 3-metric
  Grid_t dadr;
  Grid_t K_rr;   // K^r_r. rr-component of extrinsic curvature
  Grid_t dKdr;
  Grid_t alpha;  // lapse
  Grid_t dalphadr;
  Grid_t aleph;  // partial_r alpha
  Grid_t dalephdr;
  Grid_t rho;    // Primitive density... (0,0)-component of Tmunu
  Grid_t j_r;    // Radial momentum, (r,t)-component of Tmunu
  Grid_t trcS;   // Trace of the stress tensor: S = S^i_i
};

// TODO(JMM): Do we want this?
using Radius = Spiner::RegularGrid1D;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus IterativeSolve(StateDescriptor *pkg);

constexpr int FD_ORDER = 4;
constexpr int NGHOST = FD_ORDER / 2;

KOKKOS_FORCEINLINE_FUNCTION
Real CenteredFD4(const Grids::Grid_t &v, const int i, const Real dx) {
  return ((-1 / 12.) * v(i - 2) + (-2. / 3.) * v(i - 1) + (2. / 3.) * v(i + 1) +
          (1. / 12.) * v(i + 2)) /
         dx;
}
KOKKOS_FORCEINLINE_FUNCTION
Real CenteredFD2(const Grids::Grid_t &v, const int i, const Real dx) {
  return (v(i + 1) - v(i - 1))/(2*dx);
}
KOKKOS_FORCEINLINE_FUNCTION
void SummationByPartsFD4(parthenon::team_mbr_t member, const Grids::Grid_t &v,
                         const Grids::Grid_t &dvdx, const int npoints, const Real dx) {
  parthenon::par_for_inner(member, NGHOST, npoints - 1 - NGHOST, [&](const int i) {
    dvdx(i) = CenteredFD4(v, i, dx);
  });
  dvdx(0) = (v(1) - v(0))/dx;
  dvdx(1) = CenteredFD2(v, 1, dx);
  dvdx(npoints - 2) = CenteredFD2(v, npoints - 2, dx);
  dvdx(npoints - 1) = (v(npoints - 1) - v(npoints - 2))/dx;
}

} // GR1D

#endif // GR1D_GR1D_HPP_
