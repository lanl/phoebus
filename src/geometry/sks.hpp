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

#ifndef GEOMETRY_SUPERIMPOSED_KERRSCHILD_HPP_
#define GEOMETRY_SUPERIMPOSED_KERRSCHILD_HPP_

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

namespace Geometry {

class SuperimposedKerrSchild {
  // Superimposed Kerr-Schild (SKS) Spacetime
  // Paper:
  // Circumbinary Disk Accretion into Spinning Black Hole Binaries
  // Federico G. Lopez Armengol et al. 2021
  // https://iopscience.iop.org/article/10.3847/1538-4357/abf0af/meta
private:
  // Auxiliary functions
  KOKKOS_INLINE_FUNCTION
  Real Rho(Real x, Real y, Real z) const { return sqrt(x*x + y*y + z*z); }
  KOKKOS_INLINE_FUNCTION
  Real RKS(Real x, Real y, Real z, Real a) const {
    Real temp, rho;
    rho = Rho(x,y,z);
    temp = rho*rho - a*a;
    return sqrt(0.5*temp + sqrt(0.25*temp*temp + a*a*z*z));
  }
  KOKKOS_INLINE_FUNCTION
  Real H(Real x, Real y, Real z, Real a, Real M) const {
    Real rks, asq, zsq;
    Real numerator, denominator;
    rks = RKS(x,y,z,a);
    asq = a*a;
    zsq = z*z;
    numerator = M * rks;
    denominator = rks*rks + asq * zsq / (rks*rks);
    return numerator / denominator;
  }
  KOKKOS_INLINE_FUNCTION
  void Lmu(Real x, Real y, Real z, Real a, Real lmu[NDFULL]) const {
    Real rks, rkssq, asq;
    asq = a*a;
    rks = RKS(x,y,z,a);
    rkssq = rks*rks;
    lmu[0] = 1;
    lmu[1] = (rks*x + a*y)/(rkssq + asq);
    lmu[2] = (rks*y - a*x)/(rkssq + asq);
    lmu[3] = z / rks;
  }
  // Calculate local black hole coordinates
  // Input: 
  //   center of mass coordinates
  //   black hole location (Sx,Sy)
  //   black hole velocity (vx,vy)
  KOKKOS_INLINE_FUNCTION
  void CoordsBH(Real t, Real x, Real y, Real z, Real Sx, Real Sy, Real vx, Real vy, Real xBH[NDFULL]) const {
    Real beta, nx, ny, gamma;
    beta = sqrt(vx*vx + vy*vy);
    nx = vx / beta;
    ny = vy / beta;
    gamma = 1. / sqrt(1 - beta*beta);
    xBH[0] = gamma*(t - nx*x - ny*y);
    xBH[1] = -Sx + (1. + (gamma-1.)*nx*nx)*x + (gamma-1.)*nx*ny*y;
    xBH[2] = -Sy + (gamma-1.)*ny*nx*x + (1. + (gamma-1.)*ny*ny)*y;
    xBH[3] = z;
  }
  // Calculate transformation tensor for Lmu
  // Transforms from local BH coordinate basis to center of mass
  KOKKOS_INLINE_FUNCTION
  void BoostBH(Real vx, Real vy, Real Lambda[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(Lambda,NDFULL,NDFULL);
    Real beta, nx, ny, gamma;
    beta = sqrt(vx*vx + vy*vy);
    nx = vx / beta;
    ny = vy / beta;
    gamma = 1. / sqrt(1 - beta*beta);
    Lambda[0][0] = gamma;
    Lambda[0][1] = Lambda[1][0] = gamma*beta*nx;
    Lambda[0][2] = Lambda[2][0] = gamma*beta*ny;
    Lambda[1][1] = 1. + (gamma-1.)*nx*nx;
    Lambda[1][2] = Lambda[2][1] = (gamma-1.)*nx*ny;
    Lambda[2][2] = 1. + (gamma-1.)*ny*ny;
    Lambda[3][3] = 1.;
  }
public:
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {}
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {}

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3,
                       Real g[NDFULL][NDFULL]) const {
    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    Real m1, m2, a1, a2, b, Omega;
    Real Sx, Sy, vx, vy, phase;
    Real HBH1,HBH2;
    Real CoordsBH1[NDFULL], CoordsBH2[NDFULL];
    Real LBH[NDFULL], LBH1[NDFULL], LBH2[NDFULL];
    Real eta_flat[NDFULL][NDFULL], Lambda[NDFULL][NDFULL];

    LinearAlgebra::SetZero(eta_flat,NDFULL,NDFULL);
    eta_flat[0][0] = -1.;
    for(int i=1; i < NDFULL; i++) { eta_flat[i][i] = 1.; }

    //set trajectories - do better later
    m1 = 0.5; m2 = 0.5; a1 = 0; a2 = 0; b = 20.; //dennis for now hard set
    Omega = sqrt((m1+m2)/pow(b,3));
    
    //BH1 contribution
    phase = Omega*X0;
    Sx = m2*b*cos(phase)/(m1+m2);
    Sy = m2*b*sin(phase)/(m1+m2);
    vx = -Omega*m2*b*sin(phase)/(m1+m2);
    vy = Omega*m2*b*cos(phase)/(m1+m2);
    CoordsBH(X0,X1,X2,X3,Sx,Sy,vx,vy,CoordsBH1);
    HBH1 = H(CoordsBH1[1], CoordsBH1[2], CoordsBH1[3],a1,m1);
    Lmu(CoordsBH1[1],CoordsBH1[2],CoordsBH1[3],a1,LBH);
    BoostBH(vx, vy, Lambda);
    LinearAlgebra::SetZero(LBH1,NDFULL);
    SPACETIMELOOP2(mu,kappa) {
      LBH1[mu] += Lambda[kappa][mu]*LBH[kappa];
    }

    //BH2 contribution
    phase += 4.0 * atan(1.0);
    Sx = m1*b*cos(phase)/(m1+m2);
    Sy = m1*b*sin(phase)/(m1+m2);
    vx = -Omega*m1*b*sin(phase)/(m1+m2);
    vy = Omega*m1*b*cos(phase)/(m1+m2);
    CoordsBH(X0,X1,X2,X3,Sx,Sy,vx,vy,CoordsBH2);
    HBH2 = H(CoordsBH2[1], CoordsBH2[2], CoordsBH2[3],a2,m2);
    Lmu(CoordsBH2[1],CoordsBH2[2],CoordsBH2[3],a2,LBH);
    BoostBH(vx,vy,Lambda);
    LinearAlgebra::SetZero(LBH2,NDFULL);
    SPACETIMELOOP2(mu,kappa) {
      LBH2[mu] += Lambda[kappa][mu]*LBH[kappa];
    }

    //Full metric
    SPACETIMELOOP2(mu,nu) {
      g[mu][nu] = eta_flat[mu][nu] + 2.*HBH1*LBH1[mu]*LBH1[nu] + 2.*HBH2*LBH2[mu]*LBH2[nu];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    Real gcov[NDFULL][NDFULL];
    LinearAlgebra::SetZero(gcov,NDFULL,NDFULL);
    SpacetimeMetric(X0,X1,X2,X3,gcov);
    LinearAlgebra::SetZero(g,NDFULL,NDFULL);
    Utils::InvertMetric(gcov,g);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3,
              Real g[NDSPACE][NDSPACE]) const {}
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real g[NDSPACE][NDSPACE]) const {}
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {}
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    Real gcov[NDFULL][NDFULL];
    LinearAlgebra::SetZero(gcov,NDFULL,NDFULL);
    SpacetimeMetric(X0,X1,X2,X3,gcov);
    return std::sqrt(-LinearAlgebra::Determinant4D(gcov));
  }
  // Gamma^mu_{nu sigma}
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {}
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {}

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1;
    C[2] = X2;
    C[3] = X3;
  }
};

using SKSMeshBlock = Analytic<SuperimposedKerrSchild, IndexerMeshBlock>;
using SKSMesh = Analytic<SuperimposedKerrSchild, IndexerMesh>;

using CSKSMeshBlock = CachedOverMeshBlock<SKSMeshBlock>;
using CSKSMesh = CachedOverMesh<SKSMesh>;

template <>
void Initialize<CSKSMeshBlock>(ParameterInput *pin,
                                        StateDescriptor *geometry); 

} // namespace Geometry

#endif // GEOMETRY_SUPERIMPOSED_KERRSCHILD_HPP_
