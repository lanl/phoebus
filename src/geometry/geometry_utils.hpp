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

#ifndef GEOMETRY_GEOMETRY_UTILS_HPP_
#define GEOMETRY_GEOMETRY_UTILS_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/variables.hpp"

namespace Geometry {
constexpr int NDSPACE = 3;
constexpr int NDFULL = NDSPACE + 1;
constexpr Real SMALL = 10 * std::numeric_limits<Real>::epsilon();
} // namespace Geometry

#define SPACELOOP(i) for (int i = 0; i < Geometry::NDSPACE; i++)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < Geometry::NDFULL; mu++)

#define SPACELOOP2(i, j) SPACELOOP(i) SPACELOOP(j)
#define SPACETIMELOOP2(mu, nu) SPACETIMELOOP(mu) SPACETIMELOOP(nu)
#define SPACELOOP3(i, j, k) SPACELOOP(i) SPACELOOP(j) SPACELOOP(k)
#define SPACETIMELOOP3(mu, nu, sigma)                                          \
  SPACETIMELOOP(mu) SPACETIMELOOP(nu) SPACETIMELOOP(sigma)

namespace Geometry {
namespace Utils {

template <typename T> KOKKOS_INLINE_FUNCTION int sgn(const T &val) {
  return (T(0) <= val) - (val < T(0));
}
KOKKOS_INLINE_FUNCTION Real ratio(Real a, Real b) {
  return a / (b + sgn(b) * SMALL);
}

struct MeshBlockShape {
  MeshBlockShape(ParameterInput *pin) {
    ng = pin->GetOrAddInteger("parthenon/mesh", "nghost", 2);
    int mesh_nx1 = pin->GetInteger("parthenon/mesh", "nx1");
    int mesh_nx2 = pin->GetInteger("parthenon/mesh", "nx2");
    int mesh_nx3 = pin->GetInteger("parthenon/mesh", "nx3");
    ndim = (mesh_nx3 > 1 ? 3 : (mesh_nx2 > 1 ? 2 : 1));

    nx1 = pin->GetOrAddInteger("parthenon/meshblock", "nx1", mesh_nx1) + 2 * ng;
    nx2 = (ndim >= 2)
              ? pin->GetOrAddInteger("parthenon/meshblock", "nx2", mesh_nx2) +
                    2 * ng
              : 1;
    nx3 = (ndim >= 3)
              ? pin->GetOrAddInteger("parthenon/meshblock", "nx3", mesh_nx3) +
                    2 * ng
              : 1;
  }
  int nx1, nx2, nx3, ndim, ng;
};

template <typename System, typename... Args>
KOKKOS_INLINE_FUNCTION void
SetConnectionCoeffByFD(const System &s, Real Gamma[NDFULL][NDFULL][NDFULL],
                       Args... args) {
  Real dg[NDFULL][NDFULL][NDFULL];
  s.MetricDerivative(std::forward<Args>(args)..., dg);
  for (int a = 0; a < NDFULL; ++a) {
    for (int b = 0; b < NDFULL; ++b) {
      for (int c = 0; c < NDFULL; ++c) {
        Gamma[a][b][c] = 0.5 * (dg[b][a][c] + dg[c][a][b] - dg[b][c][a]);
        //Gamma[c][a][b] = 0.5 * (dg[c][a][b] + dg[c][b][a] - dg[a][b][c]);
      }
    }
  }
}

// TODO(JMM) Currently assumes static metric
template <typename System>
KOKKOS_INLINE_FUNCTION void SetGradLnAlphaByFD(const System &s, Real dx,
                                               Real X0, Real X1, Real X2,
                                               Real X3, Real da[NDFULL]) {
  LinearAlgebra::SetZero(da, NDFULL);
  for (int d = 1; d < NDFULL; ++d) {
    Real XX1 = X1;
    Real XX2 = X2;
    Real XX3 = X3;
    if (d == 1)
      XX1 += dx;
    if (d == 2)
      XX2 += dx;
    if (d == 3)
      XX3 += dx;
    Real alpha = s.Lapse(X0, X1, X2, X3);
    Real alphap = s.Lapse(X0, XX1, XX2, XX3);
    da[d] = ratio(alphap - alpha, dx * alpha);
  }
}

KOKKOS_INLINE_FUNCTION
void Lower(const double Vcon[NDFULL], const double Gcov[NDFULL][NDFULL],
           double Vcov[NDFULL]) {
  SPACETIMELOOP(mu) {
    Vcov[mu] = 0.;
    SPACETIMELOOP(nu) {
      Vcov[mu] += Gcov[mu][nu]*Vcon[nu];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void Raise(const double Vcov[NDFULL], const double Gcon[NDFULL][NDFULL],
           double Vcon[NDFULL]) {
  SPACETIMELOOP(mu) {
    Vcon[mu] = 0.;
    SPACETIMELOOP(nu) {
      Vcon[mu] += Gcon[mu][nu]*Vcov[nu];
    }
  }
}

KOKKOS_INLINE_FUNCTION
int KroneckerDelta(const int a, const int b) {
  if (a == b) {
    return 1;
  } else {
    return 0;
  }
}

// TODO(JMM): Currently assumes static metric
template <typename System>
KOKKOS_INLINE_FUNCTION void
SetMetricGradientByFD(const System &s, Real dx, Real X0, Real X1, Real X2,
                      Real X3, Real dg[NDFULL][NDFULL][NDFULL]) {
  LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
  Real gl[NDFULL][NDFULL];
  Real gr[NDFULL][NDFULL];
  for (int d = 1; d < NDFULL; ++d) {
    Real X1L = X1;
    Real X2L = X2;
    Real X3L = X3;
    Real X1R = X1;
    Real X2R = X2;
    Real X3R = X3;
    if (d == 1) {
      X1L -= dx;
      X1R += dx;
    }
    if (d == 2) {
      X2L -= dx;
      X2R += dx;
    }
    if (d == 3) {
      X3L -= dx;
      X3R += dx;
    }
    s.SpacetimeMetric(X0, X1R, X2R, X3R, gr);
    s.SpacetimeMetric(X0, X1L, X2L, X3L, gl);
    SPACETIMELOOP(mu) {
      SPACETIMELOOP(nu) { dg[mu][nu][d] = ratio(gr[mu][nu] - gl[mu][nu], 2*dx); }
    }
  }
}

KOKKOS_INLINE_FUNCTION
int Flatten2(int m, int n, int size) {
  PARTHENON_DEBUG_REQUIRE(0 <= m && 0 <= n && m < size && n < size, "bounds");
  if (m > n) { // remove recursion
    int tmp = n;
    n = m;
    m = tmp;
  }
  return n + size * m - m * (m - 1) / 2 - m;
}
KOKKOS_INLINE_FUNCTION
int SymSize(int size) { return size * size - size * (size - 1) / 2; }

KOKKOS_INLINE_FUNCTION
void InvertMetric(Real gcov[NDFULL][NDFULL], Real gcon[NDFULL][NDFULL]) {
  // A^{-1} = Ajdugate(A)/Determinant(A)
  Real determinant = LinearAlgebra::Determinant4D(gcov);
  //calculate adjugates
  gcon[0][0] = -gcov[1][3]*gcov[2][2]*gcov[3][1] + gcov[1][2]*gcov[2][3]*gcov[3][1] + 
                gcov[1][3]*gcov[2][1]*gcov[3][2] - gcov[1][1]*gcov[2][3]*gcov[3][2] - 
                gcov[1][2]*gcov[2][1]*gcov[3][3] + gcov[1][1]*gcov[2][2]*gcov[3][3];

  gcon[0][1] =  gcov[0][3]*gcov[2][2]*gcov[3][1] - gcov[0][2]*gcov[2][3]*gcov[3][1] - 
                gcov[0][3]*gcov[2][1]*gcov[3][2] + gcov[0][1]*gcov[2][3]*gcov[3][2] + 
                gcov[0][2]*gcov[2][1]*gcov[3][3] - gcov[0][1]*gcov[2][2]*gcov[3][3];

  gcon[0][2] = -gcov[0][3]*gcov[1][2]*gcov[3][1] + gcov[0][2]*gcov[1][3]*gcov[3][1] + 
                gcov[0][3]*gcov[1][1]*gcov[3][2] - gcov[0][1]*gcov[1][3]*gcov[3][2] - 
                gcov[0][2]*gcov[1][1]*gcov[3][3] + gcov[0][1]*gcov[1][2]*gcov[3][3];

  gcon[0][3] =  gcov[0][3]*gcov[1][2]*gcov[2][1] - gcov[0][2]*gcov[1][3]*gcov[2][1] - 
                gcov[0][3]*gcov[1][1]*gcov[2][2] + gcov[0][1]*gcov[1][3]*gcov[2][2] + 
                gcov[0][2]*gcov[1][1]*gcov[2][3] - gcov[0][1]*gcov[1][2]*gcov[2][3];

  gcon[1][1] = -gcov[0][3]*gcov[2][2]*gcov[3][0] + gcov[0][2]*gcov[2][3]*gcov[3][0] + 
                gcov[0][3]*gcov[2][0]*gcov[3][2] - gcov[0][0]*gcov[2][3]*gcov[3][2] - 
                gcov[0][2]*gcov[2][0]*gcov[3][3] + gcov[0][0]*gcov[2][2]*gcov[3][3];

  gcon[1][2] =  gcov[0][3]*gcov[1][2]*gcov[3][0] - gcov[0][2]*gcov[1][3]*gcov[3][0] - 
                gcov[0][3]*gcov[1][0]*gcov[3][2] + gcov[0][0]*gcov[1][3]*gcov[3][2] + 
                gcov[0][2]*gcov[1][0]*gcov[3][3] - gcov[0][0]*gcov[1][2]*gcov[3][3];

  gcon[1][3] = -gcov[0][3]*gcov[1][2]*gcov[2][0] + gcov[0][2]*gcov[1][3]*gcov[2][0] + 
                gcov[0][3]*gcov[1][0]*gcov[2][2] - gcov[0][0]*gcov[1][3]*gcov[2][2] - 
                gcov[0][2]*gcov[1][0]*gcov[2][3] + gcov[0][0]*gcov[1][2]*gcov[2][3];

  gcon[2][2] = -gcov[0][3]*gcov[1][1]*gcov[3][0] + gcov[0][1]*gcov[1][3]*gcov[3][0] + 
                gcov[0][3]*gcov[1][0]*gcov[3][1] - gcov[0][0]*gcov[1][3]*gcov[3][1] - 
                gcov[0][1]*gcov[1][0]*gcov[3][3] + gcov[0][0]*gcov[1][1]*gcov[3][3];

  gcon[2][3] =  gcov[0][3]*gcov[1][1]*gcov[2][0] - gcov[0][1]*gcov[1][3]*gcov[2][0] - 
                gcov[0][3]*gcov[1][0]*gcov[2][1] + gcov[0][0]*gcov[1][3]*gcov[2][1] + 
                gcov[0][1]*gcov[1][0]*gcov[2][3] - gcov[0][0]*gcov[1][1]*gcov[2][3];

  gcon[3][3] = -gcov[0][2]*gcov[1][1]*gcov[2][0] + gcov[0][1]*gcov[1][2]*gcov[2][0] + 
                gcov[0][2]*gcov[1][0]*gcov[2][1] - gcov[0][0]*gcov[1][2]*gcov[2][1] - 
                gcov[0][1]*gcov[1][0]*gcov[2][2] + gcov[0][0]*gcov[1][1]*gcov[2][2];

  //symmetries
  gcon[1][0] = gcon[0][1];
  gcon[2][0] = gcon[0][2];
  gcon[3][0] = gcon[0][3];
  gcon[2][1] = gcon[1][2];
  gcon[3][1] = gcon[1][3];
  gcon[3][2] = gcon[2][3];

  //inverse
  SPACETIMELOOP2(i,j) gcon[i][j] /= determinant;
}

} // namespace Utils
} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_UTILS_HPP_
