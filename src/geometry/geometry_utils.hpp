#ifndef GEOMETRY_GEOMETRY_UTILS_HPP_
#define GEOMETRY_GEOMETRY_UTILS_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <utility>

// Parthenon includes
#include <kokkos_abstraction.hpp>

namespace Geometry {

constexpr int NDSPACE = 3;
constexpr int NDFULL = NDSPACE + 1;
constexpr Real SMALL = 10 * std::numeric_limits<Real>::epsilon();

namespace Utils {

template <typename T> KOKKOS_INLINE_FUNCTION int sgn(const T &val) {
  return (T(0) < val) - (val < T(0));
}
KOKKOS_INLINE_FUNCTION Real ratio(Real a, Real b) { return a / (b + sgn(b) * SMALL); }

// TODO(JMM): Replace these with a memset?
template <typename Array> KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int size) {
  for (int i = 0; i < size; ++i)
    a[i] = 0;
}
template <typename Array> KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int n2, int n1) {
  for (int i2 = 0; i2 < n2; ++i2) {
    for (int i1 = 0; i1 < n1; ++i1) {
      a[i2][i1] = 0;
    }
  }
}
template <typename Array>
KOKKOS_INLINE_FUNCTION void SetZero(Array &a, int n3, int n2, int n1) {
  for (int i3 = 0; i3 < n3; ++i3) {
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        a[i3][i2][i1] = 0;
      }
    }
  }
}

template <typename System>
KOKKOS_INLINE_FUNCTION void SetConnectionCoeffByFD(const System &s, Real X0, Real X1,
                                                   Real X2, Real X3,
                                                   Real Gamma[NDFULL][NDFULL][NDFULL]) {
  Real dg[NDFULL][NDFULL][NDFULL];
  s.MetricDerivative(X0, X1, X2, X3, dg);
  for (int a = 0; a < NDFULL; ++a) {
    for (int b = 0; b < NDFULL; ++b) {
      for (int c = 0; c < NDFULL; ++c) {
        Gamma[a][b][c] = 0.5 * (dg[b][a][c] + dg[c][a][b] - dg[b][c][a]);
      }
    }
  }
}

template <typename System>
KOKKOS_INLINE_FUNCTION void SetGradLnAlphaByFD(const System &s, Real dx, Real X0, Real X1,
                                               Real X2, Real X3, Real da[NDFULL]) {
  SetZero(da, NDFULL);
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
  Vcov[0] = Gcov[0][0] * Vcon[0] + Gcov[0][1] * Vcon[1] + Gcov[0][2] * Vcon[2] +
            Gcov[0][3] * Vcon[3];
  Vcov[1] = Gcov[1][0] * Vcon[0] + Gcov[1][1] * Vcon[1] + Gcov[1][2] * Vcon[2] +
            Gcov[1][3] * Vcon[3];
  Vcov[2] = Gcov[2][0] * Vcon[0] + Gcov[2][1] * Vcon[1] + Gcov[2][2] * Vcon[2] +
            Gcov[2][3] * Vcon[3];
  Vcov[3] = Gcov[3][0] * Vcon[0] + Gcov[3][1] * Vcon[1] + Gcov[3][2] * Vcon[2] +
            Gcov[3][3] * Vcon[3];
}

KOKKOS_INLINE_FUNCTION
void Raise(const double Vcov[NDFULL], const double Gcon[NDFULL][NDFULL],
           double Vcon[NDFULL]) {
  Vcon[0] = Gcon[0][0] * Vcov[0] + Gcon[0][1] * Vcov[1] + Gcon[0][2] * Vcov[2] +
            Gcon[0][3] * Vcov[3];
  Vcon[1] = Gcon[1][0] * Vcov[0] + Gcon[1][1] * Vcov[1] + Gcon[1][2] * Vcov[2] +
            Gcon[1][3] * Vcov[3];
  Vcon[2] = Gcon[2][0] * Vcov[0] + Gcon[2][1] * Vcov[1] + Gcon[2][2] * Vcov[2] +
            Gcon[2][3] * Vcov[3];
  Vcon[3] = Gcon[3][0] * Vcov[0] + Gcon[3][1] * Vcov[1] + Gcon[3][2] * Vcov[2] +
            Gcon[3][3] * Vcov[3];
}

KOKKOS_INLINE_FUNCTION
int KroneckerDelta(const int a, const int b) {
  if (a == b) {
    return 1;
  } else {
    return 0;
  }
}

} // namespace Utils
} // namespace Geometry

#define SPACELOOP(i) for (int i = 0; i < Geometry::NDSPACE; i++)
#define SPACETIMELOOP(mu) for (int mu = 0; mu < Geometry::NDFULL; mu++)

#endif // GEOMETRY_GEOMETRY_UTILS_HPP_
