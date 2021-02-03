#ifndef GEOMETRY_COORDINATE_SYSTEMS_HPP_
#define GEOMETRY_COORDINATE_SYSTEMS_HPP_

#include <cmath>
#include <limits>
#include <utility>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// phoebus includes
#include "geometry/geometry_variant.hpp"
#include "phoebus_utils/cell_locations.hpp"

using namespace parthenon::package::prelude;
using parthenon::ParArray1D;

namespace Geometry {

using parthenon::Coordinates_t;
static constexpr Real SMALL = 10 * std::numeric_limits<Real>::epsilon();

// Boiler plate for analytic coordinate systems so you only need to
// define the functions of X0,X1,X2,X3, not all the functions of indices.
/*
  To utilize this template, the fullowing methods are expected:
  - A constructor/destructor
  - Lapse(X0, X1, X2, X3)
  - ContravariantShift(l, X0, X1, X2, X3)
  - Metric(l,m,X0, X1, X2, X3)
  - MetricInverse(l, m, X0, X1, X2, X3)
  - DetGamma(X0, X1, X2, X3)
  - DetG(X0, X1, X2, X3)
  - ConnectionCoefficient(mu, nu, sigma, X0, X1, X2, X3)
  - MetricDerivative(mu, l, nu, X0, X1, X2, X3)
  - GradLnAlpha(mu, X0, X1, X2, X3)
  - C0(X0, X1, X2, X3)
  - C1(X0, X1, X2, X3)
  - C2(X0, X1, X2, X3)
  - C3(X0, X1, X2, X3)
 */
// TODO(JMM): We use a horrible run-time indirection here, where the
// code will throw if you use the MeshData calls after initializing
// for a meshblock or use the MeshBlockData calls after initializing
// for a MeshData object. This COULD be done at compile time with
// metaprogramming. But I went with this version for now.
template <typename System> class Analytic {
public:
  Analytic() = default;
  template <typename... Args>
  Analytic(const Coordinates_t &coordinates, Args... args)
      : multiple_coords_(false), X0_(0), coordinates_single_(coordinates),
        system_(std::forward<Args>(args)...) {}
  template <typename... Args>
  Analytic(const Real X0, const Coordinates_t &coordinates, Args... args)
      : multiple_coords_(false), X0_(X0), coordinates_single_(coordinates),
        system_(std::forward<Args>(args)...) {}
  template <typename... Args>
  Analytic(const ParArray1D<Coordinates_t> &coordinates, Args... args)
      : multiple_coords_(true), X0_(0), coordinates_multi_(coordinates),
        system_(std::forward<Args>(args)...) {}
  template <typename... Args>
  Analytic(const Real X0, const ParArray1D<Coordinates_t> &coordinates,
           Args... args)
      : multiple_coords_(true), X0_(X0), coordinates_multi_(coordinates),
        system_(std::forward<Args>(args)...) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    return system_.Lapse(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return Lapse(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return Lapse(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, Real X0, Real X1, Real X2, Real X3) const {
    return system_.ContravariantShift(l, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return ContravariantShift(l, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, CellLocation loc, int b, int k, int j,
                          int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return ContravariantShift(l, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real Metric(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    return system_.Metric(l, m, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Metric(int l, int m, CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return Metric(l, m, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Metric(int l, int m, CellLocation loc, int b, int k, int j,
              int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return Metric(l, m, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    return system_.MetricInverse(l, m, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, CellLocation loc, int k, int j,
                     int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return MetricInverse(l, m, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, CellLocation loc, int b, int k, int j,
                     int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return MetricInverse(l, m, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real SpacetimeMetric(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    return system_.SpacetimeMetric(l, m, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real SpacetimeMetric(int l, int m, CellLocation loc, int k, int j,
                       int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return SpacetimeMetric(l, m, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real SpacetimeMetric(int l, int m, CellLocation loc, int b, int k, int j,
                       int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return SpacetimeMetric(l, m, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetGamma(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return DetGamma(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return DetGamma(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    return system_.DetG(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return DetG(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return DetG(X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, Real X0, Real X1,
                             Real X2, Real X3) const {
    return system_.ConnectionCoefficient(mu, nu, sigma, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, CellLocation loc, int k,
                             int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return ConnectionCoefficient(mu, nu, sigma, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, CellLocation loc, int b,
                             int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return ConnectionCoefficient(mu, nu, sigma, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, Real X0, Real X1, Real X2,
                        Real X3) const {
    return system_.MetricDerivative(mu, l, nu, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, CellLocation loc, int k, int j,
                        int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return MetricDerivative(mu, l, nu, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, CellLocation loc, int b, int k,
                        int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return MetricDerivative(mu, l, nu, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, Real X0, Real X1, Real X2, Real X3) const {
    return system_.GradLnAlpha(mu, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return GradLnAlpha(mu, X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return GradLnAlpha(mu, X0_, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real C0(Real X0, Real X1, Real X2, Real X3) const {
    return system_.C0(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C1(Real X0, Real X1, Real X2, Real X3) const {
    return system_.C1(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C2(Real X0, Real X1, Real X2, Real X3) const {
    return system_.C2(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C3(Real X0, Real X1, Real X2, Real X3) const {
    return system_.C3(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C0(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return C0(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C1(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return C1(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C2(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return C2(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C3(CellLocation loc, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return C3(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C0(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return C0(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C1(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return C1(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C2(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, k, j, i, X1, X2, X3);
    return C2(X0_, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real C3(CellLocation loc, int b, int k, int j, int i) const {
    Real X1, X2, X3;
    GetX(loc, b, k, j, i, X1, X2, X3);
    return C3(X0_, X1, X2, X3);
  }

private:
  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int b, int k, int j, int i, Real &X1, Real &X2,
            Real &X3) const {
    PARTHENON_REQUIRE(multiple_coords_,
                      "One Coordinates_t per meshblock is required");
    // defaults
    X1 = coordinates_multi_(b).x1v(i);
    X2 = coordinates_multi_(b).x2v(j);
    X3 = coordinates_multi_(b).x3v(k);
    // overwrite
    switch (loc) {
    case CellLocation::Face1:
      X1 = coordinates_multi_(b).x1f(i);
      break;
    case CellLocation::Face2:
      X2 = coordinates_multi_(b).x2f(j);
      break;
    case CellLocation::Face3:
      X3 = coordinates_multi_(b).x3f(k);
      break;
    case CellLocation::Corn:
      X1 = coordinates_multi_(b).x1f(i);
      X2 = coordinates_multi_(b).x2f(j);
      X3 = coordinates_multi_(b).x3f(k);
      break;
    default: // CellLocation::Cent:
      break;
    }
    return;
  }

  KOKKOS_INLINE_FUNCTION
  void GetX(CellLocation loc, int k, int j, int i, Real &X1, Real &X2,
            Real &X3) const {
    PARTHENON_REQUIRE(
        !multiple_coords_,
        "Must have set only one Coordinates_t object to resolve ambiguities");
    // defaults
    X1 = coordinates_single_.x1v(i);
    X2 = coordinates_single_.x2v(j);
    X3 = coordinates_single_.x3v(k);
    // overwrite
    switch (loc) {
    case CellLocation::Face1:
      X1 = coordinates_single_.x1f(i);
      break;
    case CellLocation::Face2:
      X2 = coordinates_single_.x2f(j);
      break;
    case CellLocation::Face3:
      X3 = coordinates_single_.x3f(k);
      break;
    case CellLocation::Corn:
      X1 = coordinates_single_.x1f(i);
      X2 = coordinates_single_.x2f(j);
      X3 = coordinates_single_.x3f(k);
      break;
    default: // CellLocation::Cent:
      break;
    }
    return;
  }

  bool multiple_coords_ = false;
  Coordinates_t coordinates_single_;
  ParArray1D<Coordinates_t> coordinates_multi_;
  System system_;
  Real X0_ = 0.;
};

class Minkowski {
public:
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, Real X0, Real X1, Real X2, Real X3) const {
    return 0.;
  }
  KOKKOS_INLINE_FUNCTION
  Real SpacetimeMetric(int mu, int nu, Real X0, Real X1, Real X2, Real X3) const {
    return (mu == nu) ? (mu == 0 ? -1. : 1) : 0.;
  }
  KOKKOS_INLINE_FUNCTION
  Real Metric(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    return SpacetimeMetric(l, m, X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    return (l == m) ? 1. : 0.;
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const { return 1.; }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const { return 1.; }

  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, Real X0, Real X1,
                             Real X2, Real X3) const {
    return 0.;
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, Real X0, Real X1, Real X2,
                        Real X3) const {
    return 0.;
  }
  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, Real X0, Real X1, Real X2, Real X3) const {
    return 0.;
  }

  KOKKOS_INLINE_FUNCTION
  Real C0(Real X0, Real X1, Real X2, Real X3) const { return X0; }
  KOKKOS_INLINE_FUNCTION
  Real C1(Real X0, Real X1, Real X2, Real X3) const { return X1; }
  KOKKOS_INLINE_FUNCTION
  Real C2(Real X0, Real X1, Real X2, Real X3) const { return X2; }
  KOKKOS_INLINE_FUNCTION
  Real C3(Real X0, Real X1, Real X2, Real X3) const { return X3; }
};

// Non-analytic coordinate systems can have constructors that take
// Parthenon variables or variable packs. The intent is these objects
// can be constructed and passed to device.

using CoordinateSystem = Variant<Analytic<Minkowski>>;

} // namespace Geometry

#endif // GEOMETRY_COORDINATE_SYSTEMS_HPP_
