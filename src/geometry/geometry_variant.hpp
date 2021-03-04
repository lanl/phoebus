#ifndef GEOMETRY_GEOMETRY_VARIANT_HPP_
#define GEOMETRY_GEOMETRY_VARIANT_HPP_

// parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// TODO(JMM): Fix this include
#include "../../external/singularity-eos/utils/variant/include/mpark/variant.hpp"

#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

using namespace parthenon::package::prelude;

namespace Geometry {

// replace this with std::variant when the time comes
template <typename... Ts> using geo_variant = mpark::variant<Ts...>;

template <typename... Systems> class Variant {

public:
  template <
      typename Choice,
      typename std::enable_if<
          !std::is_same<Variant, typename std::decay<Choice>::type>::value,
          bool>::type = true>
  KOKKOS_FUNCTION Variant(Choice &&choice)
      : system_(std::forward<Choice>(choice)) {}

  Variant() = default;

  template <
      typename Choice,
      typename std::enable_if<
          !std::is_same<Variant, typename std::decay<Choice>::type>::value,
          bool>::type = true>
  KOKKOS_FUNCTION Variant &operator=(Choice &&choice) {
    system_ = std::forward<Choice>(choice);
    return *this;
  }

  template <
      typename Choice,
      typename std::enable_if<
          !std::is_same<Variant, typename std::decay<Choice>::type>::value,
          bool>::type = true>
  Choice get() {
    return mpark::get<Choice>(system_);
  }

  // TODO(JMM): these could be variatically templated, but I think
  // it's valuable writing it out to enforce an explicit API

  // TODO(JMM): Note indexing! l,m,n range from 1 to 3,
  // corresponding to X1 through X3.

  // TODO(JMM): Functions of coordinates take all 4, including time.
  // Functions of index and cell location assume the same
  // time-centering variables. This may need to be re-thought. We
  // could instead pass X0 in even for the k,j,i versions.

  // Metric components
  // ======================================================================
  // alpha
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.Lapse(X0, X1, X2, X3); },
        system_);
  }
  // mesh version
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.Lapse(loc, k, j, i); },
        system_);
  }
  // meshblock version
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int b, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.Lapse(loc, b, k, j, i); },
        system_);
  }

  // beta^i
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          system.ContravariantShift(X0, X1, X2, X3, beta);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          system.ContravariantShift(loc, k, j, i, beta);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int b, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          system.ContravariantShift(loc, b, k, j, i, beta);
        },
        system_);
  }

  // gamma_{ij}
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3,
              Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) { system.Metric(X0, X1, X2, X3, gamma); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) { system.Metric(loc, k, j, i, gamma); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int b, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) { system.Metric(loc, b, k, j, i, gamma); },
        system_);
  }

  // gamma^{ij}
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricInverse(X0, X1, X2, X3, gamma);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricInverse(loc, k, j, i, gamma);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int b, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricInverse(loc, b, k, j, i, gamma);
        },
        system_);
  }

  // g_{mu nu}
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3,
                       Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetric(X0, X1, X2, X3, g);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetric(loc, k, j, i, g);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int b, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetric(loc, b, k, j, i, g);
        },
        system_);
  }
  // g^{mu nu}
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetricInverse(X0, X1, X2, X3, g);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetricInverse(loc, k, j, i, g);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int b, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.SpacetimeMetricInverse(loc, b, k, j, i, g);
        },
        system_);
  }
  // ======================================================================

  // Metric determinants
  // ======================================================================
  // sqrt(|gamma|)
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetGamma(loc, k, j, i); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int b, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetGamma(loc, b, k, j, i); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetGamma(X0, X1, X2, X3); },
        system_);
  }

  // sqrt(|g|)
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetG(loc, k, j, i); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(CellLocation loc, int b, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetG(loc, b, k, j, i); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetG(X0, X1, X2, X3); },
        system_);
  }
  // ======================================================================

  // Metric Derivatives
  // ======================================================================
  // Gamma_{mu nu sigma} // NOTE ALL INDICES DOWN
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.ConnectionCoefficient(X0, X1, X2, X3, Gamma);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.ConnectionCoefficient(loc, k, j, i, Gamma);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int b, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.ConnectionCoefficient(loc, b, k, j, i, Gamma);
        },
        system_);
  }

  // g_{mu nu, sigma}
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricDerivative(X0, X1, X2, X3, dg);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricDerivative(loc, k, j, i, dg);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int b, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricDerivative(loc, b, k, j, i, dg);
        },
        system_);
  }

  // (ln alpha)_{, mu}
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.GradLnAlpha(X0, X1, X2, X3, da);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int k, int j, int i,
                   Real da[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.GradLnAlpha(loc, k, j, i, da);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int b, int k, int j, int i,
                   Real da[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) {
          return system.GradLnAlpha(loc, b, k, j, i, da);
        },
        system_);
  }
  // ======================================================================

  // Coordinates
  // ======================================================================
  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) { return system.Coords(X0, X1, X2, X3, C); },
        system_);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int k, int j, int i, Real C[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) { return system.Coords(loc, k, j, i, C); },
        system_);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int b, int k, int j, int i,
              Real C[NDFULL]) const {
    return mpark::visit(
        [&](const auto &system) { return system.Coords(loc, b, k, j, i, C); },
        system_);
  }

private:
  geo_variant<Systems...> system_;
};

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_VARIANT_HPP_
