#ifndef GEOMETRY_GEOMETRY_VARIANT_HPP_
#define GEOMETRY_GEOMETRY_VARIANT_HPP_

// parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// TODO(JMM): Fix this include
#include "../../external/singularity-eos/utils/variant/include/mpark/variant.hpp"

#include "phoebus_utils/cell_locations.hpp"

using namespace parthenon::package::prelude;

namespace Geometry {

// replace this with std::variant when the time comes
template <typename... Ts> using geo_variant = mpark::variant<Ts...>;

template <typename... Systems> class Variant {

public:
  static constexpr int NDSPACE = 3;
  static constexpr int NDFULL = NDSPACE + 1;

  template <
      typename Choice,
      typename std::enable_if<
          !std::is_same<Variant, typename std::decay<Choice>::type>::value,
          bool>::type = true>
  KOKKOS_FUNCTION Variant(Choice &&choice)
      : system_(std::forward<Choice>(choice)) {}

  Variant() noexcept = default;

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
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.Lapse(loc, k, j, i); },
        system_);
  }

  // beta^i
  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, Real X0, Real X1, Real X2, Real X3) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) {
          return system.ContravariantShift(l, X0, X1, X2, X3);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {
    for (int l = 0; l < NDSPACE; ++l) {
      beta[l] = ContravariantShift(l + 1, X0, X1, X2, X3);
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real ContravariantShift(int l, CellLocation loc, int k, int j, int i) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) {
          return system.ContravariantShift(l, loc, k, j, i);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    for (int l = 0; l < NDSPACE; ++l) {
      beta[l] = ContravariantShift(l + 1, loc, k, j, i);
    }
  }

  // gamma_{ij}
  KOKKOS_INLINE_FUNCTION Real Metric(int l, int m, Real X0, Real X1, Real X2,
                                     Real X3) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE && 1 <= m && m <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) { return system.Metric(l, m, X0, X1, X2, X3); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3,
              Real gamma[NDSPACE][NDSPACE]) const {
    for (int m = 0; m < NDSPACE; ++m) {
      for (int l = m; l < NDSPACE; ++l) {
        gamma[l][m] = Metric(l + 1, m + 1, X0, X1, X2, X3);
        gamma[m][l] = gamma[l][m];
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real Metric(int l, int m, CellLocation loc, int k, int j, int i) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE && 1 <= m && m <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) { return system.Metric(l, m, loc, k, j, i); },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    for (int l = 0; l < NDSPACE; ++l) {
      for (int m = 0; m < NDSPACE; ++m) {
        gamma[l][m] = Metric(l + 1, m + 1, loc, k, j, i);
      }
    }
  }

  // gamma^{ij}
  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, Real X0, Real X1, Real X2, Real X3) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE && 1 <= m && m <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricInverse(l, m, X0, X1, X2, X3);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    for (int m = 0; m < NDSPACE; ++m) {
      for (int l = m; l < NDSPACE; ++l) {
        gamma[l][m] = MetricInverse(l + 1, m + 1, X0, X1, X2, X3);
        gamma[m][l] = gamma[l][m];
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricInverse(int l, int m, CellLocation loc, int k, int j,
                     int i) const {
    PARTHENON_DEBUG_REQUIRE(1 <= l && l <= NDSPACE && 1 <= m && m <= NDSPACE,
                            "Indices must be spacelike");
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricInverse(l, m, loc, k, j, i);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    for (int m = 0; m < NDSPACE; ++m) {
      for (int l = m; l < NDSPACE; ++l) {
        gamma[l][m] = MetricInverse(l + 1, m + 1, loc, k, j, i);
        gamma[m][l] = gamma[l][m];
      }
    }
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
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.DetG(X0, X1, X2, X3); },
        system_);
  }
  // ======================================================================

  // Metric Derivatives
  // ======================================================================
  // Gamma^l_{mn}
  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, Real X0, Real X1,
                             Real X2, Real X3) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL && 0 <= nu &&
                                nu <= NDFULL && 0 <= sigma && sigma <= NDFULL,
                            "Indices must be spacetime");
    return mpark::visit(
        [&](const auto &system) {
          return system.ConnectionCoefficient(mu, nu, sigma, X0, X1, X2, X3);
        },
        system_);
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = 0; nu < NDFULL; ++nu) {
        for (int sigma = nu; sigma < NDFULL; ++sigma) {
          Gamma[mu][nu][sigma] =
              ConnectionCoefficient(mu, nu, sigma, X0, X1, X2, X3);
          Gamma[mu][sigma][nu] = Gamma[mu][nu][sigma];
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real ConnectionCoefficient(int mu, int nu, int sigma, CellLocation loc, int k,
                             int j, int i) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL && 0 <= nu &&
                                nu <= NDFULL && 0 <= sigma && sigma <= NDFULL,
                            "Indices must be spacetime");
    return mpark::visit(
        [&](const auto &system) {
          return system.ConnectionCoefficient(mu, nu, sigma, loc, k, j, i);
        },
        system_);
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int k, int j, int i,
                             Real Gamma[NDSPACE][NDSPACE][NDSPACE]) const {
    for (int mu = 0; mu < NDSPACE; ++mu) {
      for (int nu = 0; nu < NDSPACE; ++nu) {
        for (int sigma = nu; sigma < NDSPACE; ++sigma) {
          Gamma[mu][nu][sigma] =
              ConnectionCoefficient(mu, nu, sigma, loc, k, j, i);
          Gamma[mu][sigma][nu] = Gamma[mu][nu][sigma];
        }
      }
    }
  }

  // g_{nu l, mu}
  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, Real X0, Real X1, Real X2,
                        Real X3) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL && 0 <= nu &&
                                nu <= NDFULL && 1 <= l && l <= NDSPACE,
                            "Indices must be correct");
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricDerivative(mu, l, nu, X0, X1, X2, X3);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDSPACE][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int l = 0; l < NDSPACE; ++l) {
        for (int nu = 0; nu < NDFULL; ++nu) {
          dg[mu][l][nu] = MetricDerivative(mu, l + 1, nu, X0, X1, X2, X3);
        }
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real MetricDerivative(int mu, int l, int nu, CellLocation loc, int k, int j,
                        int i) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL && 0 <= nu &&
                                nu <= NDFULL && 1 <= l && l <= NDSPACE,
                            "Indices must be correct");
    return mpark::visit(
        [&](const auto &system) {
          return system.MetricDerivative(mu, l, nu, loc, k, j, i);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int k, int j, int i,
                        Real dg[NDFULL][NDSPACE][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int l = 0; l < NDSPACE; ++l) {
        for (int nu = 0; nu < NDFULL; ++nu) {
          dg[mu][l][nu] = MetricDerivative(mu, l + 1, nu, loc, k, j, i);
        }
      }
    }
  }

  // (ln alpha)_{, mu}
  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, Real X0, Real X1, Real X2, Real X3) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL,
                            "Indices must be spacetime");
    return mpark::visit(
        [&](const auto &system) {
          return system.GradLnAlpha(mu, X0, X1, X2, X3);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      da[mu] = GradLnAlpha(mu, X0, X1, X2, X3);
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real GradLnAlpha(int mu, CellLocation loc, int k, int j, int i) const {
    PARTHENON_DEBUG_REQUIRE(0 <= mu && mu <= NDFULL,
                            "Indices must be spacetime");
    return mpark::visit(
        [&](const auto &system) {
          return system.GradLnAlpha(mu, loc, k, j, i);
        },
        system_);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int k, int j, int i,
                   Real da[NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      da[mu] = GradLnAlpha(mu, loc, k, j, i);
    }
  }
  // ======================================================================

  // Coordinates
  // ======================================================================
  KOKKOS_INLINE_FUNCTION
  Real C0(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.C1(X0, X1, X2, X3); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C1(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.C1(X0, X1, X2, X3); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C2(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.C2(X0, X1, X2, X3); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C3(Real X0, Real X1, Real X2, Real X3) const {
    return mpark::visit(
        [&](const auto &system) { return system.C3(X0, X1, X2, X3); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = C0(X0, X1, X2, X3);
    C[1] = C1(X0, X1, X2, X3);
    C[2] = C2(X0, X1, X2, X3);
    C[3] = C3(X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  Real C0(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.C0(loc, k, j, i); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C1(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.C1(loc, k, j, i); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C2(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.C2(loc, k, j, i); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  Real C3(CellLocation loc, int k, int j, int i) const {
    return mpark::visit(
        [&](const auto &system) { return system.C3(loc, k, j, i); }, system_);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int k, int j, int i, Real C[NDFULL]) const {
    C[0] = C0(loc, k, j, i);
    C[1] = C1(loc, k, j, i);
    C[2] = C2(loc, k, j, i);
    C[3] = C3(loc, k, j, i);
  }

private:
  geo_variant<Systems...> system_;
};

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_VARIANT_HPP_
