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

#ifndef GEOMETRY_LUMPY_HPP_
#define GEOMETRY_LUMPY_HPP_

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

class Lumpy {
public:
  Lumpy() = default;
  KOKKOS_INLINE_FUNCTION
  Lumpy(const Real a, const Real k, const Real betax) : a_(a), k_(k), betax_(betax) {
    betasq_ = betax_*betax_;
  }

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    const Real alpha = GetAlpha(X1);
    return -alpha*alpha + betasq_;
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {
    for (int i = 0; i < NDSPACE; ++i)
      beta[i] = 0;
    beta[0] = betax_;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3,
                       Real g[NDFULL][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = 0; nu < NDFULL; ++nu) {
        if (mu == nu) {
          g[mu][nu] = (mu == 0 ? -1 : 1);
        } else {
          g[mu][nu] = 0;
        }
      }
    }
    const Real alpha = GetAlpha(X1);
    g[0][0] = betasq_ - alpha*alpha;
    g[1][0] = betax_;
    g[0][1] = g[1][0];
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = 0; nu < NDFULL; ++nu) {
        if (mu == nu) {
          g[mu][nu] = (mu == 0 ? -1 : 1);
        } else {
          g[mu][nu] = 0;
        }
      }
    }
    const Real alpha = GetAlpha(X1);
    const Real alphasq = pow(alpha,2);
    const Real alpham1 = alpha - 1.;
    g[0][0] = -1./(alphasq);
    g[1][0] = betax_/alphasq;
    g[0][1] = g[1][0];
    g[1][1] = (alpham1*alpham1 + 2*alpham1 - betasq_ + 1)/alphasq;
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3,
              Real gamma[NDSPACE][NDSPACE]) const {
    for (int i = 0; i < NDSPACE; ++i) {
      for (int j = 0; j < NDSPACE; ++j) {
        gamma[i][j] = (i == j ? 1 : 0);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    for (int i = 0; i < NDSPACE; ++i) {
      for (int j = 0; j < NDSPACE; ++j) {
        gamma[i][j] = (i == j ? 1 : 0);
      }
    }
    const Real alpha = GetAlpha(X1);
    const Real alphasq = pow(alpha,2);
    const Real alpham1 = alpha - 1.;
    gamma[0][0] = (alpham1*alpham1 + 2*alpham1 - betasq_ + 1)/alphasq;
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return 1.;
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    const Real alpha = GetAlpha(X1);
    const Real alpham1 = alpha - 1.;
    return sqrt(alpham1*alpham1 + 2*alpham1 + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = 0; nu < NDFULL; ++nu) {
        for (int sigma = 0; sigma < NDFULL; ++sigma) {
          Gamma[mu][nu][sigma] = 0;
        }
      }
    }
    const Real k2 = k_*k_;

    Gamma[1][0][0] = -a_*k2*(a_*k_*cos(k_*X1) + 1)*sin(k_*X1);
    Gamma[0][1][0] = a_*k2*(a_*k_*cos(k_*X1) + 1)*sin(k_*X1);
    Gamma[0][0][1] = a_*k2*(a_*k_*cos(k_*X1) + 1)*sin(k_*X1);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = 0; nu < NDFULL; ++nu) {
        for (int sigma = 0; sigma < NDFULL; ++sigma) {
          dg[mu][nu][sigma] = 0;
        }
      }
    }
    const Real k2 = k_*k_;

    dg[0][0][1] = 2*a_*k2*(a_*k_*cos(k_*X1) + 1)*sin(k_*X1);
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    for (int mu = 0; mu < NDFULL; ++mu) {
      da[mu] = 0;
    }
    const Real k2 = k_*k_;

    da[1] = -a_*k2*sin(k_*X1)/(a_*k_*cos(k_*X1) + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1;
    C[2] = X2;
    C[3] = X3;
  }

  private:
    Real a_;
    Real k_;
    Real betax_;
    Real betasq_;

  KOKKOS_INLINE_FUNCTION
  Real GetAlpha(Real X1) const {
    return 1. + a_*k_*cos(k_*X1);
  }
};

using LumpyMeshBlock = Analytic<Lumpy, IndexerMeshBlock>;
using LumpyMesh = Analytic<Lumpy, IndexerMesh>;

using CLumpyMeshBlock = CachedOverMeshBlock<LumpyMeshBlock>;
using CLumpyMesh = CachedOverMesh<LumpyMesh>;

template <>
void Initialize<LumpyMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CLumpyMeshBlock>(ParameterInput *pin,
                                     StateDescriptor *geometry);

} // namespace Geometry

#endif // GEOMETRY_LUMPY_HPP_
