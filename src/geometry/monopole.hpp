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
#include "monopole_gr/monopole_gr.hpp"
#include "phoebus_utils/linear_algebra.hpp"

namespace Geometry {

/*
 * Metric from Monopole GR module.
 * Assumes spherical symmetry for the spacetime.
 * This class also assumes spherical coordinates for the fluid sector.

 * Since the metric and fluid radial grids don't necessarily align, we
 * interpolate the metric quantities and use the AnalyticSystem
 * modifier to fill in the missing functionality.
 */
class MonopoleSph {
 public:
  MonopoleSph() = default;
  KOKKOS_INLINE_FUNCTION
  MonopoleSph(const MonopoleGR::Hypersurface_t &hypersurface,
              const MonopoleGR::Alpha_t &alpha, const MonopoleGR::Beta_t &beta,
              const MonopoleGR::Gradients_t &gradients, const MonopoleGR::Radius &rgrid)
      : hypersurface_(hypersurface), alpha_(alpha), beta_(beta), gradients_(gradients),
        rgrid_(rgrid) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    return MonopoleGR::Interpolate(r, alpha_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    const Real r = std::abs(X1);
    LinearAlgebra::SetZero(beta, NDSPACE);
    beta[0] = MonopoleGR::Interpolate(r, beta_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real a2 = a * a;

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -alpha * alpha + a2 * beta * beta;
    g[0][1] = g[1][0] = a2 * beta;
    g[1][1] = a2;
    g[2][2] = r2;
    g[3][3] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              g[NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real ir2 = Utils::ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real ialpha2 = Utils::ratio(1., alpha * alpha);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -ialpha2;
    g[0][1] = g[1][0] = beta * ialpha2;
    g[1][1] = Utils::ratio(1., a * a) - beta * beta * ialpha2;
    g[2][2] = ir2;
    g[3][3] = ir2 * Utils::ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, g[NDSPACE][NDSPACE]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = a * a;
    g[1][1] = r2;
    g[2][2] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, g[NDSPACE][NDSPACE]) const {
    const Real r = std::abs(X1);
    const Real ir2 = Utils::ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = Utils::ratio(1., a * a);
    g[1][1] = ir2;
    g[2][2] = ir2 * Utils::ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    return a * r * r * std::abs(sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    return alpha * DetGamma(X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real th = X2;
    const Real sth = std::sin(th);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real K =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::K);
  }

 private:
  MonopoleGR::Hypersurface_t hypersurface_;
  MonopoleGR::Alpha_t alpha_;
  MonopoleGR::Beta_t beta_;
  MonopoleGR::Gradients_t gradients_;
  MonopoleGR::Radius rgrid_;
}

} // namespace Geometry
