// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

// Phoebus includes
#include "geometry/coordinate_systems.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"

// A test that the geometry machinery with finite differences is
// working correctly.

namespace check_cached_geom {

KOKKOS_FORCEINLINE_FUNCTION
Real RelErr(const Real a, const Real b) {
  return std::min(2 * robust::ratio(std::abs(a - b), a + b), std::abs(a - b));
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using namespace Geometry;
  using AnalyticGeometry_t = Analytic<PHOEBUS_GEOMETRY, IndexerMeshBlock>;
  PARTHENON_REQUIRE(
      typeid(CachedOverMeshBlock<AnalyticGeometry_t>) == typeid(CoordSysMeshBlock),
      "This problem generator only supports analytic geometries cached over meshblocks");
  auto &rc = pmb->meshblock_data.Get();
  auto geom_cached = GetCoordinateSystem<CoordSysMeshBlock>(rc.get());
  auto geom_analytic = GetCoordinateSystem<AnalyticGeometry_t>(rc.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  Real max_err_grad = 0;
  auto loc = CellLocation::Cent;
  pmb->par_reduce(
      "Phoebus::ProblemGenerator::CheckCachedGeometry::Grads", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &le) {
        Real da_an[NDFULL];
        Real da_ca[NDFULL];
        Real dg_an[NDFULL][NDFULL][NDFULL];
        Real dg_ca[NDFULL][NDFULL][NDFULL];
        geom_analytic.GradLnAlpha(loc, k, j, i, da_an);
        geom_cached.GradLnAlpha(loc, k, j, i, da_ca);
        geom_analytic.MetricDerivative(loc, k, j, i, dg_an);
        geom_cached.MetricDerivative(loc, k, j, i, dg_ca);

        SPACETIMELOOP(mu) {
          Real diff = RelErr(da_an[mu], da_ca[mu]);
          le = std::max(le, diff);
        }
        SPACETIMELOOP3(mu, nu, sigma) {
          Real diff = RelErr(dg_an[mu][nu][sigma], dg_ca[mu][nu][sigma]);
          le = std::max(le, diff);
        }
      },
      Kokkos::Max<Real>(max_err_grad));
  printf("Maximum error in gradients = %.14e\n", max_err_grad);

  Real max_err_interp = 0;
  const int b = 0;
  const int X0 = 0;
  auto coords = pmb->coords;
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  pmb->par_reduce(
      "Phoebus::ProblemGenerator::CheckCachedGeometry::interps", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &le) {
        // Try to interpolate a little offset from the cell center in
        // each direction.
        const Real dx1 = 0.25 * coords.Dx(X1DIR);
        const Real dx2 = 0.25 * coords.Dx(X2DIR);
        const Real dx3 = 0.25 * coords.Dx(X3DIR);
        Real X1 = coords.x1v(k, j, i) + dx1;
        Real X2 = coords.x2v(k, j, i) + dx2;
        Real X3 = coords.x3v(k, j, i) + dx3;

        Real diff;

        // Lapse
        const Real alp_an = geom_analytic.Lapse(b, X0, X1, X2, X3);
        const Real alp_ca = geom_cached.Lapse(b, X0, X1, X2, X3);
        diff = RelErr(alp_an, alp_ca);
        le = std::max(le, diff);

        // Shift
        Real beta_an[NDSPACE];
        Real beta_ca[NDSPACE];
        geom_analytic.ContravariantShift(b, X0, X1, X2, X3, beta_an);
        geom_cached.ContravariantShift(b, X0, X1, X2, X3, beta_ca);
        SPACELOOP(d) {
          diff = RelErr(beta_an[d], beta_ca[d]);
          le = std::max(le, diff);
        }

        // spatial Metric
        Real gam_an[NDSPACE][NDSPACE];
        Real gam_ca[NDSPACE][NDSPACE];
        geom_analytic.Metric(b, X0, X1, X2, X3, gam_an);
        geom_cached.Metric(b, X0, X1, X2, X3, gam_ca);
        SPACELOOP2(m, n) {
          diff = RelErr(gam_an[m][n], gam_ca[m][n]);
          le = std::max(le, diff);
        }

        // Spatial metric inverse
        geom_analytic.MetricInverse(b, X0, X1, X2, X3, gam_an);
        geom_cached.MetricInverse(b, X0, X1, X2, X3, gam_ca);
        SPACELOOP2(m, n) {
          diff = RelErr(gam_an[m][n], gam_ca[m][n]);
          le = std::max(le, diff);
        }

        // Spacetime metric
        Real g_an[NDFULL][NDFULL];
        Real g_ca[NDFULL][NDFULL];
        geom_analytic.SpacetimeMetric(b, X0, X1, X2, X3, g_an);
        geom_cached.SpacetimeMetric(b, X0, X1, X2, X3, g_ca);
        SPACETIMELOOP2(m, n) {
          diff = RelErr(g_an[m][n], g_ca[m][n]);
          le = std::max(le, diff);
        }

        // Spacetime metric inverse
        geom_analytic.SpacetimeMetricInverse(b, X0, X1, X2, X3, g_an);
        geom_cached.SpacetimeMetricInverse(b, X0, X1, X2, X3, g_ca);
        SPACETIMELOOP2(m, n) {
          diff = RelErr(g_an[m][n], g_ca[m][n]);
          le = std::max(le, diff);
        }

        // DetGamma (DetG is for free since it uses lapse * detgamma)
        const Real det_an = geom_analytic.DetGamma(b, X0, X1, X2, X3);
        const Real det_ca = geom_cached.DetGamma(b, X0, X1, X2, X3);
        diff = RelErr(det_an, det_ca);
        le = std::max(le, diff);

        // Metric derivative (Connection for free since it's computed
        // from metric derivative)
        Real dg_an[NDFULL][NDFULL][NDFULL];
        Real dg_ca[NDFULL][NDFULL][NDFULL];
        geom_analytic.MetricDerivative(b, X0, X1, X2, X3, dg_an);
        geom_cached.MetricDerivative(b, X0, X1, X2, X3, dg_ca);
        SPACETIMELOOP3(m, n, s) {
          diff = RelErr(dg_an[m][n][s], dg_ca[m][n][s]);
          le = std::max(le, diff);
        }

        // GradLnAlpha
        Real da_an[NDFULL];
        Real da_ca[NDFULL];
        geom_analytic.GradLnAlpha(b, X0, X1, X2, X3, da_an);
        geom_cached.GradLnAlpha(b, X0, X1, X2, X3, da_ca);
        SPACETIMELOOP(d) {
          diff = RelErr(da_an[d], da_ca[d]);
          le = std::max(le, diff);
        }
      },
      Kokkos::Max<Real>(max_err_interp));
  printf("Maximum error in interpolation = %.14e\n", max_err_interp);

  std::exit(1);
}

} // namespace check_cached_geom
