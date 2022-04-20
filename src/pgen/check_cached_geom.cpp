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
  return 2 * robust::ratio(std::abs(a - b), a + b);
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

  Real max_err;
  auto loc = CellLocation::Cent;
  pmb->par_reduce(
      "Phoebus::ProblemGenerator::CheckCachedGeometry", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
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
      Kokkos::Max<Real>(max_err));
  printf("Maximum error = %.14e\n", max_err);
  std::exit(1);
}

} // namespace check_cached_geom
