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

// stdlib includes
#include <cmath>

// external includes
#include "catch2/catch.hpp"
#include <Kokkos_Core.hpp>

// parthenon includes
#include <coordinates/coordinates.hpp>
#include <defs.hpp>
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>

// phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "geometry/tetrads.hpp"
#include <geometry/minkowski.hpp>
#include "../../test_utils.hpp"

using namespace Geometry;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::ParArray1D;
using parthenon::Real;
using parthenon::RegionSize;

TEST_CASE("Tetrads", "[geometry][tetrads]") {
  GIVEN("A timelike observer Ucon") {
    //CoordinateSystem system = Analytic<Minkowski, IndexerMeshBlock>();
    THEN("We can boost into the frame of a timelike observer Ucon") {
    //auto geom = Geometry::GetCoordinateSystem(rc);
    Analytic<Minkowski, IndexerMeshBlock> system;

      int n_wrong = 0;
      Kokkos::parallel_reduce(
          "Lorentz boost", 1,
          KOKKOS_LAMBDA(const int i, int &wrong) {
            Real lorentz = 2.;
            Real v = sqrt(1. - 1. / (lorentz * lorentz));
            Real theta = 2. / 3. * M_PI;
            Real phi = 1. / 4. * M_PI;
            Real TetradFrame[NDFULL] = {lorentz, lorentz * v * sin(theta) * cos(phi),
                                        lorentz * v * sin(theta) * sin(phi),
                                        lorentz * v * cos(theta)};

            Real Trial[NDFULL] = {0, 1, 0, 0};

            Real VconCoord[NDFULL] = {1, 0, 0, 0};
            Real VcovCoord[NDFULL] = {-1, 0, 0, 0};
            Real VconTetrad[NDFULL];
            Real VcovTetrad[NDFULL];
            Real VconTetradReference[NDFULL] = {
                1.999999999999999112, -1.455213750217997370e+00,
                -8.072073527955754280e-01, 4.803844614152611436e-01};
            Real VcovTetradReference[NDFULL] = {
                -1.999999999999999112e+00, -1.455213750217997370e+00,
                -8.072073527955754280e-01, 4.803844614152611436e-01};
            Real VconCoordCompare[NDFULL];
            Real VcovCoordCompare[NDFULL];

            Real Gcov[NDFULL][NDFULL];
            system.SpacetimeMetric(CellLocation::Cent, 0, 0, 0, Gcov);

            Geometry::Tetrads Tetrads(TetradFrame, Trial, Gcov);
            Tetrads.CoordToTetradCon(VconCoord, VconTetrad);
            Tetrads.TetradToCoordCon(VconTetrad, VconCoordCompare);

            for (int mu = 0; mu < NDFULL; mu++) {
              if (!SoftEquiv(VconCoordCompare[mu], VconCoord[mu])) {
                wrong++;
              }
              if (!SoftEquiv(VconTetrad[mu], VconTetradReference[mu])) {
                wrong++;
              }
            }

            Tetrads.CoordToTetradCov(VcovCoord, VcovTetrad);
            Tetrads.TetradToCoordCov(VcovTetrad, VcovCoordCompare);

            for (int mu = 0; mu < NDFULL; mu++) {
              if (!SoftEquiv(VcovCoordCompare[mu], VcovCoord[mu])) {
                wrong++;
              }
              if (!SoftEquiv(VcovTetrad[mu], VcovTetradReference[mu])) {
                wrong++;
              }
            }
          },
          n_wrong);
      REQUIRE(n_wrong == 0);
    }
  }
}
