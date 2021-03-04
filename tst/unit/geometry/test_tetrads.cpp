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
#include <geometry/coordinate_systems.hpp>
#include <geometry/tetrads.hpp>
#include <phoebus_utils/cell_locations.hpp>

using namespace Geometry;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::ParArray1D;
using parthenon::Real;
using parthenon::RegionSize;

KOKKOS_INLINE_FUNCTION
bool SoftEquiv(const Real &val, const Real &ref, const Real tol = 1.e-8,
               const bool ignore_small = true) {
  if (ignore_small) {
    if (fabs(val) < tol && fabs(ref) < tol) {
      return true;
    }
  }

  if (fabs(val - ref) < tol * fabs(ref) / 2) {
    return true;
  } else {
    return false;
  }
}

TEST_CASE("Tetrads", "[geometry][tetrads]") {
  GIVEN("A timelike observer Ucon") {
    // auto geom = Geometry::GetCoordinateSystem(rc);

    CoordinateSystem system = Analytic<Minkowski, IndexerMeshBlock>();
    THEN("We can boost into the frame of a timelike observer Ucon") {

      int n_wrong = 0;
      Kokkos::parallel_reduce(
          "Lorentz boost", 1,
          KOKKOS_LAMBDA(const int i, int &wrong) {
            Real lorentz = 2.;
            Real v = sqrt(1. - 1. / (lorentz * lorentz));
            Real theta = 2. / 3. * M_PI;
            Real phi = 1. / 4. * M_PI;
            //lorentz = 1.;
            //theta = M_PI/2.;
            //phi = 0.;
            Real TetradFrame[NDFULL] = {lorentz, lorentz * v * sin(theta) * cos(phi),
                                        lorentz * v * sin(theta) * sin(phi),
                                        lorentz * v * cos(theta)};

            // Real Ucon[NDFULL] = {-lorentz, sqrt(-1. + lorentz * lorentz), 0, 0};
            Real Trial[NDFULL] = {0, 1, 0, 0};

            Real VConCoord[NDFULL] = {1, 0, 0, 0};
            Real VCovCoord[NDFULL] = {-1, 0, 0, 0};
            Real VConTetrad[NDFULL];
            Real VCovTetrad[NDFULL];
            Real VConTetradReference[NDFULL] = {
                1.999999999999999112, -1.455213750217997370e+00,
                -8.072073527955754280e-01, 4.803844614152611436e-01};
            Real VCovTetradReference[NDFULL] = {
                -1.999999999999999112e+00, -1.455213750217997370e+00,
                -8.072073527955754280e-01, 4.803844614152611436e-01};
            Real VConCoordCompare[NDFULL];
            Real VCovCoordCompare[NDFULL];

            Real Gcov[NDFULL][NDFULL];
            system.SpacetimeMetric(CellLocation::Cent, 0, 0, 0, Gcov);

            Geometry::Tetrads Tetrads(TetradFrame, Trial, Gcov);
            Tetrads.CoordToTetradCon(VConCoord, VConTetrad);
            Tetrads.TetradToCoordCon(VConTetrad, VConCoordCompare);

            printf("TetradFrame\n");
            printf("%e %e %e %e\n", TetradFrame[0], TetradFrame[1], TetradFrame[2], TetradFrame[3]);
            printf("VConCoord\n");
            printf("%e %e %e %e\n", VConCoord[0], VConCoord[1], VConCoord[2],
                   VConCoord[3]);
            printf("VConTetrad\n");
            printf("%28.18e %28.18e %28.18e %28.18e\n", VConTetrad[0], VConTetrad[1],
                   VConTetrad[2], VConTetrad[3]);
            printf("VConCoordCompare\n");
            printf("%e %e %e %e\n", VConCoordCompare[0], VConCoordCompare[1],
                   VConCoordCompare[2], VConCoordCompare[3]);

            for (int mu = 0; mu < NDFULL; mu++) {
              if (!SoftEquiv(VConCoordCompare[mu], VConCoord[mu])) {
                wrong++;
              }
              if (!SoftEquiv(VConTetrad[mu], VConTetradReference[mu])) {
                wrong++;
              }
            }

            Tetrads.CoordToTetradCov(VCovCoord, VCovTetrad);
            Tetrads.TetradToCoordCov(VCovTetrad, VCovCoordCompare);
            printf("%28.18e %28.18e %28.18e %28.18e\n", VCovTetrad[0], VCovTetrad[1],
                   VCovTetrad[2], VCovTetrad[3]);

            for (int mu = 0; mu < NDFULL; mu++) {
              if (!SoftEquiv(VCovCoordCompare[mu], VCovCoord[mu])) {
                wrong++;
              }
              if (!SoftEquiv(VCovTetrad[mu], VCovTetradReference[mu])) {
                wrong++;
              }
            }
          },
          n_wrong);
      REQUIRE(n_wrong == 0);
    }
  }
}
