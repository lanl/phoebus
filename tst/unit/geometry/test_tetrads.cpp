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

constexpr int NDFULL = CoordinateSystem::NDFULL;

TEST_CASE("Lorentz boosting", "[geometry][tetrads]") {
  GIVEN("A timelike observer Ucon") {
    //auto geom = Geometry::GetCoordinateSystem(rc);

    CoordinateSystem system = Analytic<Minkowski>();
     THEN("We can boost into the frame of a timelike observer Ucon") {

    Kokkos::parallel_for("Lorentz boost", 1,
    KOKKOS_LAMBDA(const int i) {
    Real lorentz = 2.;
    Real Ucon[NDFULL] = {-lorentz, sqrt(-1. + lorentz * lorentz), 0, 0};
    Real Trial[NDFULL] = {0, 1, 0, 0};

    Real Gcov[NDFULL][NDFULL];
    for (int mu = 0; mu < NDFULL; mu++) {
      for (int nu = 0; nu < NDFULL; nu++) {
        Gcov[mu][nu] = system.SpacetimeMetric(mu, nu, CellLocation::Face1, i, 0, 0, 0);
      printf("Gcov[%i][%i] = %e\n", mu, nu, Gcov[mu][nu]);
      }
    }

        Geometry::Tetrads Tetrads(Ucon, Trial, Gcov);

    });

     }
  }
}
