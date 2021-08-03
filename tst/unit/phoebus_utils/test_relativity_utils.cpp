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

#include "geometry/geometry.hpp"

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

// Test utils
#include "test_utils.hpp"

// Relativity utils
#include "phoebus_utils/relativity_utils.hpp"

using namespace phoebus;
using namespace Geometry;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::ParArray1D;
using parthenon::Real;

TEST_CASE("RELATIVITY", "[relativity_utils]") {
  GIVEN("A three-velocity and a coordinate system") {
    auto mb = GetDummyMeshBlock();
    auto rc = GetDummyMeshBlockData(mb);
    auto system = GetCoordinateSystem(rc.get());
    const Real v[3] = {0., 0., 0.};
    const Real ucon_ref[4] = {1., 0., 0., 0.};
    CellLocation loc = CellLocation::Cent;
    int k = 0;
    int j = 0;
    int i = 0;
    THEN("The four-velocity is correctly calculated") {
      int n_wrong = 4;
      printf("Entering kokkos:\n");
      Kokkos::parallel_reduce(
          "get n wrong", 0,
          KOKKOS_LAMBDA(const int i, int &update) {
            Real ucon[4];
            GetFourVelocity(v, system, loc, k, j, i, ucon);
            SPACETIMELOOP(mu) {
              if (SoftEquiv(ucon[mu], ucon_ref[mu])) {
                update -= 1;
              }
            }
          }, n_wrong);
      REQUIRE(n_wrong == 0);
    }
  }
}
