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

// stdlib lib includes
#include <cmath>

// external includes
#include "catch2/catch.hpp"

// Parthenon includes
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// phoebus includes
#include "monopole_gr/monopole_gr_utils.hpp"
#include "phoebus_utils/robust.hpp"

using robust::ratio;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::RegionSize;

KOKKOS_INLINE_FUNCTION
Real GetDifference(Real a, Real b) {
  return 2*ratio(std::abs(b - a), std::abs(b) + std::abs(a));
}

TEST_CASE("Coordinates in spherical geometry", "[MonopoleGR]") {
  GIVEN("A parthenon coords object tuned to spherical geometry") {
    ParameterInput pin;
    RegionSize rs;
    rs.x1min = 0;
    rs.x1max = 100;
    rs.x2min = 0;
    rs.x2max = M_PI;
    rs.x3min = 0;
    rs.x3max = 2*M_PI;
    rs.nx1 = 1000;
    rs.nx2 = 1;
    rs.nx3 = 1;
    Coordinates_t coords(rs, &pin);
  }
}


