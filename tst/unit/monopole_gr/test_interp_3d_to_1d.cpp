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

constexpr Real EPS = 1e-5;
KOKKOS_INLINE_FUNCTION
Real GetDifference(Real a, Real b) {
  return 2*ratio(std::abs(b - a), std::abs(b) + std::abs(a));
}

TEST_CASE("Coordinates in spherical geometry", "[MonopoleGR]") {
  GIVEN("A parthenon coords object tuned to spherical geometry") {
    // TODO(JMM): This test is more of a sanity check than anything
    ParameterInput pin;
    RegionSize rs;
    rs.x1min = 0;
    rs.x1max = 100;
    rs.x2min = 0;
    rs.x2max = M_PI;
    rs.x3min = 0;
    rs.x3max = 2*M_PI;
    rs.nx1 = 10;
    rs.nx2 = 1;
    rs.nx3 = 1;
    const Real dr_true = (rs.x1max - rs.x1min)/(rs.nx1);
    const Real dth_true = (rs.x2max - rs.x2min)/(rs.nx2);
    const Real dph_true = (rs.x3max - rs.x3min)/(rs.nx3);
    Coordinates_t coords(rs, &pin);
    WHEN( "We loop through the grid" ) {
      int nwrong = 0;
      const int k = 0;
      const int j = 0;
      parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag,
        "check coords values",
        parthenon::DevExecSpace(),
        0, rs.nx1 - 1,
        KOKKOS_LAMBDA(const int i, int &nw) {
          Real r,th,ph,dr,dth,dph,dv;
          MonopoleGR::Interp3DTo1D::GetCoordsAndDerivsSph(
            k, j, i, coords, r, th, ph, dr, dth, dph, dv);
          Real idr = (1./12.)*(dr*dr*dr) + dr*(r*r);
          Real dv_true = std::sin(th)*dth*dph*idr;
          nw += ( GetDifference(dr, dr_true) > EPS );
          nw += ( GetDifference(dth, dth_true) > EPS );
          nw += ( GetDifference(dph, dph_true) > EPS );
          nw += ( GetDifference(dv, dv_true) > EPS );
        }, nwrong);
      THEN( "The coordinate values and volumes are what we expect" ) {
        REQUIRE( nwrong == 0 );
      }
    }
  }
}

inline bool CheckInBoundsHelper(const Real a, const Real r, const Real dr,
                                const bool inbounds) {
  int nwrong = 0;
  parthenon::par_reduce(
    parthenon::loop_pattern_flatrange_tag,
    "Check InBoundsHelper",
    parthenon::DevExecSpace(),
    0, 1,
    KOKKOS_LAMBDA(const int i, int &nw) {
      nw += (MonopoleGR::Interp3DTo1D::InBoundsHelper(a, r, dr) == inbounds);
    }, nwrong);
  return (nwrong > 0);
}
TEST_CASE("In Bounds", "[MonopoleGR]") {
  constexpr Real r = 10;
  constexpr Real dr = 2;
  SECTION( "The coordinate is to the left of the box" ) {
    // o  | ----- |
    REQUIRE( CheckInBoundsHelper(r - 2*dr, r, dr, false) );
  }
  SECTION( "The coordinate is to the right of the box" ) {
    // | ----- | o
    REQUIRE( CheckInBoundsHelper(r + 2*dr, r, dr, false) );
  }
  SECTION( "The coordinate is in the box" ) {
    // | --o--- |
    REQUIRE( CheckInBoundsHelper(r - dr/4., r, dr, true) );
  }
  SECTION( "The coordinate is at the left edge of the box" ) {
    // o ----- |
    REQUIRE( CheckInBoundsHelper(r - 0.5*dr, r, dr, true) );
  }
  SECTION( "The coordinate is at the right edge of the box" ) {
    // | ----- o
    REQUIRE( CheckInBoundsHelper(r + 0.5*dr, r, dr, false) );
  }
}

Real CheckGetVolIntersection(const Real rsmall, const Real drsmall,
                             const Real rbig, const Real drbig) {
  // hack par_reduce to do this
  Real sum = 0;
  constexpr int N = 1;
  parthenon::par_reduce(
    parthenon::loop_pattern_flatrange_tag,
    "Check GetVolIntersectionHelper",
    parthenon::DevExecSpace(),
    0, N-1,
    KOKKOS_LAMBDA(const int i, Real &s) {
      s += MonopoleGR::Interp3DTo1D::GetVolIntersectHelper(
        rsmall, drsmall, rbig, drbig);
    }, sum);
  return sum/static_cast<Real>(N);
}
TEST_CASE("Volume intersections", "[MonopoleGR]") {
  constexpr Real drsmall = 0.5;
  constexpr Real drbig = 1.0;
  constexpr Real rbig = 10.0;
  SECTION( "The small box does not intersect the big one." ) {
    Real rsmall = 1;
    Real vol = CheckGetVolIntersection(rsmall, drsmall, rbig, drbig);
    REQUIRE( std::abs(vol) <= EPS );
    rsmall = 19;
    vol = CheckGetVolIntersection(rsmall, drsmall, rbig, drbig);
    REQUIRE( std::abs(vol) <= EPS );
  }
  SECTION( "The small box is entirely inside the big one." ) {
    Real rsmall = rbig;
    Real vol = CheckGetVolIntersection(rsmall, drsmall, rbig, drbig);
    REQUIRE( GetDifference(vol, drsmall/drbig) <= EPS );
  }
  SECTION( "The small box is half inside the big one, left side." ) {
    Real rsmall = rbig - 0.5*drbig;
    Real vol = CheckGetVolIntersection(rsmall, drsmall, rbig, drbig);
    REQUIRE( GetDifference(vol, 0.5*drsmall/drbig) <= EPS );
  }
  SECTION( "The small box is half inside the big one, right side." ) {
    Real rsmall = rbig + 0.5*drbig;
    Real vol = CheckGetVolIntersection(rsmall, drsmall, rbig, drbig);
    REQUIRE( GetDifference(vol, 0.5*drsmall/drbig) <= EPS );
  }
}
