// ======================================================================
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
// ======================================================================

// stdlib includes

// External includes
#include "catch2/catch.hpp"

// parthenon includes
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// phoebus includes

// GR1D includes
#include "gr1d/gr1d.hpp"

using namespace parthenon::package::prelude;

constexpr int NPOINTS = 123; // just different from what's in grid.cpp
constexpr Real ROUT = 91;

TEST_CASE("GR1D is disabled by default", "[GR1D]") {
  GIVEN("A parthenon input object without enabling GR1D") {
    parthenon::ParameterInput in;
    parthenon::ParameterInput *pin = &in;

    WHEN("GR1D is initialized") {
      auto pkg = GR1D::Initialize(pin);
      THEN("It is disabled by default") {
        auto &params = pkg->AllParams();
        bool enable_gr1d = params.Get<bool>("enable_gr1d");
        REQUIRE(!enable_gr1d);

        REQUIRE(!params.hasKey("npoints"));
        REQUIRE(!params.hasKey("rin"));
        REQUIRE(!params.hasKey("rout"));
        REQUIRE(!params.hasKey("grids"));
      }
    }
  }
}

TEST_CASE("Working with GR1D Grids", "[GR1D]") {
  GIVEN("GR1D Initialized appropriately") {
    parthenon::ParameterInput in;
    parthenon::ParameterInput *pin = &in;

    pin->SetBoolean("GR1D", "enabled", true);
    pin->SetInteger("GR1D", "npoints", NPOINTS);
    pin->SetReal("GR1D", "rout", ROUT);

    auto pkg = GR1D::Initialize(pin);
    auto &params = pkg->AllParams();

    bool enabled = params.Get<bool>("enable_gr1d");
    int npoints = params.Get<int>("npoints");
    Real rin = params.Get<Real>("rin");
    Real rout = params.Get<Real>("rout");
    GR1D::Grids grids = params.Get<GR1D::Grids>("grids");

    THEN("The params match our expectations") {
      REQUIRE(enabled);
      REQUIRE(npoints == NPOINTS);
      REQUIRE(rin == 0);
      REQUIRE(rout == ROUT);
    }

    WHEN("We set the matter fields on the grid") {
      auto rho = grids.rho;
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set rho grid",
	  parthenon::DevExecSpace(),
	  0, npoints - 1,
	  KOKKOS_LAMBDA(const int i) { rho(i) = i; });
      THEN("We can retrieve this information") {
        Real sum = 0;
        Real target_sum = 0;
        for (int i = 0; i < npoints; ++i) {
          target_sum += i;
        }
        parthenon::par_reduce(
            parthenon::loop_pattern_flatrange_tag, "Check rho grid",
	    parthenon::DevExecSpace(),
	    0, npoints - 1,
            KOKKOS_LAMBDA(const int i, Real &slocal) { slocal += rho(i); },
	    sum);
        REQUIRE(sum == target_sum);
      }
    }
  }
}
