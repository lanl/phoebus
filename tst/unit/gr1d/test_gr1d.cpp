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
#include <cmath>

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

KOKKOS_INLINE_FUNCTION
Real Gaussian(const Real x, const Real a, const Real b, const Real c) {
  return a * std::exp(-(x - b) * (x - b) / (2 * c * c));
}

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
    pin->SetReal("GR1D", "niters_check", 1);

    auto pkg = GR1D::Initialize(pin);
    auto &params = pkg->AllParams();

    auto enabled = params.Get<bool>("enable_gr1d");
    auto npoints = params.Get<int>("npoints");
    auto rin = params.Get<Real>("rin");
    auto rout = params.Get<Real>("rout");
    auto radius = params.Get<GR1D::Radius>("radius");
    auto grids = params.Get<GR1D::Grids>("grids");

    THEN("The params match our expectations") {
      REQUIRE(enabled);
      REQUIRE(npoints == NPOINTS);
      REQUIRE(rin == 0);
      REQUIRE(rout == ROUT);
    }

    THEN("The radius object works as expected") {
      REQUIRE(radius.x(0) == rin);
      REQUIRE(radius.x(npoints - 1) == rout);
    }

    auto a = grids.a;
    auto dadr = grids.dadr;
    auto K_rr = grids.K_rr;
    auto dKdr = grids.dKdr;
    auto alpha = grids.alpha;
    auto dalphadr = grids.dalphadr;
    auto aleph = grids.aleph;
    auto dalephdr = grids.dalephdr;
    auto rho = grids.rho;
    auto j = grids.j_r;
    auto S = grids.trcS;
    
    WHEN("We set the matter fields on the grid, for a stationary system") {
      constexpr Real eps_eos = 1e-2;
      constexpr Real Gamma_eos = 4. / 3.;

      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid, stationary",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            Real r = radius.x(i);
            rho(i) = Gaussian(r, 1, 0, 10);
            j(i) = 0;
            S(i) = 3 * (Gamma_eos - 1.) * rho(i) * eps_eos;
          });
      THEN("We can retrieve this information") {
        int nwrong = 0;
        parthenon::par_reduce(
            parthenon::loop_pattern_flatrange_tag, "Check rho grid",
            parthenon::DevExecSpace(), 0, npoints - 1,
            KOKKOS_LAMBDA(const int i, int &nw) {
	      Real r = radius.x(i);
              if (rho(i) != Gaussian(r, 1, 0, 10)) nw += 1;
              if (j(i) != 0) nw += 1;
              if (S(i) != 3 * (Gamma_eos - 1.) * rho(i) * eps_eos) nw += 1;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }

    WHEN("We set the matter fields to zero") {
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid to 0",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            rho(i) = 0;
            j(i) = 0;
            S(i) = 0;
          });
      THEN("The iterative solver runs") {
	GR1D::IterativeSolve(pkg.get());
	AND_THEN("The solution converges to a=alpha=1, everything else 0") {
	  Real error = 0;
	  parthenon::par_reduce(parthenon::loop_pattern_flatrange_tag, "Check solution trivial",
				parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i, Real &e) {
				  e += (a(i) - 1);
				  //e += dadr(i);
				  e += K_rr(i);
				  //e += dKdr(i);
				  e + (alpha(i) - 1);
				  //e += dalphadr(i);
				  e += aleph(i);
				  //e += dalephdr(i);
				}, error);
	  std::cout << "error = " << error << std::endl;
	}
      }
    }
  }
}
