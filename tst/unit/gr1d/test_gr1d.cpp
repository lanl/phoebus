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

// C stdlib includes
#include <cmath>

// C++ stdlib includes
#include <chrono>

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
using Duration_t = std::chrono::microseconds;

constexpr int NPOINTS = 1024; // just different from what's in grid.cpp
constexpr Real ROUT = 512;
constexpr int NITERS_MAX   = 100000000;
constexpr int NITERS_CHECK = 1000000;

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
    pin->SetInteger("GR1D", "niters_check", NITERS_CHECK);

    auto pkg = GR1D::Initialize(pin);
    auto &params = pkg->AllParams();

    auto enabled = params.Get<bool>("enable_gr1d");
    auto npoints = params.Get<int>("npoints");
    auto rin = params.Get<Real>("rin");
    auto rout = params.Get<Real>("rout");
    auto radius = params.Get<GR1D::Radius>("radius");

    THEN("The params match our expectations") {
      REQUIRE(enabled);
      REQUIRE(npoints == NPOINTS);
      REQUIRE(rin == 0);
      REQUIRE(rout == ROUT);
      REQUIRE(params.Get<int>("niters_check") == NITERS_CHECK);
    }

    THEN("The radius object works as expected") {
      REQUIRE(radius.x(0) == rin);
      REQUIRE(radius.x(npoints - 1) == rout);
    }

    auto hypersurface = params.Get<GR1D::Hypersurface_t>("hypersurface");
    auto hypersurface_h = params.Get<GR1D::Hypersurface_host_t>("hypersurface_h");
    auto matter = params.Get<GR1D::Matter_t>("matter");
    auto matter_h = params.Get<GR1D::Matter_host_t>("matter_h");

    const int iA = GR1D::Hypersurface::A;
    const int iK = GR1D::Hypersurface::K;
    const int iRHO = GR1D::Matter::RHO;
    const int iJ = GR1D::Matter::J_R;
    const int iS = GR1D::Matter::trcS;

    WHEN("We set the matter fields to zero") {
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid to 0",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            matter(iRHO, i) = 0;
            matter(iJ, i) = 0;
            matter(iS, i) = 0;
          });
      THEN("We can integrate along the hypersurface, with initial guesses A=1, K=0") {
        GR1D::IntegrateHypersurface(pkg.get());
        AND_THEN("The solution converges identically to a=alpha=1, everything else 0") {
          Real error = 0;
          parthenon::par_reduce(
              parthenon::loop_pattern_flatrange_tag, "Check solution trivial",
              parthenon::DevExecSpace(), 0, npoints - 1,
              KOKKOS_LAMBDA(const int i, Real &e) {
                e += pow(hypersurface(iA, i) - 1, 2);
                e += pow(hypersurface(iK, i), 2);
              },
              error);
          error /= 3 * npoints;
          error = std::sqrt(error);
          REQUIRE(error <= 1e-12);
        }
      }
    }

    WHEN("We set the matter fields on the grid, for a stationary system") {
      constexpr Real eps_eos = 1e-2;
      constexpr Real Gamma_eos = 4. / 3.;

      Real amp = 1e-4;
      Real mu = 10;
      Real sigma = 10;
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid, stationary",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            Real r = radius.x(i);
            Real rho = Gaussian(r, amp, mu, sigma);
            matter(iRHO, i) = rho;
            matter(iJ, i) = r*1e-2 * rho;
            matter(iS, i) = 3 * (Gamma_eos - 1.) * rho * eps_eos;
          });
      THEN("We can retrieve this information") {
        int nwrong = 0;
        parthenon::par_reduce(
            parthenon::loop_pattern_flatrange_tag, "Check rho grid",
            parthenon::DevExecSpace(), 0, npoints - 1,
            KOKKOS_LAMBDA(const int i, int &nw) {
              Real r = radius.x(i);
              Real rho = Gaussian(r, amp, mu, sigma);
              if (matter(iRHO, i) != rho) {
                nw += 1;
              }
              if (matter(iJ, i) != r*1e-2 * rho) {
                nw += 1;
              }
              if (matter(iS, i) != 3 * (Gamma_eos - 1.) * rho * eps_eos) {
                nw += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
      WHEN("We integrate the hypersurface") {
	auto start = std::chrono::high_resolution_clock::now();
        GR1D::IntegrateHypersurface(pkg.get());
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<Duration_t>(stop-start);
	printf("Time for integrate hypersurface with %d points = %ld microseconds\n",
	       NPOINTS, duration.count());
        THEN("We can solve for alpha via Jacobi") {
	  auto error_tol = params.Get<Real>("error_tolerance");
	  auto niters_check = params.Get<int>("niters_check");
	  start = std::chrono::high_resolution_clock::now();
	  int niters;
          for (niters = 0; niters < NITERS_MAX; niters += niters_check) {
            GR1D::JacobiStepForLapse(pkg.get());
	    Real err = GR1D::LapseError(pkg.get());
	    printf("iter %d, err = %14e\n", niters, err);
            if (err < error_tol) break;
          }
	  stop = std::chrono::high_resolution_clock::now();
	  duration = std::chrono::duration_cast<Duration_t>(stop-start);
	  printf("Time for lapse with %d points, %d interations = %ld microseconds\n"
		 "=> %14e microseconds / iteration\n",
		 NPOINTS, niters, duration.count(),
		 static_cast<Real>(duration.count())/(niters));
          REQUIRE(GR1D::LapseConverged(pkg.get()));
          AND_THEN("We can output the solver data") {
            GR1D::DumpToTxt("gr1d.dat", pkg.get());
          }
        }
      }
    }
  }
}
