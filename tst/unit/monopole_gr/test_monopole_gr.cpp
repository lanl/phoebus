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
#include <limits>

// External includes
#include "catch2/catch.hpp"

// parthenon includes
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"

// monopole_gr includes
#include "monopole_gr/monopole_gr_base.hpp"
#include "monopole_gr/monopole_gr_interface.hpp"
#include "tov/tov.hpp"

using namespace parthenon::package::prelude;
using Duration_t = std::chrono::microseconds;

constexpr int NPOINTS = 1024 + 1;
constexpr Real ROUT = 256;
constexpr Real EPS = 1e-2;

KOKKOS_INLINE_FUNCTION
Real Gaussian(const Real x, const Real a, const Real b, const Real c) {
  return a * std::exp(-(x - b) * (x - b) / (2 * c * c));
}
KOKKOS_INLINE_FUNCTION
Real FractionalError(const Real a, const Real b) {
  Real diff = b - a;
  Real avg = 2. * (std::abs(a) + std::abs(b));
  return (std::abs(diff) < avg) ? diff / (avg + EPS) : diff;
}

TEST_CASE("monopole_gr is disabled by default", "[MonopoleGR]") {
  GIVEN("A parthenon input object without enabling MonopoleGR") {
    parthenon::ParameterInput in;
    parthenon::ParameterInput *pin = &in;

    WHEN("monopole_gr is initialized") {
      auto pkg = MonopoleGR::Initialize(pin);
      THEN("It is disabled by default") {
        auto &params = pkg->AllParams();
        bool enable_monopole_gr = params.Get<bool>("enable_monopole_gr");
        REQUIRE(!enable_monopole_gr);
      }
    }
  }
}

TEST_CASE("Working with monopole_gr Grids", "[MonopoleGR]") {
  GIVEN("monopole_gr Initialized appropriately") {
    parthenon::ParameterInput in;
    parthenon::ParameterInput *pin = &in;

    pin->SetBoolean("monopole_gr", "enabled", true);
    pin->SetInteger("monopole_gr", "npoints", NPOINTS);
    pin->SetReal("monopole_gr", "rout", ROUT);

    pin->SetReal("parthenon/mesh", "x1max", ROUT);
    pin->SetReal("parthenon/mesh", "x2max", M_PI);
    pin->SetReal("parthenon/mesh", "x3max", 2 * M_PI);

    auto pkg = MonopoleGR::Initialize(pin);
    auto &params = pkg->AllParams();

    auto enabled = params.Get<bool>("enable_monopole_gr");
    auto npoints = params.Get<int>("npoints");
    auto rin = params.Get<Real>("rin");
    auto rout = params.Get<Real>("rout");
    auto radius = params.Get<MonopoleGR::Radius>("radius");

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

    auto hypersurface = params.Get<MonopoleGR::Hypersurface_t>("hypersurface");
    auto hypersurface_h = params.Get<MonopoleGR::Hypersurface_host_t>("hypersurface_h");
    auto matter = params.Get<MonopoleGR::Matter_t>("matter");
    auto matter_h = params.Get<MonopoleGR::Matter_host_t>("matter_h");

    const int iA = MonopoleGR::Hypersurface::A;
    const int iK = MonopoleGR::Hypersurface::K;
    const int iRHO = MonopoleGR::Matter::RHO;
    const int iJ = MonopoleGR::Matter::J_R;
    const int iS = MonopoleGR::Matter::trcS;
    const int iSrr = MonopoleGR::Matter::Srr;

    WHEN("We set the matter fields to zero") {
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid to 0",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            matter(iRHO, i) = 0;
            matter(iJ, i) = 0;
            matter(iS, i) = 0;
            matter(iSrr, i) = 0;
          });
      THEN("We can integrate along the hypersurface, with initial guesses A=1, K=0") {
        MonopoleGR::IntegrateHypersurface(pkg.get());
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

      Real amp = 1e-5;
      Real mu = 50;
      Real sigma = 15;
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "Set matter grid, stationary",
          parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
            Real r = radius.x(i);
            Real rho = Gaussian(r, amp, mu, sigma);
            matter(iRHO, i) = rho;
            matter(iJ, i) = -r * 1e-2 * rho;
            matter(iS, i) = 3 * (Gamma_eos - 1.) * rho * eps_eos;
            matter(iSrr, i) = (Gamma_eos - 1.) * rho * eps_eos;
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
              if (matter(iJ, i) != -r * 1e-2 * rho) {
                nw += 1;
              }
              if (matter(iS, i) != 3 * (Gamma_eos - 1.) * rho * eps_eos) {
                nw += 1;
              }
              if (matter(iSrr, i) != (Gamma_eos - 1.) * rho * eps_eos) {
                nw += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
      WHEN("We integrate the hypersurface") {
        auto start = std::chrono::high_resolution_clock::now();
        MonopoleGR::MatterToHost(pkg.get(), false);
        MonopoleGR::IntegrateHypersurface(pkg.get());
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<Duration_t>(stop - start);
        printf("Time for integrate hypersurface with %d points = %ld microseconds\n",
               NPOINTS, duration.count());
        THEN("We can tridiagonal solve for alpha") {
          start = std::chrono::high_resolution_clock::now();
          MonopoleGR::LinearSolveForAlpha(pkg.get());
          MonopoleGR::SpacetimeToDevice(pkg.get());
          stop = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<Duration_t>(stop - start);
          printf("Time for linear solve with %d points = %ld microseconds\n", NPOINTS,
                 duration.count());
          AND_THEN("We can output the solver data") {
            MonopoleGR::DumpToTxt("monopole_gr.dat", pkg.get());
          }
        }
      }
    }
  }
}

TEST_CASE("The solution of MonopoleGR matches TOV", "[MonopoleGR]") {
  GIVEN("TOV, EOS, and MonopoleGR packages initialized appropriately") {
    parthenon::ParameterInput in;
    parthenon::ParameterInput *pin = &in;

    pin->SetBoolean("monopole_gr", "enabled", true);
    pin->SetInteger("monopole_gr", "npoints", 16 * NPOINTS);
    pin->SetReal("monopole_gr", "rout", 250);

    pin->SetReal("parthenon/mesh", "x1max", ROUT);
    pin->SetReal("parthenon/mesh", "x2max", M_PI);
    pin->SetReal("parthenon/mesh", "x3max", 2 * M_PI);

    pin->SetBoolean("tov", "enabled", true);
    pin->SetReal("tov", "Pc", 1e-2);
    pin->SetReal("tov", "entropy", 8);

    // P = (Gamma - 1) rho e = (Gamma - 1) rho (Cv T)
    // P = K rho^Gamma with e set to force it
    Real Gamma = 2; // degenerate matter
    Real gm1 = Gamma - 1;
    Real Cv = gm1; // sets m/kb = 1
    pin->SetString("eos", "type", "IdealGas");
    pin->SetReal("eos", "Gamma", Gamma);
    pin->SetReal("eos", "Cv", Cv);

    auto monopole_pkg = MonopoleGR::Initialize(pin);
    auto tov_pkg = TOV::Initialize(pin);
    auto eos_pkg = Microphysics::EOS::Initialize(pin);

    WHEN("We integrate the TOV equations") {
      TOV::IntegrateTov(tov_pkg.get(), monopole_pkg.get(), eos_pkg.get());
      THEN("We can solve for the metric") {
        MonopoleGR::MatterToHost(monopole_pkg.get(), false);
        MonopoleGR::IntegrateHypersurface(monopole_pkg.get());
        MonopoleGR::LinearSolveForAlpha(monopole_pkg.get());
        MonopoleGR::SpacetimeToDevice(monopole_pkg.get());
        AND_THEN("The metric is the same as the TOV metric") {
          TOV::IntegrateTov(tov_pkg.get(), monopole_pkg.get(), eos_pkg.get());

          using MonopoleGR::Alpha_t;
          using MonopoleGR::Gradients_t;
          using MonopoleGR::Hypersurface_t;
          using TOV::State_t;

          auto radius = monopole_pkg->Param<MonopoleGR::Radius>("radius");
          auto hypersurface = monopole_pkg->Param<Hypersurface_t>("hypersurface");
          auto alpha_array = monopole_pkg->Param<Alpha_t>("lapse");
          auto gradients = monopole_pkg->Param<Gradients_t>("gradients");
          auto tov_state = tov_pkg->Param<State_t>("tov_state");
          int npoints = monopole_pkg->Param<int>("npoints");

          int nwrong = 0;
          int im = TOV::M;
          int iphi = TOV::PHI;
          parthenon::par_reduce(
              parthenon::loop_pattern_flatrange_tag,
              "check TOV solution against monopole", parthenon::DevExecSpace(), 0,
              npoints - 1,
              KOKKOS_LAMBDA(const int i, int &nw) {
                Real r = radius.x(i);
                Real m = tov_state(im, i);
                Real phi = tov_state(iphi, i);
                Real tov_grr = 1. / (1 - 2 * (m / r));
                Real tov_alpha = std::exp(phi);

                Real a = hypersurface(MonopoleGR::Hypersurface::A, i);
                Real grr = a * a;
                Real K = hypersurface(MonopoleGR::Hypersurface::K, i);
                Real alpha = alpha_array(i);

                Real dadt = gradients(MonopoleGR::Gradients::DADT, i);
                Real dalphadt = gradients(MonopoleGR::Gradients::DALPHADT, i);
                Real dKdt = gradients(MonopoleGR::Gradients::DKDT, i);
                Real dbetadt = gradients(MonopoleGR::Gradients::DBETADT, i);

                if (FractionalError(grr, tov_grr) > EPS) {
                  nw += 1;
                }
                if (FractionalError(alpha, tov_alpha) > EPS) {
                  nw += 1;
                }
                if (K > EPS) {
                  nw += 1;
                }
                if (dadt > EPS) {
                  nw += 1;
                }
                if (dalphadt > EPS) {
                  nw += 1;
                }
                if (dKdt > EPS) {
                  nw += 1;
                }
                if (dbetadt > EPS) {
                  nw += 1;
                }
              },
              nwrong);
          REQUIRE(nwrong == 0);
        }
        AND_THEN("We can output the solver data") {
          MonopoleGR::DumpToTxt("tov.dat", monopole_pkg.get());
        }
      }
    }
  }
}
