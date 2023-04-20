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
#include "phoebus_utils/robust.hpp"

// coordinate system includes
#include <geometry/analytic_system.hpp>
#include <geometry/boyer_lindquist.hpp>
#include <geometry/cached_system.hpp>
#include <geometry/flrw.hpp>
#include <geometry/fmks.hpp>
#include <geometry/geometry_defaults.hpp>
#include <geometry/mckinney_gammie_ryan.hpp>
#include <geometry/minkowski.hpp>
#include <geometry/modified_system.hpp>
#include <geometry/spherical_kerr_schild.hpp>
#include <geometry/spherical_minkowski.hpp>

using namespace Geometry;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::ParArray1D;
using parthenon::Real;
using parthenon::RegionSize;
using robust::ratio;

constexpr int NTRIALS = 100;
constexpr int NG = 2;
constexpr int NX = 32 + 2 * NG;
constexpr Real EPS = 1e-5;

KOKKOS_INLINE_FUNCTION
Real GetDifference(Real a, Real b) {
  return 2 * ratio(std::abs(b - a), std::abs(b) + std::abs(a));
}

TEST_CASE("Minkowski Coordinates", "[geometry]") {
  GIVEN("A Cartesian Minkowski coordinate system") {
    Analytic<Minkowski, IndexerMeshBlock> system;
    THEN("The coordinate system can be called on device") {
      int n_wrong = 100; // > 0
      Kokkos::parallel_reduce(
          "get n wrong", NTRIALS,
          KOKKOS_LAMBDA(const int i, int &update) {
            // lapse
            Real alpha = system.Lapse(0, 0, 0, 0);
            if (alpha != 1.) update += 1;

            // shift
            Real beta[NDSPACE];
            system.ContravariantShift(CellLocation::Corn, 0, 0, 0, beta);
            for (int l = 0; l < NDSPACE; ++l) {
              if (beta[l] != 0.0) update += 1;
            }

            // metric
            Real gamma[NDSPACE][NDSPACE];
            system.Metric(CellLocation::Face1, 0, 0, 0, gamma);
            for (int l = 0; l < NDSPACE; l++) {
              for (int m = 0; m < NDSPACE; m++) {
                if (l == m && gamma[l][m] != 1.) update += 1;
                if (l != m && gamma[l][m] != 0.) update += 1;
              }
            }

            // inverse metric
            system.MetricInverse(CellLocation::Cent, 0, 0, 0, gamma);
            for (int l = 0; l < NDSPACE; l++) {
              for (int m = 0; m < NDSPACE; m++) {
                if (l == m && gamma[l][m] != 1.) update += 1;
                if (l != m && gamma[l][m] != 0.) update += 1;
              }
            }

            // metric determinants
            for (int k = 0; k < NX; ++k) {
              for (int j = 0; j < NX; ++j) {
                for (int ii = 0; ii < NX; ++ii) {
                  Real dgamma = system.DetGamma(CellLocation::Face2, k, j, ii);
                  if (dgamma != 1.0) update += 1;
                  Real dg = system.DetG(CellLocation::Face3, k, j, ii);
                  if (dg != 1.0) update += 1;
                }
              }
            }

            // connection coefficient
            Real Gamma[NDFULL][NDFULL][NDFULL];
            system.ConnectionCoefficient(0, 0, 0, 0, Gamma);
            for (int l = 0; l < NDFULL; ++l) {
              for (int m = 0; m < NDFULL; ++m) {
                for (int n = 0; n < NDFULL; ++n) {
                  if (Gamma[l][m][n] != 0.0) update += 1;
                }
              }
            }

            // grad g
            // g_{nu l, mu}
            Real dg[NDFULL][NDFULL][NDFULL];
            system.MetricDerivative(0, 0, 0, 0, dg);
            for (int nu = 0; nu < NDFULL; ++nu) {
              for (int l = 0; l < NDFULL; ++l) {
                for (int mu = 0; mu < NDFULL; ++mu) {
                  if (dg[nu][l][mu] != 0.0) update += 1;
                }
              }
            }

            // grad ln(alpha)
            Real grada[NDFULL];
            system.GradLnAlpha(CellLocation::Corn, 0, 0, 0, grada);
            for (int mu = 0; mu < NDFULL; ++mu) {
              if (grada[mu] != 0.0) update += 1;
            }
          },
          n_wrong);
      REQUIRE(n_wrong == 0);
    }
  }

  GIVEN("A ParArray1D of Coordinates_ts") {
    ParArray1D<Coordinates_t> coords("coords view", NTRIALS);
    auto coords_h = Kokkos::create_mirror_view(coords);
    for (int i = 0; i < NTRIALS; ++i) {
      coords_h(i) = Coordinates_t(); // or something more complicated
    }
    Kokkos::deep_copy(coords, coords_h);
    THEN("We can create a Minkowski coordinate system using this") {
      Real time = 3.0;
      IndexerMesh indexer(coords);
      Analytic<Minkowski, IndexerMesh> system(time, indexer);
      AND_THEN("The coordinate system can be called on device") {
        int n_wrong = 100; // > 0
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int b, int &update) {
              // Indexer
              Real X1, X2, X3;
              indexer.GetX(CellLocation::Cent, b, 0, 0, 0, X1, X2, X3);

              // shift
              Real beta[NDSPACE];
              system.ContravariantShift(CellLocation::Corn, b, 0, 0, 0, beta);
              for (int l = 0; l < NDSPACE; ++l) {
                if (beta[l] != 0.0) update += 1;
              }

              // metric
              Real gamma[NDSPACE][NDSPACE];
              system.Metric(CellLocation::Face1, b, 0, 0, 0, gamma);
              for (int l = 1; l < NDSPACE; l++) {
                for (int m = 1; m < NDSPACE; m++) {
                  if (l == m && gamma[l][m] != 1.) {
                    update += 1;
                  }
                  if (l != m && gamma[l][m] != 0.) {
                    update += 1;
                  }
                }
              }

              Real g[NDFULL][NDFULL];
              system.SpacetimeMetric(CellLocation::Face1, b, 0, 0, 0, g);
              for (int mu = 0; mu < NDFULL; ++mu) {
                for (int nu = 0; nu < NDFULL; ++nu) {
                  Real comp = g[mu][nu];
                  if (mu == nu && mu == 0 && comp != -1.) update += 1;
                  if (mu == nu && mu != 0 && comp != 1.) update += 1;
                  if (mu != nu && comp != 0.) update += 1;
                }
              }

              // metric determinants
              for (int k = 0; k < NX; ++k) {
                for (int j = 0; j < NX; ++j) {
                  for (int ii = 0; ii < NX; ++ii) {
                    Real dgamma = system.DetGamma(CellLocation::Face2, b, k, j, ii);
                    if (dgamma != 1.0) update += 1;
                    Real dg = system.DetG(CellLocation::Face3, b, k, j, ii);
                    if (dg != 1.0) update += 1;
                  }
                }
              }

              // grad ln(alpha)
              Real grada[NDFULL];
              system.GradLnAlpha(CellLocation::Corn, b, 0, 0, 0, grada);
              for (int mu = 0; mu < NDFULL; ++mu) {
                if (grada[mu] != 0.0) update += 1;
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

TEST_CASE("FLRW Coordinates", "[geometry]") {
  GIVEN("A coords object") {
    Coordinates_t coords;
    WHEN("We create an FLRW object") {
      Real t = 1.0;
      Real a0 = 1;
      Real dadt = 1;
      Real x, y, z;
      x = y = z = 0;
      IndexerMeshBlock indexer(coords);
      Analytic<FLRW, IndexerMeshBlock> system(t, indexer, a0, dadt);
      THEN("The coordinate system can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              // lapse
              Real alpha = system.Lapse(t, x, y, z);
              if (alpha != 1) update += 1;
              // shift
              Real beta[NDSPACE];
              system.ContravariantShift(t, x, y, z, beta);
              SPACELOOP(m) {
                if (beta[m] != 0) {
                  update += 1;
                }
              }
              //  metric
              Real gamma[NDSPACE][NDSPACE];
              system.Metric(t, x, y, z, gamma);
              SPACELOOP2(i, j) {
                if ((i != j) && gamma[i][j] != 0) update += 1;
                if ((i == j) && GetDifference(gamma[i][j], 4) > EPS) update += 1;
              }
              // Inverse metric
              system.MetricInverse(t, x, y, z, gamma);
              SPACELOOP2(i, j) {
                if ((i != j) && gamma[i][j] != 0) update += 1;
                if ((i == j) && GetDifference(gamma[i][j], 1. / 4.) > EPS) update += 1;
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

TEST_CASE("Spherical Minkowski Coordinates", "[geometry]") {
  GIVEN("A coords object") {
    Coordinates_t coords;
    WHEN("We create a Spherical Minkowski object") {
      IndexerMeshBlock indexer(coords);
      Analytic<SphericalMinkowski, IndexerMeshBlock> system(indexer);
      THEN("The coordinate system can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              Real r, th, ph;
              // lapse
              Real alpha = system.Lapse(0, 1, M_PI / 2., 0);
              if (alpha != 1) update += 1;
              // shift
              Real beta[NDSPACE];
              system.ContravariantShift(0, 3, 3 * M_PI / 4., M_PI / 2., beta);
              SPACELOOP(m) {
                if (beta[m] != 0) {
                  update += 1;
                }
              }
              //  metric
              Real gamma[NDSPACE][NDSPACE];
              r = 5;
              th = 3 * M_PI / 4;
              ph = 3 * M_PI / 4;
              system.Metric(0, r, th, ph, gamma);
              if (gamma[0][0] != 1) update += 1;
              if (GetDifference(gamma[1][1], r * r) > EPS) update += 1;
              if (GetDifference(gamma[2][2], std::pow(r * std::sin(th), 2)) > EPS)
                update += 1;
              if (gamma[0][1] != 0) update += 1;
              if (gamma[0][2] != 0) update += 1;
              if (gamma[1][2] != 0) update += 1;
              // Inverse metric
              system.MetricInverse(0, r, th, ph, gamma);
              if (GetDifference(gamma[1][1], 1 / (r * r)) > EPS) update += 1;
              if (GetDifference(gamma[2][2], std::pow(r * std::sin(th), -2)) > EPS)
                update += 1;
              // Connection Coefficient
              Real Gamma[NDFULL][NDFULL][NDFULL];
              system.ConnectionCoefficient(0, r, th, ph, Gamma);
              SPACETIMELOOP2(mu, nu) {
                if (Gamma[0][mu][nu] != 0) update += 1;
              }
              if (GetDifference(Gamma[1][2][2], -r) > EPS) update += 1;
              if (GetDifference(Gamma[1][3][3], -r * std::sin(th) * std::sin(th)) > EPS)
                update += 1;
              if (GetDifference(Gamma[2][1][2], r) > EPS) update += 1;
              if (GetDifference(Gamma[2][3][3], -r * r * std::cos(th) * std::sin(th)) >
                  EPS)
                update += 1;
              if (GetDifference(Gamma[3][2][3], r * r * std::cos(th) * std::sin(th)) >
                  EPS)
                update += 1;
              if (GetDifference(Gamma[3][1][3], r * std::sin(th) * std::sin(th)) > EPS)
                update += 1;
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

// Strategy for more complex metrics: Evaluate at equator as a
// consistency check between code implementation and mathematica

// BL is horrible, and not currently super important, so these tests
// won't be complete
TEST_CASE("Boyer-Lindquist Coordinates", "[geometry]") {
  GIVEN("A coords object") {
    Coordinates_t coords;
    WHEN("We create a Boyer-Lindquist object") {
      constexpr Real a = 0.9;
      IndexerMeshBlock indexer(coords);
      Analytic<BoyerLindquist, IndexerMeshBlock> system(indexer, a);
      THEN("The coordinate system can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              constexpr Real r = 10;
              constexpr Real th = M_PI / 2;
              constexpr Real ph = 0;
              // Lapse
              Real alpha = system.Lapse(0, r, th, ph);
              if (GetDifference(alpha, std::sqrt(r / (r - 2))) > EPS) update += 1;
              // Shift
              Real beta[NDSPACE];
              system.ContravariantShift(0, r, th, ph, beta);
              if (beta[0] != 0) update += 1;
              if (beta[1] != 0) update += 1;
              if (GetDifference(beta[2], -2 * a / (r - 2)) > EPS) update += 1;
              // metric
              Real gamma[NDSPACE][NDSPACE];
              system.Metric(0, r, th, ph, gamma);
              if (GetDifference(gamma[2][2], (r - 2) / (r * (a * a + (r - 2) * r))) > EPS)
                update += 1;
              // Inverse metric
              Real g[NDFULL][NDFULL];
              system.SpacetimeMetricInverse(0, r, th, ph, g);
              if (GetDifference(g[3][3], a * a * (1 + (2 / r)) + r * r) > EPS)
                update += 1;
              // Gradient
              Real dg[NDFULL][NDFULL][NDFULL];
              system.MetricDerivative(0, r, th, ph, dg);
              if (GetDifference(dg[2][2][1], -2 / (r * r * r)) > EPS) {
                update += 1;
              }
              // connection coefficent
              Real connection[NDFULL][NDFULL][NDFULL];
              system.ConnectionCoefficient(0, r, th, ph, connection);
              if (GetDifference(connection[1][1][1], (r - a * a) / (r * r * r)) > EPS) {
                update += 1;
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

TEST_CASE("Spherical Kerr-Schild Coordinates", "[geometry]") {
  GIVEN("A coords object") {
    Coordinates_t coords;
    WHEN("We create a Kerr-Schild object") {
      constexpr Real a = 0.9;
      IndexerMeshBlock indexer(coords);
      Analytic<SphericalKerrSchild, IndexerMeshBlock> system(indexer, a);
      THEN("The coordinate system can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              constexpr Real r = 6;
              constexpr Real th = M_PI / 2;
              constexpr Real ph = 3 * M_PI / 2;
              // Lapse
              Real alpha = system.Lapse(1, r, th, ph);
              if (GetDifference(alpha, std::sqrt(r / (r + 2))) > EPS) update += 1;
              // metric determinant
              Real gamdet = system.DetGamma(-3, r, th, ph);
              if (GetDifference(gamdet, std::sqrt(r * r * r * (2 + r))) > EPS)
                update += 1;
              Real gdet = system.DetG(-5, r, th, ph);
              if (GetDifference(gdet, r * r) > EPS) update += 1;
              // Shift
              Real beta[NDSPACE];
              system.ContravariantShift(2, r, th, ph, beta);
              if (GetDifference(beta[0], 2 / (2 + r)) > EPS) update += 1;
              if (beta[1] != 0 || beta[2] != 0) update += 1;
              // metric
              Real gamma[NDSPACE][NDSPACE];
              system.Metric(3, r, th, ph, gamma);
              if (gamma[0][1] != 0 || gamma[1][0] != 0 || gamma[1][2] != 0 ||
                  gamma[2][1] != 0)
                update += 1;
              if (GetDifference(gamma[0][0], (2 + r) / r) > EPS) {
                printf("%g %g\n", gamma[0][0], (2 + r) / r);
                update += 1;
              }
              if (GetDifference(gamma[1][1], r * r) > EPS) update += 1;
              if (GetDifference(gamma[2][2], r * r + ((a * a * (2 + r)) / r)) > EPS)
                update += 1;
              if (GetDifference(gamma[0][2], -a * (r + 2) / r) > EPS) update += 1;
              // Inverse metric
              system.MetricInverse(4, r, th, ph, gamma);
              if (gamma[0][1] != 0 || gamma[1][0] != 0 || gamma[1][2] != 0 ||
                  gamma[2][1] != 0)
                update += 1;
              if (GetDifference(gamma[0][0], ((a * a) / (r * r)) + (r / (2 + r))) > EPS)
                update += 1;
              if (GetDifference(gamma[1][1], (1 / (r * r))) > EPS) update += 1;
              if (GetDifference(gamma[2][2], (1 / (r * r))) > EPS) update += 1;
              if (GetDifference(gamma[2][0], a / (r * r)) > EPS) update += 1;
              if (GetDifference(gamma[0][2], a / (r * r)) > EPS) update += 1;
              // Metric Derivative
              Real dg[NDFULL][NDFULL][NDFULL];
              system.MetricDerivative(11, r, th, ph, dg);
              SPACETIMELOOP2(mu, nu) {
                if (dg[mu][nu][0] != 0) update += 1;
              }
              for (int mu = 0; mu < 2; ++mu) {
                for (int nu = 0; nu < 2; ++nu) {
                  if (GetDifference(dg[mu][nu][1], -2 / (r * r)) > EPS) update += 1;
                }
              }
              if (GetDifference(dg[2][2][1], 2 * r) > EPS) update += 1;
              if (GetDifference(dg[3][3][1], 2 * r - (2 * a * a / (r * r))) > EPS)
                update += 1;
              if (GetDifference(dg[0][3][1], 2 * a / (r * r)) > EPS) update += 1;
              if (GetDifference(dg[3][0][1], 2 * a / (r * r)) > EPS) update += 1;
              if (GetDifference(dg[1][3][1], 2 * a / (r * r)) > EPS) update += 1;
              if (GetDifference(dg[3][1][1], 2 * a / (r * r)) > EPS) update += 1;
              if (dg[0][2][1] != 0 || dg[2][0][1] != 0 || dg[1][2][1] != 0 ||
                  dg[2][1][1] != 0 || dg[2][3][1] != 0 || dg[3][2][1] != 0)
                update += 1;
              SPACETIMELOOP2(mu, nu) {
                if (std::abs(dg[mu][nu][2]) > EPS) update += 1;
                if (dg[mu][nu][3] != 0) update += 1;
              }
              // Christoffel connection
              Real Gamma[NDFULL][NDFULL][NDFULL];
              system.ConnectionCoefficient(10, r, th, ph, Gamma);
              if (std::abs(Gamma[0][0][0]) > EPS || std::abs(Gamma[0][0][2]) > EPS ||
                  std::abs(Gamma[0][0][3]) > EPS || std::abs(Gamma[0][1][2]) > EPS ||
                  std::abs(Gamma[0][2][0]) > EPS || std::abs(Gamma[0][2][1]) > EPS ||
                  std::abs(Gamma[0][2][2]) > EPS || std::abs(Gamma[0][2][3]) > EPS ||
                  std::abs(Gamma[0][3][0]) > EPS || std::abs(Gamma[0][3][2]) > EPS ||
                  std::abs(Gamma[0][3][3]) > EPS)
                update += 1;
              if (GetDifference(Gamma[0][0][1], -1 / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[0][1][0], -1 / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[0][1][1], -2 / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[0][3][1], a / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[0][1][3], a / (r * r)) > EPS) update += 1;
              if (std::abs(Gamma[1][0][1]) > EPS || std::abs(Gamma[1][1][0]) > EPS ||
                  std::abs(Gamma[1][0][2]) > EPS || std::abs(Gamma[1][2][0]) > EPS ||
                  std::abs(Gamma[1][1][2]) > EPS || std::abs(Gamma[1][2][1]) > EPS ||
                  std::abs(Gamma[1][1][3]) > EPS || std::abs(Gamma[1][3][1]) > EPS ||
                  std::abs(Gamma[1][2][3]) > EPS || std::abs(Gamma[1][3][2]) > EPS)
                update += 1;
              if (GetDifference(Gamma[1][0][0], 1 / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[1][1][1], -1 / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[1][2][2], -r) > EPS) update += 1;
              if (GetDifference(Gamma[1][3][3], ((a * a) / (r * r)) - r) > EPS)
                update += 1;
              if (GetDifference(Gamma[1][0][3], -a / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[1][3][0], -a / (r * r)) > EPS) update += 1;
              for (int mu = 0; mu < NDFULL; ++mu) {
                for (int nu = 0; nu < NDFULL; ++nu) {
                  if (!((mu == 1 && nu == 2) || (mu == 2 && nu == 1)) &&
                      std::abs(Gamma[2][mu][nu]) > EPS)
                    update += 1;
                }
              }
              if (GetDifference(Gamma[2][2][1], r) > EPS) update += 1;
              if (GetDifference(Gamma[2][1][2], r) > EPS) update += 1;
              if (std::abs(Gamma[3][0][0]) > EPS || std::abs(Gamma[3][0][2]) > EPS ||
                  std::abs(Gamma[3][2][0]) > EPS || std::abs(Gamma[3][0][3]) > EPS ||
                  std::abs(Gamma[3][3][0]) > EPS || std::abs(Gamma[3][1][2]) > EPS ||
                  std::abs(Gamma[3][2][1]) > EPS || std::abs(Gamma[3][2][2]) > EPS ||
                  std::abs(Gamma[3][2][3]) > EPS || std::abs(Gamma[3][3][2]) > EPS ||
                  std::abs(Gamma[3][3][3]) > EPS)
                update += 1;
              if (GetDifference(Gamma[3][0][1], a / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[3][1][0], a / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[3][1][1], 2 * a / (r * r)) > EPS) update += 1;
              if (GetDifference(Gamma[3][1][3], r - ((a * a) / (r * r))) > EPS)
                update += 1;
              if (GetDifference(Gamma[3][3][1], r - ((a * a) / (r * r))) > EPS)
                update += 1;
              // gradlnalpha
              Real da[NDFULL];
              system.GradLnAlpha(2021, r, th, ph, da);
              if (std::abs(da[0]) > EPS || std::abs(da[2]) > EPS || std::abs(da[3]) > EPS)
                update += 1;
              if (GetDifference(da[1], 1 / (2 * r + r * r)) > EPS) update += 1;
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

TEST_CASE("McKinneyGammieRyan", "[geometry]") {
  GIVEN("The parameters for a modifier") {
    constexpr bool derefine_poles = true;
    constexpr Real hslope = 0.3;
    constexpr Real poly_xt = 0.82;
    constexpr Real poly_alpha = 14.0;
    constexpr Real mks_smooth = 0.5;
    constexpr Real x0 = -0.4409148982097008;
    constexpr Real hexp_br = 1000.;
    constexpr Real hexp_nsq = 1.;
    constexpr Real hexp_csq = 4.;
    WHEN("We create a modifier object") {
      McKinneyGammieRyan transformation(derefine_poles, hslope, poly_xt, poly_alpha, x0,
                                        mks_smooth, hexp_br, hexp_nsq, hexp_csq);
      THEN("The modifier can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              // locations chosen from a nubhlight sim
              constexpr Real X[] = {0, 1.1324898310315077, 0.2638888888888889,
                                    3.2724923474893677};
              Real C[NDSPACE];
              Real Jcov[NDSPACE][NDSPACE];
              Real Jcon[NDSPACE][NDSPACE];
              transformation(X[1], X[2], X[3], C, Jcov, Jcon);
              // r
              if (GetDifference(C[0], 3.1033737650977384) > EPS) update += 1;
              // theta
              if (GetDifference(C[1], 1.1937404314936582) > EPS) update += 1;
              // phi
              if (GetDifference(C[2], 3.2724923474893677) > EPS) update += 1;
              // Jcov
              if (std::abs(Jcov[0][1] > EPS)) update += 1;
              if (std::abs(Jcov[0][2] > EPS)) update += 1;
              if (std::abs(Jcov[1][2] > EPS)) update += 1;
              if (std::abs(Jcov[2][0] > EPS)) update += 1;
              if (std::abs(Jcov[2][1] > EPS)) update += 1;
              if (GetDifference(Jcov[2][2], 1) > EPS) update += 1;
              if (GetDifference(Jcov[0][0], 3.10337377) > EPS) update += 1;
              if (GetDifference(Jcov[1][0], -0.00802045) > EPS) update += 1;
              if (GetDifference(Jcov[1][1], 2.29713534) > EPS) update += 1;
              // Jcon
              if (std::abs(Jcon[0][1] > EPS)) update += 1;
              if (std::abs(Jcon[0][2] > EPS)) update += 1;
              if (std::abs(Jcon[1][2] > EPS)) update += 1;
              if (std::abs(Jcon[2][0] > EPS)) update += 1;
              if (std::abs(Jcon[2][1] > EPS)) update += 1;
              if (GetDifference(Jcon[2][2], 1) > EPS) update += 1;
              if (GetDifference(Jcon[0][0], 0.32222996) > EPS) update += 1;
              if (GetDifference(Jcon[1][0], 0.00112507) > EPS) update += 1;
              if (GetDifference(Jcon[1][1], 0.43532481) > EPS) update += 1;

              // Now for a location where hyper-exponential coordinates are active
              constexpr Real Xfar[] = {0., 8.5, X[2], X[3]};
              transformation(Xfar[1], Xfar[2], Xfar[3], C, Jcov, Jcon);
              // r
              if (GetDifference(C[0], 4.914768840299134354e+03) > EPS) update += 1;
              // theta
              if (GetDifference(C[1], 1.178102621601838651e+00) > EPS) update += 1;
              // phi
              if (GetDifference(C[2], 3.2724923474893677) > EPS) update += 1;
              // Jcov
              if (std::abs(Jcov[0][1] > EPS)) update += 1;
              if (std::abs(Jcov[0][2] > EPS)) update += 1;
              if (std::abs(Jcov[1][2] > EPS)) update += 1;
              if (std::abs(Jcov[2][0] > EPS)) update += 1;
              if (std::abs(Jcov[2][1] > EPS)) update += 1;
              if (GetDifference(Jcov[2][2], 1) > EPS) update += 1;
              if (GetDifference(Jcov[0][0], 4.914768840299134354e+03) > EPS) update += 1;
              if (GetDifference(Jcov[1][0], -2.015412862105097876e-04) > EPS) update += 1;
              if (GetDifference(Jcov[1][1], 2.933523525877780980e+00) > EPS) update += 1;
              // Jcon
              if (std::abs(Jcon[0][1] > EPS)) update += 1;
              if (std::abs(Jcon[0][2] > EPS)) update += 1;
              if (std::abs(Jcon[1][2] > EPS)) update += 1;
              if (std::abs(Jcon[2][0] > EPS)) update += 1;
              if (std::abs(Jcon[2][1] > EPS)) update += 1;
              if (GetDifference(Jcon[2][2], 1) > EPS) update += 1;
              if (GetDifference(Jcon[0][0], 2.034683690106441685e-04) > EPS) update += 1;
              if (GetDifference(Jcon[1][0], 1.397884708672635997e-08) > EPS) update += 1;
              if (GetDifference(Jcon[1][1], 3.408869883532895662e-01) > EPS) update += 1;
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}

TEST_CASE("Modified Kerr-Schild", "[geometry]") {
  GIVEN("A coords object and a modifier object") {
    constexpr Real a = 0.8; // Taken from a nubhlight run
    constexpr bool derefine_poles = true;
    constexpr Real hslope = 0.3;
    constexpr Real poly_xt = 0.82;
    constexpr Real poly_alpha = 14.0;
    constexpr Real mks_smooth = 0.5;
    constexpr Real x0 = -0.4409148982097008;
    constexpr Real dxfd = 1e-10;
    Coordinates_t coords;
    IndexerMeshBlock indexer(coords);
    McKinneyGammieRyan transformation(derefine_poles, hslope, poly_xt, poly_alpha, x0,
                                      mks_smooth);
    WHEN("We create an FMKS object") {
      FMKSMeshBlock system(indexer, dxfd, transformation, a);
      THEN("The coordinate system can be called on device") {
        int n_wrong = 100;
        Kokkos::parallel_reduce(
            "get n wrong", NTRIALS,
            KOKKOS_LAMBDA(const int i, int &update) {
              // locations chosen from a nubhlight sim
              constexpr Real X[] = {0, 1.1324898310315077, 0.2638888888888889,
                                    3.2724923474893677};
              // lapse
              Real alpha = system.Lapse(X[0], X[1], X[2], X[3]);
              if (GetDifference(alpha, 0.7811769941965497) > EPS) update += 1;
              // metric determinant
              Real gdet = system.DetG(X[0], X[1], X[2], X[3]);
              if (GetDifference(gdet, 64.40965903079623) > EPS) update += 1;
              // Spacetime metric
              Real g[NDFULL][NDFULL];
              system.SpacetimeMetric(X[0], X[1], X[2], X[3], g);
              if (GetDifference(g[0][0], -0.3612937485396168) > EPS) update += 1;
              if (GetDifference(g[0][1], 1.9821442243860723) > EPS) update += 1;
              if (GetDifference(g[1][0], 1.9821442243860723) > EPS) update += 1;
              if (std::abs(g[0][2]) > EPS) update += 1;
              if (std::abs(g[2][0]) > EPS) update += 1;
              if (GetDifference(g[1][1], 15.78288823) > EPS) update += 1;
              if (GetDifference(g[1][2], -0.17903916) > EPS) update += 1;
              if (GetDifference(g[2][1], -0.17903916) > EPS) update += 1;
              if (GetDifference(g[1][3], -3.51690001) > EPS) update += 1;
              if (GetDifference(g[3][1], -3.51690001) > EPS) update += 1;
              if (GetDifference(g[2][2], 51.27859056) > EPS) update += 1;
              if (std::abs(g[2][3]) > EPS || std::abs(g[3][2]) > EPS) update += 1;
              if (GetDifference(g[3][3], 9.18405881) > EPS) update += 1;
              // Inverse metric
              Real ginv[NDFULL][NDFULL];
              system.SpacetimeMetricInverse(X[0], X[1], X[2], X[3], ginv);
              if (GetDifference(ginv[0][0], -1.63870625e+00) > EPS) update += 1;
              if (GetDifference(ginv[0][1], 2.05810289e-01) > EPS) update += 1;
              if (GetDifference(ginv[1][0], 2.05810289e-01) > EPS) update += 1;
              if (std::abs(ginv[0][3]) > EPS || std::abs(ginv[3][0]) > EPS) update += 1;
              if (GetDifference(ginv[1][1], 4.34252153e-02) > EPS) update += 1;
              if (GetDifference(ginv[1][2], 0.00015161910494636605) > EPS) {
                printf("Bad ginv[1][2] %g %g\n", ginv[1][2], 0.00015161910494636605);
                update += 1;
              }
              if (GetDifference(ginv[2][1], 0.00015161910494636605) > EPS) update += 1;
              if (GetDifference(ginv[1][3], 2.65272964e-02) > EPS) update += 1;
              if (GetDifference(ginv[3][1], 2.65272964e-02) > EPS) update += 1;
              if (GetDifference(ginv[2][2], 1.95018454e-02) > EPS) update += 1;
              if (GetDifference(ginv[3][3], 1.19042557e-01) > EPS) update += 1;
              // Do the metric and inverse metric match?
              Real delta[NDFULL][NDFULL];
              LinearAlgebra::SetZero(delta, NDFULL, NDFULL);
              SPACETIMELOOP3(mu, nu, sigma) {
                delta[mu][nu] += ginv[mu][sigma] * g[sigma][nu];
              }
              SPACETIMELOOP2(mu, nu) {
                if (mu == nu) {
                  if (GetDifference(delta[mu][nu], 1) > 1e-10) {
                    printf("bad delta %d %d %g\n", mu, nu, delta[mu][nu]);
                    update += 1;
                  }
                } else if (std::abs(delta[mu][nu]) > 1e-10) {
                  printf("bad delta %d %d %g\n", mu, nu, delta[mu][nu]);
                  update += 1;
                }
              }
              // Connection
              Real Gamma[NDFULL][NDFULL][NDFULL];
              Real Gamma_lo[NDFULL][NDFULL][NDFULL];
              system.ConnectionCoefficient(X[0], X[1], X[2], X[3], Gamma_lo);
              SPACETIMELOOP3(sigma, mu, nu) {
                Gamma[sigma][mu][nu] = 0;
                SPACETIMELOOP(rho) {
                  Gamma[sigma][mu][nu] += ginv[sigma][rho] * Gamma_lo[rho][mu][nu];
                }
              }
              constexpr Real Gamma_nubhlight[NDFULL][NDFULL][NDFULL] = {
                  {{6.45525676e-02, 5.14097064e-01, -3.30779101e-02, -4.46414555e-02},
                   {5.14097064e-01, 2.56904200e+00, -0.06613399, -3.55524840e-01},
                   {-3.30779101e-02, -0.06613399, -1.04594396e+01, 2.28750940e-02},
                   {-4.46414555e-02, -3.55524840e-01, 2.28750940e-02, -1.68257280e+00}},
                  {{1.36203547e-02, -5.87986418e-02, -2.03004280e-13, -9.41918320e-03},
                   {-5.87986418e-02, 5.04209748e-01, -4.40835235e-02, 2.61511601e-01},
                   {-2.03004280e-13, -4.40835235e-02, -2.20690336e+00, -1.18727250e-12},
                   {-9.41918320e-03, 2.61511601e-01, -1.18727250e-12, -3.55016683e-01}},
                  {{-5.97507320e-04, -2.20716631e-03, -7.08797170e-16, 8.24885520e-03},
                   {-2.20716631e-03, -9.62764242e-03, 1.13318725e+00, 6.46865963e-02},
                   {-7.08797170e-16, 1.13318725e+00, -3.33103600e+00, -4.14512175e-15},
                   {8.24885520e-03, 6.46865963e-02, -4.14512175e-15, -1.69890236e-01}},
                  {{8.32030847e-03, 2.59880304e-02, -4.78314158e-02, -5.75392575e-03},
                   {2.59880304e-02, 8.27752972e-02, -3.76137106e-01, 9.69923918e-01},
                   {-4.78314158e-02, -3.76137106e-01, -1.34813792e+00, 9.42750263e-01},
                   {-5.75392575e-03, 9.69923918e-01, 9.42750263e-01, -2.16870146e-01}}};
              SPACETIMELOOP3(sigma, mu, nu) {
                if (std::abs(Gamma_nubhlight[sigma][mu][nu]) > 0.1) {
                  if (GetDifference(Gamma[sigma][mu][nu],
                                    Gamma_nubhlight[sigma][mu][nu]) > 0.1) {
                    printf("%d %d %d: %g %g\n", sigma, mu, nu, Gamma[sigma][mu][nu],
                           Gamma_nubhlight[sigma][mu][nu]);
                    update += 1;
                  }
                } else {
                  if (std::abs(Gamma[sigma][mu][nu]) > 0.1) {
                    printf("%d %d %d: %g %g\n", sigma, mu, nu, Gamma[sigma][mu][nu],
                           Gamma_nubhlight[sigma][mu][nu]);
                    update += 1;
                  }
                }
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}
