// stdlib includes
#include <cmath>

// external includes
#include "catch2/catch.hpp"
#include <Kokkos_Core.hpp>

// parthenon includes
#include <coordinates/coordinates.hpp>
#include <defs.hpp>
#include <parameter_input.hpp>

// phoebus includes
#include <geometry/coordinate_systems.hpp>
#include <phoebus_utils/cell_locations.hpp>

using namespace Geometry;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::RegionSize;

constexpr int NTRIALS = 100;
constexpr int ND = CoordinateSystem::NDSPACE;
constexpr int NDFULL = CoordinateSystem::NDFULL;
constexpr int NG = NGHOST;
constexpr int NX = 128 + 2 * NG;
// constexpr Real EPS = 1e-5;

KOKKOS_INLINE_FUNCTION
Real GetDifference(Real a, Real b) {
  return 2. * std::abs(b - a) / (std::abs(b) + std::abs(a) + SMALL);
}

TEST_CASE("Minkowski Coordinates", "[geometry]") {
  GIVEN("A Cartesian Minkowski coordinate system") {
    CoordinateSystem system = Analytic<Minkowski>();
    THEN("The coordinate system can be called on device") {
      int n_wrong = 100; // > 0
      Kokkos::parallel_reduce(
          "get n wrong", NTRIALS,
          KOKKOS_LAMBDA(const int i, int &update) {

            // lapse
            Real alpha = system.Lapse(0, 0, 0, 0);
            if (alpha != 1.) update += 1;

            // shift
            for (int l = 0; l < ND; ++l) {
              Real beta = system.ContravariantShift(l, CellLocation::Corn, 0, 0, 0);
              if (beta != 0.0) update += 1;
            }

            // metric
            for (int l = 0; l < ND; l++) {
              for (int m = 0; m < ND; m++) {
                Real comp = system.Metric(l, m, CellLocation::Face1, 0, 0, 0);
                if (l == m && comp != 1.)
                  update += 1;
                if (l != m && comp != 0.)
                  update += 1;
              }
            }

            // inverse metric 
            Real gamma[ND][ND];
            system.MetricInverse(0, 0, 0, 0, gamma);
            for (int l = 0; l < ND; l++) {
              for (int m = 0; m < ND; m++) {
                if (l == m && gamma[l][m] != 1.)
                  update += 1;
                if (l != m && gamma[l][m] != 0.)
                  update += 1;
              }
            }

            
            // metric determinants
            for (int k = 0; k < NX; ++k) {
              for (int j = 0; j < NX; ++j) {
                for (int ii = 0; ii < NX; ++ii) {
                  Real dgamma =
                      system.DetGamma(CellLocation::Face2, k, j, ii);
                  if (dgamma != 1.0)
                    update += 1;
                  Real dg = system.DetG(CellLocation::Face3, k, j, ii);
                  if (dg != 1.0) update += 1;
                }
              }
            }

            // connection coefficient
            for (int l = 0; l < NDFULL; ++l) {
              for (int m = 0; m < NDFULL; ++m) {
                for (int n = 0; n < NDFULL; ++n) {
                  Real conn = system.ConnectionCoefficient(l, m, n, 0, 0, 0, 0);
                  if (conn != 0.0)
                    update += 1;
                }
              }
            }

            // grad g
            for (int nu = 0; nu < NDFULL; ++nu) {
              for (int l = 0; l < ND; ++l) {
                for (int mu = 0; mu < NDFULL; ++mu) {
                  Real dg = system.MetricDerivative(mu,l,nu,0,0,0,0);
                  if (dg != 0.0) update += 1;
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
}
