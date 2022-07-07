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
    // Dummy grid position
    const CellLocation loc = CellLocation::Cent;
    const int k = 0;
    const int j = 0;
    // const int i = 0;
    {
      const Real v[3] = {0., 0., 0.};
      const Real Gamma_ref = 1.;
      const Real ucon_ref[4] = {1., 0., 0., 0.};
      THEN("The Lorentz factor and four-velocity are correctly calculated") {
        int n_wrong = 1;
        Kokkos::parallel_reduce(
            "get n wrong", 1,
            KOKKOS_LAMBDA(const int i, int &update) {
              Real Gamma = GetLorentzFactor(v, system, loc, k, j, i);
              Real gammacov[3][3] = {0};
              system.Metric(loc, k, j, i, gammacov);
              Real Gamma1 = GetLorentzFactor(v, gammacov);
              Real gcov[4][4] = {0};
              system.SpacetimeMetric(loc, k, j, i, gcov);
              Real Gamma2 = GetLorentzFactor(v, gcov);
              if (!SoftEquiv(Gamma, Gamma_ref)) {
                update += 1;
              }
              if (!SoftEquiv(Gamma1, Gamma_ref)) {
                update += 1;
              }
              if (!SoftEquiv(Gamma2, Gamma_ref)) {
                update += 1;
              }
              Real ucon[4];
              GetFourVelocity(v, system, loc, k, j, i, ucon);
              SPACETIMELOOP(mu) {
                if (!SoftEquiv(ucon[mu], ucon_ref[mu])) {
                  update += 1;
                }
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
    {
      Real v[3] = {0.7, 0.2, 0.1};
      const Real Gamma_ref = 1.474419561548971247e+00;
      v[0] *= Gamma_ref;
      v[1] *= Gamma_ref;
      v[2] *= Gamma_ref;
      const Real ucon_ref[4] = {1.474419561548971247e+00, 1.032093693084279895e+00,
                                2.948839123097942716e-01, 1.474419561548971358e-01};
      THEN("The Lorentz factor and four-velocity are correctly calculated") {
        int n_wrong = 1;
        Kokkos::parallel_reduce(
            "get n wrong", 1,
            KOKKOS_LAMBDA(const int i, int &update) {
              Real Gamma = GetLorentzFactor(v, system, loc, k, j, i);
              Real gammacov[3][3] = {0};
              system.Metric(loc, k, j, i, gammacov);
              Real Gamma1 = GetLorentzFactor(v, gammacov);
              Real gcov[4][4] = {0};
              system.SpacetimeMetric(loc, k, j, i, gcov);
              Real Gamma2 = GetLorentzFactor(v, gcov);
              if (!SoftEquiv(Gamma, Gamma_ref)) {
                update += 1;
              }
              if (!SoftEquiv(Gamma1, Gamma_ref)) {
                update += 1;
              }
              if (!SoftEquiv(Gamma2, Gamma_ref)) {
                update += 1;
              }
              Real ucon[4];
              GetFourVelocity(v, system, loc, k, j, i, ucon);
              SPACETIMELOOP(mu) {
                if (!SoftEquiv(ucon[mu], ucon_ref[mu])) {
                  update += 1;
                }
              }
            },
            n_wrong);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}
