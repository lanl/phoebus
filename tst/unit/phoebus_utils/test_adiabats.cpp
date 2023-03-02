// © 2023. Triad National Security, LLC. All rights reserved.  This
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
#include <limits>

// external includes
#include "catch2/catch.hpp"
#include <Kokkos_Core.hpp>
#include <singularity-eos/eos/eos.hpp>

// parthenon includes
#include <coordinates/coordinates.hpp>
#include <defs.hpp>
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>

#include <spiner/databox.hpp>

// Test utils
#include "test_utils.hpp"

// Relativity utils
#include "phoebus_utils/adiabats.hpp"

// using namespace phoebus;
using singularity::StellarCollapse;

TEST_CASE("ADIABATS", "[compute_adiabats]") {
  GIVEN("A stellar collapse table") {
    std::string filename = "./unit/phoebus_utils/table_dir/"
                           "Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5";
    StellarCollapse eos(filename);

    const int nsamps = 180;
    Spiner::DataBox rho_d(Spiner::AllocationTarget::Device, nsamps);
    Spiner::DataBox temp_d(Spiner::AllocationTarget::Device, nsamps);
    const Real lrho_min = eos.lRhoMin();
    const Real lrho_max = eos.lRhoMax();
    const Real rho_min = std::pow(10.0, lrho_min);
    const Real rho_max = std::pow(10.0, lrho_max);
    const Real T_min = std::pow(10.0, eos.lTMin());
    const Real T_max = std::pow(10.0, eos.lTMax());

    const Real Ye = 0.1;
    const Real S0 = 20.0;

    THEN("We can compute adiabats") {
      /**
       * Sample rho. For this adiabat, we do not go to the table upper limit
       * May need generalization in the future [TODO(BLB)]
       **/
      Real lrho_min_adiabat, lrho_max_adiabat; // rho bounds for adiabat
      GetRhoBounds(eos, rho_min, rho_max, T_min, T_max, Ye, S0, 
                   lrho_min_adiabat, lrho_max_adiabat);
      SampleRho(rho_d, lrho_min_adiabat, lrho_max_adiabat, nsamps);

      ComputeAdiabats(rho_d, temp_d, eos, Ye, S0, T_min, T_max, nsamps);

      AND_THEN("We verify that the adiabats produce the original entropy") {
        // Given the adiabats, we should be able to get back the original entropy S0
        int n_wrong = 1;
        Kokkos::parallel_reduce(
            "get n_wrong (adiabats)", nsamps,
            KOKKOS_LAMBDA(const int i, int &update) {
              Real lambda[2];
              lambda[0] = Ye;
              const Real rho = std::pow(10.0, rho_d(i));
              const Real S_i = eos.EntropyFromDensityTemperature(rho, temp_d(i), lambda);
              if (!SoftEquiv(S_i, S0, 1.e-6, false)) {
                update += 1;
              }
            },
            n_wrong);

        free(rho_d);
        free(temp_d);
        REQUIRE(n_wrong == 0);
      }
    }
  }
}
