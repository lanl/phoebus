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

#include <Kokkos_Core.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main(int argc, char *argv[]) {
  /* This currently only calls to Kokkos::initialize()/finalize().
   * If MPI unit tests are desired in the future, it probably
   * makes more sense to call parthenon::ParthenonManager::ParthenonInit/
   * ParthenonFinalize but some work to break the parallel initialization apart
   * from the rest will be required.
   */
  Kokkos::initialize();
  int result = 0;
  { result = Catch::Session().run(argc, argv); }
  Kokkos::finalize();
  return result;
}
