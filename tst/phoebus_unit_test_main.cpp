#include <Kokkos_Core.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main(int argc, char* argv[]) {
  /* This currently only calls to Kokkos::initialize()/finalize().
   * If MPI unit tests are desired in the future, it probably
   * makes more sense to call parthenon::ParthenonManager::ParthenonInit/
   * ParthenonFinalize but some work to break the parallel initialization apart
   * from the rest will be required.
   */
  Kokkos::initialize();
  int result = 0;
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
