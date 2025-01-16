// © 2023 Triad National Security, LLC. All rights reserved.  This
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

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif // MPI_PARALLEL

#include <Kokkos_Core.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main(int argc, char *argv[]) {
  /* This currently calls Kokkos and MPI manually. It might
   * make more sense to call parthenon::ParthenonManager::ParthenonInit/
   * ParthenonFinalize but some work to break the parallel initialization apart
   * from the rest will be required.
   */
#ifdef MPI_PARALLEL
  MPI_Init(&argc, &argv);
#endif // MPI_PARALLEL
  Kokkos::initialize();
  int result = 0;
  { result = Catch::Session().run(argc, argv); }
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif // MPI_PARALLEL
  return result;
}
