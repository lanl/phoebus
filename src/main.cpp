//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <string>
#include <vector>

#include <defs.hpp>
#include <globals.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>

#include "fluid/con2prim_statistics.hpp"
#include "geometry/geometry.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"
#include "phoebus_driver.hpp"
#include "radiation/radiation.hpp"

int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;

  // Set up kokkos and read pin
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = phoebus::ProcessPackages;
  // pman.app_input->ProcessProperties = phoebus::ProcessProperties;
  pman.app_input->ProblemGenerator = phoebus::ProblemGenerator;
  pman.app_input->InitMeshBlockUserData = Geometry::SetGeometryBlock;
  // pman.app_input->UserWorkAfterLoop = phoebus::UserWorkAfterLoop;
  // pman.app_input->SetFillDerivedFunctions = phoebus::SetFillDerivedFunctions;

  phoebus::ProblemModifier(pman.pinput.get());

  Boundaries::ProcessBoundaryConditions(pman);

  // call ParthenonInit to set up the mesh
  pman.ParthenonInitPackagesAndMesh();

  // call post-initialization
  phoebus::PostInitializationModifier(pman.pinput.get(), pman.pmesh.get());

  // Initialize the driver
  phoebus::PhoebusDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

  // Communicate ghost buffers before executing
  driver.PostInitializationCommunication();

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

#if CON2PRIM_STATISTICS
  con2prim_statistics::Stats::report();
#endif

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
