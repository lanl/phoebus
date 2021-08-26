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

#include <defs.hpp>
#include <parthenon_manager.hpp>

#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"
#include "phoebus_driver.hpp"

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

  // TODO(JMM): Move this into another function somewhere?
  const std::string bc_ix1 =
      pman.pinput->GetOrAddString("phoebus", "bc_ix1", "reflect");
  const std::string bc_ox1 =
      pman.pinput->GetOrAddString("phoebus", "bc_ox1", "outflow");

  if (bc_ix1 == "reflect") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        Boundaries::ReflectInnerX1;
  } else {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        Boundaries::OutflowInnerX1;
  }
  if (bc_ox1 == "reflect") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::ReflectOuterX1;
  } else {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::OutflowOuterX1;
  }

  std::string problem_name = pman.pinput->GetString("phoebus", "problem");
  /*if (problem_name == "bondi") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::BondiOuterX1;
  }*/

  phoebus::ProblemModifier(pman.pinput.get());

  // call ParthenonInit to set up the mesh
  pman.ParthenonInitPackagesAndMesh();

  // Initialize the driver
  phoebus::PhoebusDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
