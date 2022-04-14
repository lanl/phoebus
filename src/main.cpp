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
#include <globals.hpp>
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

  // TODO(JMM): Move this into another function somewhere?
  // Ensure only allowed parthenon boundary conditions are used
  const std::string parth_ix1_bc = pman.pinput->GetString("parthenon/mesh", "ix1_bc");
  PARTHENON_REQUIRE(parth_ix1_bc == "user" || parth_ix1_bc == "periodic",
                    "Only \"user\" and \"periodic\" allowed for parthenon/mesh/ix1_bc");
  const std::string parth_ox1_bc = pman.pinput->GetString("parthenon/mesh", "ox1_bc");
  PARTHENON_REQUIRE(parth_ox1_bc == "user" || parth_ox1_bc == "periodic",
                    "Only \"user\" and \"periodic\" allowed for parthenon/mesh/ox1_bc");

  const std::string ix1_bc = pman.pinput->GetOrAddString("phoebus", "ix1_bc", "outflow");
  const std::string ox1_bc = pman.pinput->GetOrAddString("phoebus", "ox1_bc", "outflow");

  const std::string rad_method =
      pman.pinput->GetOrAddString("radiation", "method", "None");

  if (ix1_bc == "reflect") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        Boundaries::ReflectInnerX1;
  } else if (ix1_bc == "outflow") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        Boundaries::OutflowInnerX1;
    if (rad_method == "mocmc") {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
          radiation::SetSwarmNoWorkBC;
    } else {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
          Boundaries::SetSwarmIX1Outflow;
    }
  } else if (ix1_bc == "fixed_temp") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        Boundaries::OutflowInnerX1;
    if (rad_method == "mocmc") {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
          radiation::SetSwarmNoWorkBC;
    } else {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
          Boundaries::SetSwarmIX1Outflow;
    }
  } // else, parthenon periodic boundaries
  if (ox1_bc == "reflect") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::ReflectOuterX1;
  } else if (ox1_bc == "outflow") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::OutflowOuterX1;
    if (rad_method == "mocmc") {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
          radiation::SetSwarmNoWorkBC;
    } else {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
          Boundaries::SetSwarmOX1Outflow;
    }
  } else if (ox1_bc == "fixed_temp") {
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        Boundaries::OutflowOuterX1;
    if (rad_method == "mocmc") {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
          radiation::SetSwarmNoWorkBC;
    } else {
      pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
          Boundaries::SetSwarmOX1Outflow;
    }
  } // else, parthenon periodic boundaries

  phoebus::ProblemModifier(pman.pinput.get());

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
