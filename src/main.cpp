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

void ProcessBoundaryConditions(parthenon::ParthenonManager &pman);

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

  ProcessBoundaryConditions(pman);

  // call ParthenonInit to set up the mesh
  pman.ParthenonInitPackagesAndMesh();

  // call post-initialization
  phoebus::PostInitializationModifier(pman.pinput.get(), pman.pmesh.get());

  // Initialize the driver
  phoebus::PhoebusDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

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

// TODO(JMM): Move this somewhere else?
void ProcessBoundaryConditions(parthenon::ParthenonManager &pman) {
  // Ensure only allowed parthenon boundary conditions are used
  const std::vector<std::string> inner_outer = {"i", "o"};
  static const parthenon::BoundaryFace loc[][2] = {
      {parthenon::BoundaryFace::inner_x1, parthenon::BoundaryFace::outer_x1},
      {parthenon::BoundaryFace::inner_x2, parthenon::BoundaryFace::outer_x2},
      {parthenon::BoundaryFace::inner_x3, parthenon::BoundaryFace::outer_x3}};
  static const parthenon::BValFunc outflow[][2] = {
      {Boundaries::OutflowInnerX1, Boundaries::OutflowOuterX1},
      {Boundaries::OutflowInnerX2, Boundaries::OutflowOuterX2},
      {Boundaries::OutflowInnerX3, Boundaries::OutflowOuterX3}};
  static const parthenon::BValFunc reflect[][2] = {
      {Boundaries::ReflectInnerX1, Boundaries::ReflectOuterX1},
      {Boundaries::ReflectInnerX2, Boundaries::ReflectOuterX2},
      {Boundaries::ReflectInnerX3, Boundaries::ReflectOuterX3}};

  for (int d = 1; d <= 3; ++d) {
    // outer = 0 for inner face, outer = 1 for outer face
    for (int outer = 0; outer <= 1; ++outer) {
      auto &face = inner_outer[outer];
      const std::string name = face + "x" + std::to_string(d) + "_bc";
      const std::string parth_bc = pman.pinput->GetString("parthenon/mesh", name);
      PARTHENON_REQUIRE(parth_bc == "user" || parth_bc == "periodic",
                        "Only \"user\" and \"periodic\" allowed for parthenon/mesh/" +
                            name);

      const std::string bc = pman.pinput->GetOrAddString("phoebus", name, "outflow");
      if (bc == "reflect") {
        pman.app_input->boundary_conditions[loc[d - 1][outer]] = reflect[d - 1][outer];
      } else if (bc == "outflow") {
        pman.app_input->boundary_conditions[loc[d - 1][outer]] = outflow[d - 1][outer];
      } // periodic boundaries, which are handled by parthenon, so no need to set anything
    }
  }
}
