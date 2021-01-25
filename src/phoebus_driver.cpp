//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "geometry/geometry.hpp"
#include "microphysics/eos/eos.hpp"
#include "phoebus_driver.hpp"

using namespace parthenon::driver::prelude;

namespace phoebus {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
PhoebusDriver::PhoebusDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : Driver(pin, app_in, pm) {
  InitializeOutputs();

  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");
}

// TODO(JMM): In the end we should probably use a different driver,
// like evolution driver, and then this is handled for us.
parthenon::DriverStatus PhoebusDriver::Execute() {
  pouts->MakeOutputs(pmesh, pinput);

  ConstructAndExecuteTaskLists<>(this);

  return DriverStatus::complete;
}

template<typename T>
TaskCollection PhoebusDriver::MakeTaskCollection(T &blocks) {
  TaskCollection tc;

  return tc;
}

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;

  packages.Add(Microphysics::EOS::Initialize(pin.get()));
  packages.Add(Geometry::Initialize(pin.get()));

  return packages;
}

} // namespace phoebus
