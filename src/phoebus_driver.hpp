//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef PHOEBUS_DRIVER_HPP_
#define PHOEBUS_DRIVER_HPP_

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::driver::prelude;

namespace phoebus {

// TODO(JMM): What kind of driver should this be?
class PhoebusDriver : public EvolutionDriver {
 public:
  PhoebusDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm,
                const bool is_restart);

  void PostInitializationCommunication();

  TaskCollection RungeKuttaStage(const int stage);
  TaskListStatus RadiationPreStep();
  TaskListStatus RadiationPostStep();
  TaskListStatus MonteCarloStep();

  TaskListStatus Step();

 private:
  std::unique_ptr<parthenon::LowStorageIntegrator> integrator;
  const bool is_restart_;
  Real dt_init, dt_init_fact;
};

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin);

} // namespace phoebus

#endif // PHOEBUS_DRIVER_HPP_
