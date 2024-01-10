// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

#ifndef TRACERS_HPP_
#define TRACERS_HPP_

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "microphysics/eos_phoebus/eos_phoebus.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;
using Microphysics::EOS::EOS;

#include "phoebus_utils/variables.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus AdvectTracers(MeshBlockData<Real> *rc, const Real dt);

void FillTracers(MeshBlockData<Real> *rc);

} // namespace tracers

#endif // TRACERS_HPP_