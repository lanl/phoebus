//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#ifndef RADIATION_HPP_
#define RADIATION_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "compile_constants.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/constants.hpp"

namespace radiation {

extern parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt);

TaskStatus CalculateRadiationFourForce(MeshBlockData<Real> *rc, const double dt);

} // namespace radiation

#endif // RADIATION_HPP_
