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

#ifndef PHOEBUS_BOUNDARIES_PHOEBUS_BOUNDARIES_HPP_
#define PHOEBUS_BOUNDARIES_PHOEBUS_BOUNDARIES_HPP_

#include <memory>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

namespace Boundaries {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

std::unique_ptr<parthenon::ParticleBound,
                parthenon::DeviceDeleter<parthenon::DevMemSpace>>
SetSwarmIx1Outflow();
std::unique_ptr<parthenon::ParticleBound,
                parthenon::DeviceDeleter<parthenon::DevMemSpace>>
SetSwarmOx1Outflow();

TaskStatus ConvertBoundaryConditions(std::shared_ptr<MeshBlockData<Real>> &rc);

} // namespace Boundaries

#endif // PHOEBUS_BOUNDARIES_PHOEBUS_BOUNDARIES_HPP_
