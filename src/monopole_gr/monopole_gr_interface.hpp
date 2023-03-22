// © 2021. Triad National Security, LLC. All rights reserved.  This
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

#ifndef MONOPOLE_GR_MONOPOLE_GR_INTERFACE_HPP_
#define MONOPOLE_GR_MONOPOLE_GR_INTERFACE_HPP_

// stdlib
#include <memory>
#include <typeinfo>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

namespace MonopoleGR {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
Real EstimateTimeStep(StateDescriptor *pkg);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);
Real EstimateTimestepMesh(MeshBlockData<Real> *rc);
TaskStatus MatterToHost(StateDescriptor *pkg, bool do_vols);
TaskStatus IntegrateHypersurface(StateDescriptor *pkg);
TaskStatus LinearSolveForAlpha(StateDescriptor *pkg);
TaskStatus SpacetimeToDevice(StateDescriptor *pkg);
TaskStatus CheckRateOfChange(StateDescriptor *pkg, Real dt);
void DumpToTxt(const std::string &filename, StateDescriptor *pkg);
inline void DumpCurrentState(StateDescriptor *pkg) { DumpToTxt("metric-last.dat", pkg); }
TaskStatus DivideVols(StateDescriptor *pkg);
template<typename T>
TaskStatus InterpMetricToGrid(T *rc);
} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_INTERFACE_HPP_
