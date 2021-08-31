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

#ifndef FLUID_HPP_
#define FLUID_HPP_

#include <memory>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc);
//template <typename T>
TaskStatus PrimitiveToConservedRegion(MeshBlockData<Real> *rc, const IndexRange &ib, const IndexRange &jb, const IndexRange &kb);
template <typename T>
TaskStatus ConservedToPrimitive(T *rc);
template <typename T>
TaskStatus ConservedToPrimitiveClassic(T *rc, const IndexRange &ib, const IndexRange &jb, const IndexRange &kb);
template <typename T>
TaskStatus ConservedToPrimitiveRobust(T *rc, const IndexRange &ib, const IndexRange &jb, const IndexRange &kb);
TaskStatus CalculateFluidSourceTerms(MeshBlockData<Real> *rc, MeshBlockData<Real> *rc_src);
TaskStatus CopyFluxDivergence(MeshBlockData<Real> *rc);
TaskStatus ZeroUpdate(MeshBlockData<Real> *rc);
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc);
TaskStatus FluxCT(MeshBlockData<Real> *rc);
TaskStatus CalculateDivB(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);


template <typename T>
using c2p_type = TaskStatus(*)(T *, const IndexRange &, const IndexRange &, const IndexRange &);
using c2p_meshblock_type = c2p_type<MeshBlockData<Real>>;
using c2p_mesh_type = c2p_type<MeshData<Real>>;

} // namespace fluid

#endif // FLUID_HPP_
