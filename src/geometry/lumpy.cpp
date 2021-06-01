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

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/lumpy.hpp"

namespace Geometry {

template <>
void Initialize<LumpyMeshBlock>(ParameterInput *pin,
                                      StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("geometry", "a", 0.3);
  Real k = pin->GetOrAddReal("geometry", "k", M_PI/2.);
  Real betax = pin->GetOrAddReal("geometry", "betax", 0.);
  params.Add("a", a);
  params.Add("k", k);
  params.Add("betax", betax);
}

template <> void SetGeometry<LumpyMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
LumpyMeshBlock
GetCoordinateSystem<LumpyMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  Real betax = pkg->Param<Real>("betax");
  return LumpyMeshBlock(indexer, a, k, betax);
}
template <>
LumpyMesh GetCoordinateSystem<LumpyMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real k = pkg->Param<Real>("k");
  Real betax = pkg->Param<Real>("betax");
  return LumpyMesh(indexer, a, k, betax);
}

template <>
void Initialize<CLumpyMeshBlock>(ParameterInput *pin,
                                     StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<LumpyMeshBlock>(pin, geometry);
}
template <>
CLumpyMeshBlock
GetCoordinateSystem<CLumpyMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<LumpyMeshBlock>(rc);
}
template <>
CLumpyMesh GetCoordinateSystem<CLumpyMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<LumpyMesh>(rc);
}
template <> void SetGeometry<CLumpyMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<LumpyMeshBlock>(rc);
}

} // namespace Geometry
