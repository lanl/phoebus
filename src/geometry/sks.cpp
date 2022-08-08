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

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/sks.hpp"

namespace Geometry {

// Don't need to overwrite SetGeometry
template <>
void Initialize<SuperimposedKerrSchildMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real m1 = pin->GetOrAddReal("geometry", "m1", 0.5);
  Real m2 = pin->GetOrAddReal("geometry", "m2", 0.5);
  Real a1 = pin->GetOrAddReal("geometry", "a1", 0);
  Real a2 = pin->GetOrAddReal("geometry", "a2", 0);
  Real b  = pin->GetOrAddReal("geometry", "b", 20);
  Real dxfd = pin->GetOrAddReal("geometry", "finite_differences_dx", 1e-8);
  params.Add("m1", m1);
  params.Add("m2", m2);
  params.Add("a1", a1);
  params.Add("a2", a2);
  params.Add("b", b);
  params.Add("dxfd", dxfd);
}

template <>
void SetGeometry<SuperimposedKerrSchildMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
void SetGeometry<SuperimposedKerrSchildMesh>(MeshData<Real> *rc) {}

template <>
SuperimposedKerrSchildMeshBlock
GetCoordinateSystem<SuperimposedKerrSchildMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real m1 = pkg->Param<Real>("m1");
  Real m2 = pkg->Param<Real>("m2");
  Real a1 = pkg->Param<Real>("a1");
  Real a2 = pkg->Param<Real>("a2");
  Real b  = pkg->Param<Real>("b");
  Real dxfd = pkg->Param<Real>("dxfd");
  return SuperimposedKerrSchildMeshBlock(indexer, m1, m2, a1, a2, b, dxfd);
}
template <>
SuperimposedKerrSchildMesh GetCoordinateSystem<SuperimposedKerrSchildMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real m1 = pkg->Param<Real>("m1");
  Real m2 = pkg->Param<Real>("m2");
  Real a1 = pkg->Param<Real>("a1");
  Real a2 = pkg->Param<Real>("a2");
  Real b  = pkg->Param<Real>("b");
  Real dxfd = pkg->Param<Real>("dxfd");
  return SuperimposedKerrSchildMesh(indexer, m1, m2, a1, a2, b, dxfd);
}

template <>
void Initialize<CSuperimposedKerrSchildMeshBlock>(ParameterInput *pin,
                                          StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SuperimposedKerrSchildMeshBlock>(pin, geometry);
}
template <>
CSuperimposedKerrSchildMeshBlock
GetCoordinateSystem<CSuperimposedKerrSchildMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SuperimposedKerrSchildMeshBlock>(rc);
}
template <>
CSuperimposedKerrSchildMesh GetCoordinateSystem<CSuperimposedKerrSchildMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SuperimposedKerrSchildMesh>(rc);
}
template <>
void SetGeometry<CSuperimposedKerrSchildMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SuperimposedKerrSchildMeshBlock>(rc);
}
template <>
void SetGeometry<CSuperimposedKerrSchildMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<SuperimposedKerrSchildMesh>(rc);
}
} // namespace Geometry
