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

#include "geometry/boyer_lindquist.hpp"

namespace Geometry {

// Don't need to overwrite SetGeometry
template <>
void Initialize<BoyerLindquistMeshBlock>(ParameterInput *pin,
                                         StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("coordinates", "a", 0);
  Real dxfd = pin->GetOrAddReal("coordinates", "finite_differences_dx", 1e-8);
  params.Add("a", a);
  params.Add("dxfd", dxfd);
}
template <>
BoyerLindquistMeshBlock
GetCoordinateSystem<BoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  return BoyerLindquistMeshBlock(indexer, a, dxfd);
}
template <>
BoyerLindquistMesh GetCoordinateSystem<BoyerLindquistMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  return BoyerLindquistMesh(indexer, a, dxfd);
}

template <>
void Initialize<CBoyerLindquistMeshBlock>(ParameterInput *pin,
                                          StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<BoyerLindquistMeshBlock>(pin, geometry);
}
template <>
CBoyerLindquistMeshBlock
GetCoordinateSystem<CBoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<BoyerLindquistMeshBlock>(rc);
}
template <>
CBoyerLindquistMesh
GetCoordinateSystem<CBoyerLindquistMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<BoyerLindquistMesh>(rc);
}
template <>
void SetGeometry<CBoyerLindquistMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<BoyerLindquistMeshBlock>(rc);
}

}  // namespace Geometry
