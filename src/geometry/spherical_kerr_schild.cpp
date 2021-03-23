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

#include "geometry/spherical_kerr_schild.hpp"

namespace Geometry {

template <>
void Initialize<SphericalKSMeshBlock>(ParameterInput *pin,
                                      StateDescriptor *geometry) {
  std::cout << "Initializing SphericalKSMeshBlock" << std::endl;
  Params &params = geometry->AllParams();
  Real a = pin->GetOrAddReal("coordinates", "a", 0);
  params.Add("a", a);
}
template <>
void SetGeometry<SphericalKSMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
SphericalKSMeshBlock
GetCoordinateSystem<SphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  return SphericalKSMeshBlock(indexer, a);
}
template <>
SphericalKSMesh GetCoordinateSystem<SphericalKSMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  return SphericalKSMesh(indexer, a);
}

template <>
void Initialize<CSphericalKSMeshBlock>(ParameterInput *pin,
                                       StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SphericalKSMeshBlock>(pin, geometry);
}
template <>
CSphericalKSMeshBlock
GetCoordinateSystem<CSphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SphericalKSMeshBlock>(rc);
}
template <>
CSphericalKSMesh
GetCoordinateSystem<CSphericalKSMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SphericalKSMesh>(rc);
}
template <>
void SetGeometry<CSphericalKSMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SphericalKSMeshBlock>(rc);
}

} // namespace Geometry
