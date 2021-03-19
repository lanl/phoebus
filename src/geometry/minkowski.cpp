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

#include "geometry/minkowski.hpp"

namespace Geometry {

// We can utilize the default Initialize and SetGeometry because
// Minkowski doesn't need anything
template <>
MinkowskiMeshBlock
GetCoordinateSystem<MinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return MinkowskiMeshBlock(indexer);
}
template <>
MinkowskiMesh GetCoordinateSystem<MinkowskiMesh>(MeshData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return MinkowskiMesh(indexer);
}

template <>
void Initialize<CMinkowskiMeshBlock>(ParameterInput *pin,
                                     StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<MinkowskiMeshBlock>(pin, geometry);
}
template <>
CMinkowskiMeshBlock
GetCoordinateSystem<CMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<MinkowskiMeshBlock>(rc);
}
template <>
CMinkowskiMesh GetCoordinateSystem<CMinkowskiMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<MinkowskiMesh>(rc);
}
template <> void SetGeometry<CMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<MinkowskiMeshBlock>(rc);
}

} // namespace Geometry
