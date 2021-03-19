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

#include "geometry/spherical_minkowski.hpp"

namespace Geometry {

template <>
void Initialize<SphMinkowskiMeshBlock>(ParameterInput *pin,
                                       StateDescriptor *geometry) {}

template <>
SphMinkowskiMeshBlock
GetCoordinateSystem<SphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return SphMinkowskiMeshBlock(indexer);
}
template <> void SetGeometry<SphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
SphMinkowskiMesh GetCoordinateSystem<SphMinkowskiMesh>(MeshData<Real> *rc) {
  auto indexer = GetIndexer(rc);
  return SphMinkowskiMesh(indexer);
}

template <>
void Initialize<CSphMinkowskiMeshBlock>(ParameterInput *pin,
                                        StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<SphMinkowskiMeshBlock>(pin, geometry);
}
template <>
CSphMinkowskiMeshBlock
GetCoordinateSystem<CSphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<SphMinkowskiMeshBlock>(rc);
}
template <>
CSphMinkowskiMesh GetCoordinateSystem<CSphMinkowskiMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<SphMinkowskiMesh>(rc);
}
template <> void SetGeometry<CSphMinkowskiMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<SphMinkowskiMeshBlock>(rc);
}

} // namespace Geometry
