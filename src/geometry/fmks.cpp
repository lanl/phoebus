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
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/fmks.hpp"

namespace Geometry {

template <>
void Initialize<FMKSMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Initialize<SphericalKSMeshBlock>(pin, geometry);
  Params &params = geometry->AllParams();
  Real dxfd_geom = pin->GetOrAddReal("geometry", "finite_differences_dx", 1e-8);
  Real dxfd = pin->GetOrAddReal("coordinates", "finite_differences_dx", dxfd_geom);
  bool derefine_poles =
      pin->GetOrAddBoolean("coordinates", "derefine_poles", true);
  Real h = pin->GetOrAddReal("coordinates", "hslope", 0.3);
  Real xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
  Real alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14);
  Real x0 = pin->GetReal("parthenon/mesh", "x1min");
  Real smooth = pin->GetOrAddReal("coordinates", "smooth", 0.5);
  if (!params.hasKey("dxfd")) {
    params.Add("dxfd", dxfd);
  }
  params.Add("derefine_poles", derefine_poles);
  params.Add("h", h);
  params.Add("xt", xt);
  params.Add("alpha", alpha);
  params.Add("x0", x0);
  params.Add("smooth", smooth);
}
template <> void SetGeometry<FMKSMeshBlock>(MeshBlockData<Real> *rc) {}

template <>
FMKSMeshBlock GetCoordinateSystem<FMKSMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<McKinneyGammieRyan>(pkg.get());
  return FMKSMeshBlock(indexer, dxfd, transformation, a);
}
template <> FMKSMesh GetCoordinateSystem<FMKSMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("geometry");
  auto indexer = GetIndexer(rc);
  Real a = pkg->Param<Real>("a");
  Real dxfd = pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<McKinneyGammieRyan>(pkg.get());
  return FMKSMesh(indexer, dxfd, transformation, a);
}

template <>
void Initialize<CFMKSMeshBlock>(ParameterInput *pin,
                                StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<FMKSMeshBlock>(pin, geometry);
}
template <>
CFMKSMeshBlock GetCoordinateSystem<CFMKSMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<FMKSMeshBlock>(rc);
}
template <> CFMKSMesh GetCoordinateSystem<CFMKSMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<FMKSMesh>(rc);
}
template <> void SetGeometry<CFMKSMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<FMKSMeshBlock>(rc);
}
template <>
bool CoordinatesNeedSetting<FMKSMeshBlock>(MeshBlockData<Real> *rc) {
  return false;
}
template <>
bool CoordinatesNeedSetting<CFMKSMeshBlock>(MeshBlockData<Real> *rc) {
  return CachedCoordinatesNeedSetting<FMKSMeshBlock>(rc);
}

} // namespace Geometry
