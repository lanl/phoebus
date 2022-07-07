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
#include "monopole_gr/monopole_gr_base.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/monopole.hpp"

namespace Geometry {

template <>
void Initialize<MplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  const bool time_dependent = true;
  params.Add("time_dependent", time_dependent);
  return;
}
template <>
void Initialize<MplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  Params &params = geometry->AllParams();
  const bool time_dependent = true;
  const bool do_fd_on_grid = true;
  params.Add("time_dependent", time_dependent);
  params.Add("do_fd_on_grid", do_fd_on_grid);
  Real dxfd_geom = pin->GetOrAddReal("geometry", "finite_differences_dx", 1e-8);
  Real dxfd = pin->GetOrAddReal("coordinates", "finite_differences_dx", dxfd_geom);
  if (!params.hasKey("dxfd")) {
    params.Add("dxfd", dxfd);
  }
  return;
}

// The geometry on a block is set by the monopole solver or by the
// Cached machinery
template <>
void SetGeometry<MplSphMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<MplCartMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<MplSphMesh>(MeshData<Real> *rc) {}
template <>
void SetGeometry<MplCartMesh>(MeshData<Real> *rc) {}

template <>
MplSphMeshBlock GetCoordinateSystem<MplSphMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("monopole_gr");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  PARTHENON_REQUIRE_THROWS(enabled, "MonopoleGR must be enabled for this metric");

  auto hypersurface = params.Get<MonopoleGR::Hypersurface_t>("hypersurface");
  auto alpha = params.Get<MonopoleGR::Alpha_t>("lapse");
  auto beta = params.Get<MonopoleGR::Beta_t>("shift");
  auto gradients = params.Get<MonopoleGR::Gradients_t>("gradients");
  auto rgrid = params.Get<MonopoleGR::Radius>("radius");
  auto indexer = GetIndexer(rc);
  return MplSphMeshBlock(indexer, hypersurface, alpha, beta, gradients, rgrid);
}
template <>
MplSphMesh GetCoordinateSystem<MplSphMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("monopole_gr");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  PARTHENON_REQUIRE_THROWS(enabled, "MonopoleGR must be enabled for this metric");

  auto hypersurface = params.Get<MonopoleGR::Hypersurface_t>("hypersurface");
  auto alpha = params.Get<MonopoleGR::Alpha_t>("lapse");
  auto beta = params.Get<MonopoleGR::Beta_t>("shift");
  auto gradients = params.Get<MonopoleGR::Gradients_t>("gradients");
  auto rgrid = params.Get<MonopoleGR::Radius>("radius");
  auto indexer = GetIndexer(rc);
  return MplSphMesh(indexer, hypersurface, alpha, beta, gradients, rgrid);
}

template <>
MplCartMeshBlock GetCoordinateSystem<MplCartMeshBlock>(MeshBlockData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("monopole_gr");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  PARTHENON_REQUIRE_THROWS(enabled, "MonopoleGR must be enabled for this metric");

  auto hypersurface = params.Get<MonopoleGR::Hypersurface_t>("hypersurface");
  auto alpha = params.Get<MonopoleGR::Alpha_t>("lapse");
  auto beta = params.Get<MonopoleGR::Beta_t>("shift");
  auto gradients = params.Get<MonopoleGR::Gradients_t>("gradients");
  auto rgrid = params.Get<MonopoleGR::Radius>("radius");
  auto indexer = GetIndexer(rc);

  auto &geom_pkg = rc->GetParentPointer()->packages.Get("geometry");
  Real dxfd = geom_pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<SphericalToCartesian>(pkg.get());

  return MplCartMeshBlock(indexer, dxfd, transformation, hypersurface, alpha, beta,
                          gradients, rgrid);
}
template <>
MplCartMesh GetCoordinateSystem<MplCartMesh>(MeshData<Real> *rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("monopole_gr");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  PARTHENON_REQUIRE_THROWS(enabled, "MonopoleGR must be enabled for this metric");

  auto hypersurface = params.Get<MonopoleGR::Hypersurface_t>("hypersurface");
  auto alpha = params.Get<MonopoleGR::Alpha_t>("lapse");
  auto beta = params.Get<MonopoleGR::Beta_t>("shift");
  auto gradients = params.Get<MonopoleGR::Gradients_t>("gradients");
  auto rgrid = params.Get<MonopoleGR::Radius>("radius");
  auto indexer = GetIndexer(rc);

  auto &geom_pkg = rc->GetParentPointer()->packages.Get("geometry");
  Real dxfd = geom_pkg->Param<Real>("dxfd");
  auto transformation = GetTransformation<SphericalToCartesian>(pkg.get());

  return MplCartMesh(indexer, dxfd, transformation, hypersurface, alpha, beta, gradients,
                     rgrid);
}

template <>
void Initialize<CMplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<MplSphMeshBlock>(pin, geometry);
}
template <>
void Initialize<CMplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  InitializeCachedCoordinateSystem<MplCartMeshBlock>(pin, geometry);
}

template <>
CMplSphMeshBlock GetCoordinateSystem<CMplSphMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<MplSphMeshBlock>(rc);
}

template <>
CMplCartMeshBlock GetCoordinateSystem<CMplCartMeshBlock>(MeshBlockData<Real> *rc) {
  return GetCachedCoordinateSystem<MplCartMeshBlock>(rc);
}

template <>
CMplSphMesh GetCoordinateSystem<CMplSphMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<MplSphMesh>(rc);
}

template <>
CMplCartMesh GetCoordinateSystem<CMplCartMesh>(MeshData<Real> *rc) {
  return GetCachedCoordinateSystem<MplCartMesh>(rc);
}

template <>
void SetGeometry<CMplSphMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<MplSphMeshBlock>(rc);
}

template <>
void SetGeometry<CMplCartMeshBlock>(MeshBlockData<Real> *rc) {
  SetCachedCoordinateSystem<MplCartMeshBlock>(rc);
}

template <>
void SetGeometry<CMplSphMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<MplSphMesh>(rc);
}

template <>
void SetGeometry<CMplCartMesh>(MeshData<Real> *rc) {
  SetCachedCoordinateSystem<MplCartMesh>(rc);
}

} // namespace Geometry
