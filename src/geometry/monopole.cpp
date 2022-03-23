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
  // All initialization done by MonopoleGR machinery
  return;
}
template <>
void Initialize<MplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry) {
  // All initialization done by MonopoleGR machinery
  return;
}
// TODO(JMM): Might need to figure this out more carefully
template <>
void SetGeometry<MplSphMeshBlock>(MeshBlockData<Real> *rc) {}
template <>
void SetGeometry<MplCartMeshBlock>(MeshBlockData<Real> *rc) {}

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
  return MplCartMeshBlock(indexer, hypersurface, alpha, beta, gradients, rgrid);
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
  return MplCartMesh(indexer, hypersurface, alpha, beta, gradients, rgrid);
}

// TODO(JMM): These cached coordinate system calls may need to be revisited.

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

} // namespace Geometry
