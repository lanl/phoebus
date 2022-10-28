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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include "geometry/coordinate_systems.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::ParArray1D;

namespace Geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  auto geometry = std::make_shared<StateDescriptor>("geometry");
  Initialize<CoordSysMeshBlock>(pin, geometry.get());

  Params &params = geometry->AllParams();

  // Store name of geometry in parameters
  std::string geometry_name;
  if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski)) {
    geometry_name = "Minkowski";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::BoostedMinkowski)) {
    geometry_name = "BoostedMinkowski";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski)) {
    geometry_name = "SphericalMinkowski";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::BoyerLindquist)) {
    geometry_name = "BoyerLindquist";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalKerrSchild)) {
    geometry_name = "SphericalKerrSchild";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FLRW)) {
    geometry_name = "FLRW";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS)) {
    geometry_name = "FMKS";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake)) {
    geometry_name = "Snake";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Inchworm)) {
    geometry_name = "Inchworm";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph)) {
    geometry_name = "MonopoleSph";
  } else if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleCart)) {
    geometry_name = "MonopoleCart";
  } else {
    PARTHENON_THROW("PHOEBUS_GEOMETRY not recognized!");
  }
  params.Add("geometry_name", geometry_name);

  // Always add coodinates fields
  Utils::MeshBlockShape dims(pin);
  std::vector<int> cell_shape = {4};
  Metadata gcoord_cell =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, cell_shape);
  // TODO(JMM): Make this actual node-centered data when available
  std::vector<int> node_shape = {dims.nx1 + 1, dims.nx2 + 1, dims.nx3 + 1, 4};
  Metadata gcoord_node = Metadata({Metadata::Derived, Metadata::OneCopy}, node_shape);
  geometry->AddField(geometric_variables::cell_coords, gcoord_cell);
  geometry->AddField(geometric_variables::node_coords, gcoord_node);

  return geometry;
}

CoordSysMeshBlock GetCoordinateSystem(MeshBlockData<Real> *rc) {
  return GetCoordinateSystem<CoordSysMeshBlock>(rc);
}
CoordSysMesh GetCoordinateSystem(MeshData<Real> *rc) {
  return GetCoordinateSystem<CoordSysMesh>(rc);
}

void SetGeometryBlock(MeshBlock *pmb, ParameterInput *pin) {
  MeshBlockData<Real> *rc = pmb->meshblock_data.Get().get();
  auto *pparent = rc->GetParentPointer().get();
  StateDescriptor *pkg = pparent->packages.Get("geometry").get();
  bool do_defaults = pkg->AllParams().Get("do_defaults", true);
  auto system = GetCoordinateSystem(rc);
  SetGeometry<CoordSysMeshBlock>(rc);
  if (do_defaults) impl::SetGeometryDefault(rc, system);
}

template <>
TaskStatus UpdateGeometry<MeshBlockData<Real>>(MeshBlockData<Real> *rc) {
  auto *pparent = rc->GetParentPointer().get();
  StateDescriptor *pkg = pparent->packages.Get("geometry").get();
  bool update_coords = pkg->AllParams().Get("update_coords", false);
  auto system = GetCoordinateSystem(rc);
  SetGeometry<CoordSysMeshBlock>(rc);
  if (update_coords) impl::SetGeometryDefault(rc, system);
  return TaskStatus::complete;
}

template <>
TaskStatus UpdateGeometry<MeshData<Real>>(MeshData<Real> *rc) {
  auto *pparent = rc->GetParentPointer();
  StateDescriptor *pkg = pparent->packages.Get("geometry").get();
  bool update_coords = pkg->AllParams().Get("update_coords", false);
  auto system = GetCoordinateSystem(rc);
  SetGeometry<CoordSysMesh>(rc);
  if (update_coords) impl::SetGeometryDefault(rc, system);
  return TaskStatus::complete;
}

} // namespace Geometry
