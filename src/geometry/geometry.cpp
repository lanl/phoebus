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
#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::ParArray1D;

namespace Geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  auto geometry = std::make_shared<StateDescriptor>("geometry");
  Initialize<CoordSysMeshBlock>(pin, geometry.get());

  // Always add coodinates fields
  Utils::MeshBlockShape dims(pin);
  std::vector<int> cell_shape = {4};
  Metadata gcoord_cell = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, cell_shape);
  // TODO(JMM): Make this actual node-centered data when available
  std::vector<int> node_shape = {dims.nx1 + 1, dims.nx2 + 1, dims.nx3 + 1, 4};
  Metadata gcoord_node =
      Metadata({Metadata::Derived, Metadata::OneCopy}, node_shape);
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
void SetGeometry(MeshBlockData<Real> *rc) {
  SetGeometry<CoordSysMeshBlock>(rc);
}

} // namespace Geometry
