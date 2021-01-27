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

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::ParArray1D;

namespace Geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto geometry = std::make_shared<StateDescriptor>("geometry");
  Params &params = geometry->AllParams();

  // has to be a pointer, because inheritence
  // TODO(JMM): Perhaps another reason to use variants here?
  std::string coord_system =
      pin->GetOrAddString("coordinates", "system", "minkowski");
  CoordSystemTag tag;
  if (coord_system == "minkowski") {
    // more pinputs could go here
    // and should be added to params as needed.
    tag = CoordSystemTag::Minkowski;
  } else { // default
    PARTHENON_THROW("unknown coordinate system");
  }
  params.Add("coordinate_system", tag);

  // Add fields here if needed
  /*
  std::string field_name;
  std::vector<int> shape;
  std::vector<MetadataFlag> flags;
  Metadata m;
  // ...
  */

  return geometry;
}

// These are overloaded, rather than templated because
// the per-system logic probably needs to change between MeshData
// and MeshBlockData.
CoordinateSystem GetCoordinateSystem(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  auto &geom = pmb->packages.Get("geometry");
  auto tag = geom->Param<CoordSystemTag>("coordinate_system");
  auto &coords = pmb->coords;

  // some coordinate systems may require more inputs to
  // constructor. Can be pulled out of params or the MeshBlockData
  // object.
  switch(tag) {
  case CoordSystemTag::Minkowski:
    return CoordinateSystem(Analytic<Minkowski>(coords));
  default:
    PARTHENON_THROW("unknown coordinate system");
  }
}

CoordinateSystem GetCoordinateSystem(MeshData<Real> *rc) {
  auto pmesh = rc->GetParentPointer();
  auto &geom = pmesh->packages.Get("geometry");
  auto tag = geom->Param<CoordSystemTag>("coordinate_system");

  // TODO(JMM): Cache this somehow?
  int nblocks = rc->NumBlocks();
  ParArray1D<Coordinates_t> coords("GetCoordinateSystem::coords",nblocks);
  auto coords_h = Kokkos::create_mirror_view(coords);
  for (int i = 0; i < nblocks; ++i) {
    coords_h(i) = rc->GetBlockData(i)->GetBlockPointer()->coords;
  }
  Kokkos::deep_copy(coords, coords_h);

  // some coordinate systems may require more inputs to
  // constructor. Can be pulled out of params or the MeshBlockData
  // object.
  switch(tag) {
  case CoordSystemTag::Minkowski:
    return CoordinateSystem(Analytic<Minkowski>(coords));
  default:
    PARTHENON_THROW("unknown coordinate system");
  }
}

// For now, a no-op.
void SetGeometry(MeshBlockData<Real> *rc) {}

} // namespace Geometry
