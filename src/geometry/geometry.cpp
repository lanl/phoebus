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
#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::ParArray1D;

namespace Geometry {

template <typename System>
void SetGeometryDefault(MeshBlockData<Real> *rc, const System &system) {
  auto pmb = rc->GetParentPointer();
  std::vector<std::string> coord_names = {geometric_variables::cell_coords,
                                          geometric_variables::node_coords};
  PackIndexMap imap;
  auto pack = rc->PackVariables(coord_names, imap);
  PARTHENON_REQUIRE(imap["g.c.coord"].second >= 0, "g.c.coord exists");
  PARTHENON_REQUIRE(imap["g.n.coord"].second >= 0, "g.n.coord exists");
  PARTHENON_REQUIRE(imap["g.c.coord"].second - imap["g.c.coord"].first + 1 == 4,
                    "g.c.coord has correct shape");
  PARTHENON_REQUIRE(imap["g.n.coord"].second - imap["g.n.coord"].first + 1 == 4,
                    "g.n.coord has correct shape");
  int icoord_c = imap["g.c.coord"].first;
  int icoord_n = imap["g.n.coord"].first;

  auto lamb =
      KOKKOS_LAMBDA(const int k, const int j, const int i, CellLocation loc) {
    Real C[NDFULL];
    int icoord = (loc == CellLocation::Cent) ? icoord_c : icoord_n;
    system.Coords(loc, k, j, i, C);
    SPACETIMELOOP(mu) pack(icoord + mu, k, j, i) = C[mu];
  };
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "SetGeometry::Set Cached data, Cent", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Cent);
      });
  pmb->par_for(
      "SetGeometry::Set Cached data, Corn", kb.s, kb.e + 1, jb.s, jb.e + 1,
      ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Corn);
      });
  pmb->exec_space.fence(); // do not let users interact with coords
                           // unless meshblock data is set
}

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
TaskStatus SetGeometryTask(MeshBlockData<Real> *rc) {
  if (CoordinatesNeedSetting<CoordSysMeshBlock>(rc)) {
    std::cout << "Setting" << std::endl;
    auto system = GetCoordinateSystem(rc);
    SetGeometry<CoordSysMeshBlock>(rc);
    SetGeometryDefault(rc, system);
  }
  return TaskStatus::complete;
}

} // namespace Geometry
