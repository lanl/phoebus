#ifndef GEOMETRY_GEOMETRY_DEFAULTS_HPP_
#define GEOMETRY_GEOMETRY_DEFAULTS_HPP_

#include <string>
#include <vector>

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace Geometry {

template <typename System>
void Initialize(ParameterInput *pin, StateDescriptor *geometry) {}

template <typename System> System GetCoordinateSystem(MeshBlockData<Real> *rc);// {
//  PARTHENON_THROW("Unknown coordinate system");
//}

template <typename System> System GetCoordinateSystem(MeshData<Real> *rc);// {
//  PARTHENON_THROW("Unknown coordinate system");
//}

template <typename System> void SetGeometry(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  std::vector<std::string> coord_names = {geometric_variables::cell_coords,
                                          geometric_variables::node_coords};
  PackIndexMap imap;
  auto pack = rc->PackVariables(coord_names, imap);
  int icoord_c = imap["g.c.coord"].first;
  int icoord_n = imap["g.n.coord"].first;
  auto system = GetCoordinateSystem<System>(rc);

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
      "SetGeometry::Set Cached data, Corn", kb.s, kb.e + 1, jb.s, jb.e+1, ib.s,
      ib.e+1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Corn);
      });
}

template <typename Transformation>
Transformation GetTransformation(StateDescriptor *pkg) {
  PARTHENON_THROW("Unknown transformation");
}

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_DEFAULTS_HPP_
