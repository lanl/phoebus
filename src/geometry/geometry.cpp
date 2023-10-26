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

#include "analysis/history.hpp"
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
  const std::string gname = GEOMETRY_NAME;
  const std::string geometry_name = gname.substr(gname.find("::") + 2);
  PARTHENON_REQUIRE_THROWS(geometry_name.size() > 0, "Invalid geometry name!");
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

  // Reductions
  const bool do_mhd = pin->GetorAddBoolean("fluid", "mhd", false);
  if (params.hasKey("xh") && do_mhd) {
    auto HstSum = parthenon::UserHistoryOperation::sum;
    using History::ReduceJetEnergyFlux;
    using History::ReduceJetMomentumFlux;
    using History::ReduceMagneticFluxPhi;
    using History::ReduceMassAccretionRate;
    using parthenon::HistoryOutputVar;
    parthenon::HstVar_list hst_vars = {};

    auto ReduceAccretionRate = [](MeshData<Real> *md) {
      return ReduceMassAccretionRate(md);
    };

    auto ComputeJetEnergyFlux = [](MeshData<Real> *md) {
      return ReduceJetEnergyFlux(md);
    };

    auto ComputeJetMomentumFlux = [](MeshData<Real> *md) {
      return ReduceJetMomentumFlux(md);
    };

    auto ComputeMagneticFluxPhi = [](MeshData<Real> *md) {
      return ReduceMagneticFluxPhi(md);
    };

    hst_vars.emplace_back(HistoryOutputVar(HstSum, ReduceAccretionRate, "mdot"));
    hst_vars.emplace_back(
        HistoryOutputVar(HstSum, ComputeJetEnergyFlux, "jet_energy_flux"));
    hst_vars.emplace_back(
        HistoryOutputVar(HstSum, ComputeJetMomentumFlux, "jet_momentum_flux"));
    hst_vars.emplace_back(
        HistoryOutputVar(HstSum, ComputeMagneticFluxPhi, "magnetic_Phi"));
    params.Add(parthenon::hist_param_key, hst_vars);
  }

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
