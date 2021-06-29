//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <memory>

#include <bvals/boundary_conditions.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"

namespace Boundaries {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "OutflowInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        q(l, k, j, i) = gratio * q(l, k, j, ref);
      });
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "OutflowOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        q(l, k, j, i) = gratio * q(l, k, j, ref);
      });
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "ReflectInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        int iref = 2*ref - i - 1;
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, iref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = gratio * reflect * q(l, k, j, iref);
      });
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "ReflectOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        int iref = 2*ref - i + 1;
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, iref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = gratio * reflect * q(l, k, j, iref);
      });
}

/*void OutflowInnerX1Primitives(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "OutflowInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        q(l, k, j, i) = gratio * q(l, k, j, ref);
      });
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "OutflowOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        q(l, k, j, i) = gratio * q(l, k, j, ref);
      });
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "ReflectInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        int iref = 2*ref - i - 1;
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, iref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = gratio * reflect * q(l, k, j, iref);
      });
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  pmb->par_for_bndry(
      "ReflectOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        int iref = 2*ref - i + 1;
        Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, iref);
        Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real gratio = Geometry::Utils::ratio(detg, detg_ref);
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = gratio * reflect * q(l, k, j, iref);
      });
}*/

TaskStatus ConvertBoundaryConditions (std::shared_ptr<MeshBlockData<Real>> &rc) {
  return TaskStatus::complete;
}

} // namespace Boundaries
