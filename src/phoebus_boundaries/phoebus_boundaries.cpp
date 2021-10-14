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

#include "fluid/fluid.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"

namespace Boundaries {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  PackIndexMap imap;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, imap, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  auto domain = IndexDomain::inner_x1;

  const int pv1 = imap[fluid_prim::velocity].first;

  auto &pkg = rc->GetParentPointer()->packages.Get("fluid");
  std::string bc_vars = pkg->Param<std::string>("bc_vars");

  if (bc_vars == "conserved") {
    pmb->par_for_bndry(
        "OutflowInnerX1Cons", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
          Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gratio = Geometry::Utils::ratio(detg, detg_ref);
          q(l, k, j, i) = gratio * q(l, k, j, ref);
        });
  } else if (bc_vars == "primitive") {
    pmb->par_for_bndry(
        "OutflowInnerX1Prim", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          q(l, k, j, i) = q(l, k, j, ref);

          // Check that u^1 <= 0
          Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent,k,j,i,beta);
          //if (q(pv1, k, j, i) - beta[0]/alpha > 0.) {
          if (q(pv1, k, j, i) > 0.) {
            //q(pv1, k, j, i) = beta[0]/alpha;
            q(pv1, k, j, i) = 0.;
          }
        });
  }
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  PackIndexMap imap;
  auto q = rc->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::FillGhost}, imap, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  auto domain = IndexDomain::outer_x1;
  
  const int pv1 = imap[fluid_prim::velocity].first;

  auto &pkg = rc->GetParentPointer()->packages.Get("fluid");
  std::string bc_vars = pkg->Param<std::string>("bc_vars");

  if (bc_vars == "conserved") {
    pmb->par_for_bndry(
        "OutflowOuterX1Cons", nb, IndexDomain::outer_x1, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
          Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gratio = Geometry::Utils::ratio(detg, detg_ref);
          q(l, k, j, i) = gratio * q(l, k, j, ref);
        });
  } else if (bc_vars == "primitive") {
    pmb->par_for_bndry(
        "OutflowOuterX1Prim", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          q(l, k, j, i) = q(l, k, j, ref);
          
          // Check that u^1 <= 0
          Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent,k,j,i,beta);
          //if (q(pv1, k, j, i) - beta[0]/alpha < 0.) {
          if (q(pv1, k, j, i) < 0.) {
            //q(pv1, k, j, i) = beta[0]/alpha;
            q(pv1, k, j, i) = 0.;
          }
        });
  }
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

TaskStatus ConvertBoundaryConditions (std::shared_ptr<MeshBlockData<Real>> &rc) {
  auto &pkg = rc->GetParentPointer()->packages.Get("fluid");
  std::string bc_vars = pkg->Param<std::string>("bc_vars");
  auto pmb = rc->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  std::vector<IndexDomain> domains = {IndexDomain::inner_x1, IndexDomain::outer_x1};
  if (ndim > 1) {
    domains.push_back(IndexDomain::inner_x2);
    domains.push_back(IndexDomain::outer_x2);
    if (ndim > 2) {
      domains.push_back(IndexDomain::inner_x3);
      domains.push_back(IndexDomain::outer_x3);
    }
  }

  if (bc_vars == "primitive") {
    //auto c2p = pkg->Param<fluid::c2p_meshblock_type>("c2p_func");
    for (auto &domain : domains) {
      IndexRange ib = rc->GetBoundsI(domain);
      IndexRange jb = rc->GetBoundsJ(domain);
      IndexRange kb = rc->GetBoundsK(domain);
      //c2p(rc.get(), ib, jb, kb);
      fluid::PrimitiveToConservedRegion(rc.get(), ib, jb, kb);
    }
  }

  return TaskStatus::complete;
}

} // namespace Boundaries
