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
#include <string>

#include <bvals/boundary_conditions.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "fluid/fluid.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "radiation/radiation.hpp"

namespace Boundaries {

// Copied out of Parthenon, with slight modification
enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };
template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  // convenient shorthands
  constexpr bool IS_X1 = (DIR == X1DIR);
  constexpr bool IS_X2 = (DIR == X2DIR);
  constexpr bool IS_X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  // Pull out loop bounds
  auto pmb = rc->GetBlockPointer();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = IS_X1 ? bounds.GetBoundsI(IndexDomain::interior)
                            : (IS_X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                                     : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  // Variable pack
  auto q = rc->PackVariables(std::vector<parthenon::MetadataFlag>{Metadata::FillGhost},
                             coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  // Loop label
  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  // Index Domain
  constexpr IndexDomain domain =
      INNER ? (IS_X1 ? IndexDomain::inner_x1
                     : (IS_X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (IS_X1 ? IndexDomain::outer_x1
                     : (IS_X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  // needed because our conserved variables are densitized
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  // Do the thing
  pmb->par_for_bndry(
      label, nb, domain, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        int kref, jref, iref, sgn;
        if (TYPE == BCType::Reflect) {
          kref = IS_X3 ? offset - k : k;
          jref = IS_X2 ? offset - j : j;
          iref = IS_X1 ? offset - i : i;
          sgn = (q.VectorComponent(l) == DIR) ? -1.0 : 1.0;
        } else {
          kref = IS_X3 ? ref : k;
          jref = IS_X2 ? ref : j;
          iref = IS_X1 ? ref : i;
          sgn = 1;
        }

        const Real detg_ref = geom.DetGamma(CellLocation::Cent, kref, jref, iref);
        const Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
        const Real gratio = robust::ratio(detg, detg_ref);

        //        if ((i == 117 && j == 0)) {
        //          printf("[%i %i %i][%i] ref: q(%i %i %i %i) = %e result: %e sgn = %i
        //          gratio = %e\n",
        //            k, j, i, l, l, kref, jref, iref, q(l, kref, jref, iref),
        //              q(l, k, j, i), sgn, gratio);
        //        }
        q(l, k, j, i) = sgn * gratio * q(l, kref, jref, iref);
      });
}

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  PackIndexMap imap;
  auto q = rc->PackVariables(std::vector<parthenon::MetadataFlag>{Metadata::FillGhost},
                             imap, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  auto nb1 = IndexRange{0, 0};
  auto domain = IndexDomain::inner_x1;

  const int pv_lo = imap[fluid_prim::velocity].first;
  auto idx_J = imap.GetFlatIdx(radmoment_prim::J, false);
  auto idx_H = imap.GetFlatIdx(radmoment_prim::H, false);

  auto &fluid = rc->GetParentPointer()->packages.Get("fluid");
  auto &rad = rc->GetParentPointer()->packages.Get("radiation");
  std::string bc_vars = fluid->Param<std::string>("bc_vars");
  const int num_species = rad->Param<bool>("active") ? rad->Param<int>("num_species") : 0;

  if (bc_vars == "conserved") {
    pmb->par_for_bndry(
        "OutflowInnerX1Cons", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
          Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gratio = robust::ratio(detg, detg_ref);
          q(l, k, j, i) = gratio * q(l, k, j, ref);
        });
  } else if (bc_vars == "primitive") {
    pmb->par_for_bndry(
        "OutflowInnerX1Prim", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          q(l, k, j, i) = q(l, k, j, ref);

          //          // Enforce u^1 <= 0
          //          Real vcon[3] = {q(pv_lo, k, j, i), q(pv_lo + 1, k, j, i),
          //                          q(pv_lo + 2, k, j, i)};
          //          Real gammacov[3][3];
          //          geom.Metric(CellLocation::Cent, k, j, i, gammacov);
          //          Real W = phoebus::GetLorentzFactor(vcon, gammacov);
          //          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          //          Real beta[3];
          //          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          //
          //          Real ucon1 = vcon[0] - W * beta[0] / alpha;
          //
          //          if (ucon1 > 0) {
          //            SPACELOOP(ii) { vcon[ii] /= W; }
          //            vcon[0] = beta[0] / alpha;
          //            Real vsq = 0.;
          //            SPACELOOP2(ii, jj) { vsq += gammacov[ii][jj] * vcon[ii] *
          //            vcon[jj]; } W = 1. / sqrt(1. - vsq);
          //
          //            SPACELOOP(ii) { q(pv_lo + ii, k, j, i) = W * vcon[ii]; }
          //          }
          //
        });

    pmb->par_for_bndry(
        "OutflowInnerX1PrimFixup", nb1, domain, coarse,
        KOKKOS_LAMBDA(const int &dummy, const int &k, const int &j, const int &i) {
          // Enforce u^1 <= 0
          Real vcon[3] = {q(pv_lo, k, j, i), q(pv_lo + 1, k, j, i),
                          q(pv_lo + 2, k, j, i)};
          Real gammacov[3][3];
          geom.Metric(CellLocation::Cent, k, j, i, gammacov);
          Real W = phoebus::GetLorentzFactor(vcon, gammacov);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);

          Real ucon1 = vcon[0] - W * beta[0] / alpha;

          if (ucon1 > 0) {
            SPACELOOP(ii) { vcon[ii] /= W; }
            vcon[0] = beta[0] / alpha;
            Real vsq = 0.;
            SPACELOOP2(ii, jj) { vsq += gammacov[ii][jj] * vcon[ii] * vcon[jj]; }
            W = 1. / sqrt(1. - vsq);

            SPACELOOP(ii) { q(pv_lo + ii, k, j, i) = W * vcon[ii]; }
          }

          // No flux of radiation into the simulation
          if (idx_H.IsValid()) {
            Real gammacon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gammacon);
            for (int ispec = 0; ispec < num_species; ispec++) {
              Real Hcon[3] = {0};
              SPACELOOP2(ii, jj) {
                Hcon[ii] += gammacon[ii][jj] * q(idx_H(ispec, jj), k, j, i);
              }
              if (Hcon[0] > 0.) {
                Hcon[0] = 0.;

                // Check xi
                Real xi = 0.;
                SPACELOOP2(ii, jj) { xi += gammacov[ii][jj] * Hcon[ii] * Hcon[jj]; }
                xi = std::sqrt(xi);
                if (xi > 0.99) {
                  SPACELOOP(ii) { Hcon[ii] *= 0.99 / xi; }
                }

                SPACELOOP(ii) {
                  q(idx_H(ispec, ii), k, j, i) = 0.;
                  SPACELOOP(jj) {
                    q(idx_H(ispec, ii), k, j, i) += gammacov[ii][jj] * Hcon[jj];
                  }
                }
              }
            }
          }
        });
  }
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  PackIndexMap imap;
  auto q = rc->PackVariables(std::vector<parthenon::MetadataFlag>{Metadata::FillGhost},
                             imap, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  auto nb1 = IndexRange{0, 0};

  auto domain = IndexDomain::outer_x1;

  const int pv_lo = imap[fluid_prim::velocity].first;
  auto idx_H = imap.GetFlatIdx(radmoment_prim::H, false);
  // TODO(BRR): debug
  auto idx_J = imap.GetFlatIdx(radmoment_prim::J, false);

  auto &fluid = rc->GetParentPointer()->packages.Get("fluid");
  auto &rad = rc->GetParentPointer()->packages.Get("radiation");
  std::string bc_vars = fluid->Param<std::string>("bc_vars");
  const int num_species = rad->Param<bool>("active") ? rad->Param<int>("num_species") : 0;

  if (bc_vars == "conserved") {
    pmb->par_for_bndry(
        "OutflowOuterX1Cons", nb, IndexDomain::outer_x1, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          Real detg_ref = geom.DetGamma(CellLocation::Cent, k, j, ref);
          Real detg = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gratio = robust::ratio(detg, detg_ref);
          q(l, k, j, i) = gratio * q(l, k, j, ref);
        });
  } else if (bc_vars == "primitive") {
    pmb->par_for_bndry(
        "OutflowOuterX1Prim", nb, domain, coarse,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          q(l, k, j, i) = q(l, k, j, ref);

          // Enforce u^1 >= 0
          //          Real vcon[3] = {q(pv_lo, k, j, i), q(pv_lo + 1, k, j, i),
          //                          q(pv_lo + 2, k, j, i)};
          //          Real gammacov[3][3];
          //          geom.Metric(CellLocation::Cent, k, j, i, gammacov);
          //          Real W = phoebus::GetLorentzFactor(vcon, gammacov);
          //          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          //          Real beta[3];
          //          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          //
          //          Real ucon1 = vcon[0] - W * beta[0] / alpha;
          //
          //          if (ucon1 < 0) {
          //            SPACELOOP(ii) { vcon[ii] /= W; }
          //            vcon[0] = beta[0] / alpha;
          //            Real vsq = 0.;
          //            SPACELOOP2(ii, jj) { vsq += gammacov[ii][jj] * vcon[ii] *
          //            vcon[jj]; } W = 1. / sqrt(1. - vsq);
          //
          //            SPACELOOP(ii) { q(pv_lo + ii, k, j, i) = W * vcon[ii]; }
          //          }
        });

    pmb->par_for_bndry(
        "OutflowOuterX1PrimFixup", nb1, domain, coarse,
        KOKKOS_LAMBDA(const int &dummy, const int &k, const int &j, const int &i) {
          //       q(l, k, j, i) = q(l, k, j, ref);

          // Enforce u^1 >= 0
          Real vcon[3] = {q(pv_lo, k, j, i), q(pv_lo + 1, k, j, i),
                          q(pv_lo + 2, k, j, i)};
          Real gammacov[3][3];
          geom.Metric(CellLocation::Cent, k, j, i, gammacov);
          Real W = phoebus::GetLorentzFactor(vcon, gammacov);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);

          Real ucon1 = vcon[0] - W * beta[0] / alpha;

          if (ucon1 < 0) {
            SPACELOOP(ii) { vcon[ii] /= W; }
            vcon[0] = beta[0] / alpha;
            Real vsq = 0.;
            SPACELOOP2(ii, jj) { vsq += gammacov[ii][jj] * vcon[ii] * vcon[jj]; }
            W = 1. / sqrt(1. - vsq);

            SPACELOOP(ii) { q(pv_lo + ii, k, j, i) = W * vcon[ii]; }
          }

          // No flux of radiation into the simulation
          if (idx_H.IsValid()) {
            Real gammacon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gammacon);
            for (int ispec = 0; ispec < num_species; ispec++) {
              Real Hcon[3] = {0};
              SPACELOOP2(ii, jj) {
                Hcon[ii] += gammacon[ii][jj] * q(idx_H(ispec, jj), k, j, i);
              }
              if (Hcon[0] < 0.) {
                Hcon[0] = 0.;

                // Check xi
                Real xi = 0.;
                SPACELOOP2(ii, jj) { xi += gammacov[ii][jj] * Hcon[ii] * Hcon[jj]; }
                xi = std::sqrt(xi);
                if (xi > 0.99) {
                  SPACELOOP(ii) { Hcon[ii] *= 0.99 / xi; }
                }

                SPACELOOP(ii) {
                  q(idx_H(ispec, ii), k, j, i) = 0.;
                  SPACELOOP(jj) {
                    q(idx_H(ispec, ii), k, j, i) += gammacov[ii][jj] * Hcon[jj];
                  }
                }
              }
              // q(idx_H(ispec, 0), k, j, i) = std::max<Real>(q(idx_H(ispec), 0, k, j, i),
              // 0.);
            }
          }
        });
  }
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Inner, BCType::Reflect>(rc, coarse);
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Outer, BCType::Reflect>(rc, coarse);
}

void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Inner, BCType::Outflow>(rc, coarse);
}

void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Outer, BCType::Outflow>(rc, coarse);
}

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Inner, BCType::Reflect>(rc, coarse);
}

void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Outer, BCType::Reflect>(rc, coarse);
}

void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Inner, BCType::Outflow>(rc, coarse);
}

void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Outer, BCType::Outflow>(rc, coarse);
}

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Inner, BCType::Reflect>(rc, coarse);
}

void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Outer, BCType::Reflect>(rc, coarse);
}

TaskStatus ConvertBoundaryConditions(std::shared_ptr<MeshBlockData<Real>> &rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
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

  auto &pkg = rc->GetParentPointer()->packages.Get("fluid");
  if (pkg->Param<bool>("active")) {
    std::string bc_vars = pkg->Param<std::string>("bc_vars");
    if (bc_vars == "primitive") {
      // auto c2p = pkg->Param<fluid::c2p_meshblock_type>("c2p_func");
      for (auto &domain : domains) {
        IndexRange ib = rc->GetBoundsI(domain);
        IndexRange jb = rc->GetBoundsJ(domain);
        IndexRange kb = rc->GetBoundsK(domain);
        fluid::PrimitiveToConservedRegion(rc.get(), ib, jb, kb);
      }
    }
  }

  auto &pkg_rad = rc->GetParentPointer()->packages.Get("radiation");
  if (pkg_rad->Param<bool>("active")) {
    std::string bc_vars = pkg_rad->Param<std::string>("bc_vars");
    if (bc_vars == "primitive") {
      for (auto &domain : domains) {
        // TODO(BRR) Make this per-region!
        radiation::MomentPrim2Con(rc.get(), domain);
      }
    }
  }

  return TaskStatus::complete;
}

void ProcessBoundaryConditions(parthenon::ParthenonManager &pman) {
  // Ensure only allowed parthenon boundary conditions are used
  const std::vector<std::string> inner_outer = {"i", "o"};
  static const parthenon::BoundaryFace loc[][2] = {
      {parthenon::BoundaryFace::inner_x1, parthenon::BoundaryFace::outer_x1},
      {parthenon::BoundaryFace::inner_x2, parthenon::BoundaryFace::outer_x2},
      {parthenon::BoundaryFace::inner_x3, parthenon::BoundaryFace::outer_x3}};
  static const parthenon::BValFunc outflow[][2] = {
      {Boundaries::OutflowInnerX1, Boundaries::OutflowOuterX1},
      {Boundaries::OutflowInnerX2, Boundaries::OutflowOuterX2},
      {Boundaries::OutflowInnerX3, Boundaries::OutflowOuterX3}};
  static const parthenon::BValFunc reflect[][2] = {
      {Boundaries::ReflectInnerX1, Boundaries::ReflectOuterX1},
      {Boundaries::ReflectInnerX2, Boundaries::ReflectOuterX2},
      {Boundaries::ReflectInnerX3, Boundaries::ReflectOuterX3}};

  const std::string rad_method =
      pman.pinput->GetOrAddString("radiation", "method", "None");

  for (int d = 1; d <= 3; ++d) {
    // outer = 0 for inner face, outer = 1 for outer face
    for (int outer = 0; outer <= 1; ++outer) {
      auto &face = inner_outer[outer];
      const std::string name = face + "x" + std::to_string(d) + "_bc";
      const std::string parth_bc = pman.pinput->GetString("parthenon/mesh", name);
      PARTHENON_REQUIRE(parth_bc == "user" || parth_bc == "periodic",
                        "Only \"user\" and \"periodic\" allowed for parthenon/mesh/" +
                            name);

      const std::string bc = pman.pinput->GetOrAddString("phoebus", name, "outflow");
      if (bc == "reflect") {
        pman.app_input->boundary_conditions[loc[d - 1][outer]] = reflect[d - 1][outer];
      } else if (bc == "outflow") {
        pman.app_input->boundary_conditions[loc[d - 1][outer]] = outflow[d - 1][outer];
        if (d == 1) {
          if (outer == 0) {
            if (rad_method == "mocmc") {
              pman.app_input
                  ->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
                  Boundaries::SetSwarmNoWorkBC;
            } else {
              pman.app_input
                  ->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
                  Boundaries::SetSwarmIX1Outflow;
            }
          } else if (outer == 1) {
            if (rad_method == "mocmc") {
              pman.app_input
                  ->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
                  Boundaries::SetSwarmNoWorkBC;
            } else {
              pman.app_input
                  ->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
                  Boundaries::SetSwarmOX1Outflow;
            }
          }
        }
      } // periodic boundaries, which are handled by parthenon, so no need to set anything
    }
  }
}

} // namespace Boundaries
