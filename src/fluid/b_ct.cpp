// © 2026. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

#include "b_ct.hpp"

#include <string>
#include <vector>

#include "b_ct_functions.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

namespace b_ct {

using TE = parthenon::TopologicalElement;

TaskStatus BlockFaceToCell(MeshBlockData<Real> *rc, IndexDomain domain) {
  auto pmb = rc->GetBlockPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd")) return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;
  auto Bf = rc->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  auto Bc = rc->PackVariables(std::vector<std::string>{fluid_cons::bfield::name()});
  auto Bp = rc->PackVariables(std::vector<std::string>{fluid_prim::bfield::name()});
  auto geom = Geometry::GetCoordinateSystem(rc);

  IndexRange ib = rc->GetBoundsI(domain);
  IndexRange jb = rc->GetBoundsJ(domain);
  IndexRange kb = rc->GetBoundsK(domain);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "b_ct::BlockFaceToCell", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real bp1 =
            0.5 * (Bf(TE::F1, 0, k, j, i) / geom.DetGamma(CellLocation::Face1, k, j, i) +
                   Bf(TE::F1, 0, k, j, i + 1) /
                       geom.DetGamma(CellLocation::Face1, k, j, i + 1));
        Real bp2, bp3;
        if (ndim > 1) {
          bp2 = 0.5 *
                (Bf(TE::F2, 0, k, j, i) / geom.DetGamma(CellLocation::Face2, k, j, i) +
                 Bf(TE::F2, 0, k, j + 1, i) /
                     geom.DetGamma(CellLocation::Face2, k, j + 1, i));
        } else {
          bp2 = Bf(TE::F2, 0, k, j, i) / geom.DetGamma(CellLocation::Face2, k, j, i);
        }
        if (ndim > 2) {
          bp3 = 0.5 *
                (Bf(TE::F3, 0, k, j, i) / geom.DetGamma(CellLocation::Face3, k, j, i) +
                 Bf(TE::F3, 0, k + 1, j, i) /
                     geom.DetGamma(CellLocation::Face3, k + 1, j, i));
        } else {
          bp3 = Bf(TE::F3, 0, k, j, i) / geom.DetGamma(CellLocation::Face3, k, j, i);
        }
        Bp(0, k, j, i) = bp1;
        Bp(1, k, j, i) = bp2;
        Bp(2, k, j, i) = bp3;
        const Real gdetc = geom.DetGamma(CellLocation::Cent, k, j, i);
        Bc(0, k, j, i) = bp1 * gdetc;
        Bc(1, k, j, i) = bp2 * gdetc;
        Bc(2, k, j, i) = bp3 * gdetc;
      });
  return TaskStatus::complete;
}

TaskStatus MeshFaceToCell(MeshData<Real> *md, IndexDomain domain) {
  if (!md->GetMeshPointer()->packages.Get("fluid")->Param<bool>("mhd"))
    return TaskStatus::complete;
  for (int b = 0; b < md->NumBlocks(); b++) {
    BlockFaceToCell(md->GetBlockData(b).get(), domain);
  }
  return TaskStatus::complete;
}

TaskStatus CalculateEMF(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  if (!pmesh->packages.Get("fluid")->Param<bool>("mhd")) return TaskStatus::complete;
  const int ndim = pmesh->ndim;

  auto emf =
      md->PackVariables(std::vector<std::string>{internal_variables::eemf::name()});
  auto Bflux =
      md->PackVariablesAndFluxes(std::vector<std::string>{fluid_cons::bfield::name()});

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int dk = (ndim > 2) ? 1 : 0;
  const IndexRange block = IndexRange{0, emf.GetDim(5) - 1};

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  if (ndim == 1) {
    const auto ibe = md->GetBoundsI(IndexDomain::interior, TE::E2);
    const auto jbe = md->GetBoundsJ(IndexDomain::interior, TE::E2);
    const auto kbe = md->GetBoundsK(IndexDomain::interior, TE::E2);
    pmb0->par_for(
        "b_ct::CalculateEMF_1D", block.s, block.e, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s,
        ibe.e, KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
          emf(bl, TE::E2, 0, k, j, i) = Bflux(bl).flux(X1DIR, 2, k, j, i);
          emf(bl, TE::E3, 0, k, j, i) = -Bflux(bl).flux(X1DIR, 1, k, j, i);
        });
    return TaskStatus::complete;
  }

  pmb0->par_for(
      "b_ct::CalculateEMF_bs99", block.s, block.e, kb.s, kb.e + dk, jb.s, jb.e + 1, ib.s,
      ib.e + 1, KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        // Average Riemann B fluxes onto edges.
        if (ndim > 2) {
          emf(bl, TE::E1, 0, k, j, i) = 0.25 * (-Bflux(bl).flux(X2DIR, 2, k - 1, j, i) -
                                                Bflux(bl).flux(X2DIR, 2, k, j, i) +
                                                Bflux(bl).flux(X3DIR, 1, k, j - 1, i) +
                                                Bflux(bl).flux(X3DIR, 1, k, j, i));
          emf(bl, TE::E2, 0, k, j, i) = 0.25 * (-Bflux(bl).flux(X3DIR, 0, k, j, i - 1) -
                                                Bflux(bl).flux(X3DIR, 0, k, j, i) +
                                                Bflux(bl).flux(X1DIR, 2, k - 1, j, i) +
                                                Bflux(bl).flux(X1DIR, 2, k, j, i));
          emf(bl, TE::E3, 0, k, j, i) = 0.25 * (-Bflux(bl).flux(X1DIR, 1, k, j - 1, i) -
                                                Bflux(bl).flux(X1DIR, 1, k, j, i) +
                                                Bflux(bl).flux(X2DIR, 0, k, j, i - 1) +
                                                Bflux(bl).flux(X2DIR, 0, k, j, i));
        } else {
          emf(bl, TE::E1, 0, k, j, i) = -Bflux(bl).flux(X2DIR, 2, k, j, i);
          emf(bl, TE::E2, 0, k, j, i) = Bflux(bl).flux(X1DIR, 2, k, j, i);
          emf(bl, TE::E3, 0, k, j, i) = 0.25 * (-Bflux(bl).flux(X1DIR, 1, k, j - 1, i) -
                                                Bflux(bl).flux(X1DIR, 1, k, j, i) +
                                                Bflux(bl).flux(X2DIR, 0, k, j, i - 1) +
                                                Bflux(bl).flux(X2DIR, 0, k, j, i));
        }
      });

  // Gardiner-Stone upwind correction to the edge-average EMF.
  auto cemf =
      md->PackVariables(std::vector<std::string>{internal_variables::cemf::name()});
  auto Vel = md->PackVariables(std::vector<std::string>{fluid_prim::velocity::name()});
  auto Bp = md->PackVariables(std::vector<std::string>{fluid_prim::bfield::name()});
  auto Rflux =
      md->PackVariablesAndFluxes(std::vector<std::string>{fluid_cons::density::name()});
  auto geom = Geometry::GetCoordinateSystem(md);

  pmb0->par_for(
      "b_ct::CalculateEMF_cemf", block.s, block.e, kb.s - dk, kb.e + dk, jb.s - 1,
      jb.e + 1, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        const Real vpcon[3] = {Vel(bl, 0, k, j, i), Vel(bl, 1, k, j, i),
                               Vel(bl, 2, k, j, i)};
        const Real invW =
            1.0 / phoebus::GetLorentzFactor(vpcon, geom, CellLocation::Cent, bl, k, j, i);
        const Real B[3] = {Bp(bl, 0, k, j, i), Bp(bl, 1, k, j, i), Bp(bl, 2, k, j, i)};
        const Real inv_alpha = 1.0 / geom.Lapse(CellLocation::Cent, bl, k, j, i);
        Real beta[3];
        geom.ContravariantShift(CellLocation::Cent, bl, k, j, i, beta);
        const Real vtil[3] = {vpcon[0] * invW - beta[0] * inv_alpha,
                              vpcon[1] * invW - beta[1] * inv_alpha,
                              vpcon[2] * invW - beta[2] * inv_alpha};
        const Real detg = geom.DetG(CellLocation::Cent, bl, k, j, i);
        cemf(bl, 0, k, j, i) = detg * (B[1] * vtil[2] - B[2] * vtil[1]);
        cemf(bl, 1, k, j, i) = detg * (B[2] * vtil[0] - B[0] * vtil[2]);
        cemf(bl, 2, k, j, i) = detg * (B[0] * vtil[1] - B[1] * vtil[0]);
      });

  pmb0->par_for(
      "b_ct::CalculateEMF_upwind", block.s, block.e, kb.s, kb.e + dk, jb.s, jb.e + 1,
      ib.s, ib.e + 1, KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        if (ndim > 2) {
          const Real e1_l3 =
              (Rflux(bl).flux(X2DIR, 0, k - 1, j, i) >= 0.0)
                  ? Bflux(bl).flux(X3DIR, 1, k, j - 1, i) - cemf(bl, 0, k - 1, j - 1, i)
                  : Bflux(bl).flux(X3DIR, 1, k, j, i) - cemf(bl, 0, k - 1, j, i);
          const Real e1_r3 =
              (Rflux(bl).flux(X2DIR, 0, k, j, i) >= 0.0)
                  ? Bflux(bl).flux(X3DIR, 1, k, j - 1, i) - cemf(bl, 0, k, j - 1, i)
                  : Bflux(bl).flux(X3DIR, 1, k, j, i) - cemf(bl, 0, k, j, i);
          const Real e1_l2 =
              (Rflux(bl).flux(X3DIR, 0, k, j - 1, i) >= 0.0)
                  ? -Bflux(bl).flux(X2DIR, 2, k - 1, j, i) - cemf(bl, 0, k - 1, j - 1, i)
                  : -Bflux(bl).flux(X2DIR, 2, k, j, i) - cemf(bl, 0, k, j - 1, i);
          const Real e1_r2 =
              (Rflux(bl).flux(X3DIR, 0, k, j, i) >= 0.0)
                  ? -Bflux(bl).flux(X2DIR, 2, k - 1, j, i) - cemf(bl, 0, k - 1, j, i)
                  : -Bflux(bl).flux(X2DIR, 2, k, j, i) - cemf(bl, 0, k, j, i);
          emf(bl, TE::E1, 0, k, j, i) += 0.25 * (e1_l3 + e1_r3 + e1_l2 + e1_r2);

          const Real e2_l3 =
              (Rflux(bl).flux(X1DIR, 0, k - 1, j, i) >= 0.0)
                  ? -Bflux(bl).flux(X3DIR, 0, k, j, i - 1) - cemf(bl, 1, k - 1, j, i - 1)
                  : -Bflux(bl).flux(X3DIR, 0, k, j, i) - cemf(bl, 1, k - 1, j, i);
          const Real e2_r3 =
              (Rflux(bl).flux(X1DIR, 0, k, j, i) >= 0.0)
                  ? -Bflux(bl).flux(X3DIR, 0, k, j, i - 1) - cemf(bl, 1, k, j, i - 1)
                  : -Bflux(bl).flux(X3DIR, 0, k, j, i) - cemf(bl, 1, k, j, i);
          const Real e2_l1 =
              (Rflux(bl).flux(X3DIR, 0, k, j, i - 1) >= 0.0)
                  ? Bflux(bl).flux(X1DIR, 2, k - 1, j, i) - cemf(bl, 1, k - 1, j, i - 1)
                  : Bflux(bl).flux(X1DIR, 2, k, j, i) - cemf(bl, 1, k, j, i - 1);
          const Real e2_r1 =
              (Rflux(bl).flux(X3DIR, 0, k, j, i) >= 0.0)
                  ? Bflux(bl).flux(X1DIR, 2, k - 1, j, i) - cemf(bl, 1, k - 1, j, i)
                  : Bflux(bl).flux(X1DIR, 2, k, j, i) - cemf(bl, 1, k, j, i);
          emf(bl, TE::E2, 0, k, j, i) += 0.25 * (e2_l3 + e2_r3 + e2_l1 + e2_r1);
        }

        const Real e3_l2 =
            (Rflux(bl).flux(X1DIR, 0, k, j - 1, i) >= 0.0)
                ? Bflux(bl).flux(X2DIR, 0, k, j, i - 1) - cemf(bl, 2, k, j - 1, i - 1)
                : Bflux(bl).flux(X2DIR, 0, k, j, i) - cemf(bl, 2, k, j - 1, i);
        const Real e3_r2 =
            (Rflux(bl).flux(X1DIR, 0, k, j, i) >= 0.0)
                ? Bflux(bl).flux(X2DIR, 0, k, j, i - 1) - cemf(bl, 2, k, j, i - 1)
                : Bflux(bl).flux(X2DIR, 0, k, j, i) - cemf(bl, 2, k, j, i);
        const Real e3_l1 =
            (Rflux(bl).flux(X2DIR, 0, k, j, i - 1) >= 0.0)
                ? -Bflux(bl).flux(X1DIR, 1, k, j - 1, i) - cemf(bl, 2, k, j - 1, i - 1)
                : -Bflux(bl).flux(X1DIR, 1, k, j, i) - cemf(bl, 2, k, j, i - 1);
        const Real e3_r1 =
            (Rflux(bl).flux(X2DIR, 0, k, j, i) >= 0.0)
                ? -Bflux(bl).flux(X1DIR, 1, k, j - 1, i) - cemf(bl, 2, k, j - 1, i)
                : -Bflux(bl).flux(X1DIR, 1, k, j, i) - cemf(bl, 2, k, j, i);
        emf(bl, TE::E3, 0, k, j, i) += 0.25 * (e3_l2 + e3_r2 + e3_l1 + e3_r1);
      });

  return TaskStatus::complete;
}

TaskStatus BoundaryEMF(MeshData<Real> *md, const bool coarse) {
  auto pmesh = md->GetMeshPointer();
  auto fluid = pmesh->packages.Get("fluid");
  if (!fluid->Param<bool>("mhd")) return TaskStatus::complete;

  static const std::vector<
      std::pair<parthenon::BoundaryFace, std::pair<std::string, IndexDomain>>>
      faces = {
          {parthenon::BoundaryFace::inner_x1, {"ix1_bc", IndexDomain::inner_x1}},
          {parthenon::BoundaryFace::outer_x1, {"ox1_bc", IndexDomain::outer_x1}},
          {parthenon::BoundaryFace::inner_x2, {"ix2_bc", IndexDomain::inner_x2}},
          {parthenon::BoundaryFace::outer_x2, {"ox2_bc", IndexDomain::outer_x2}},
          {parthenon::BoundaryFace::inner_x3, {"ix3_bc", IndexDomain::inner_x3}},
          {parthenon::BoundaryFace::outer_x3, {"ox3_bc", IndexDomain::outer_x3}},
      };

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto rc = md->GetBlockData(b);
    auto pmb = rc->GetBlockPointer();
    for (const auto &face : faces) {
      if (pmb->boundary_flag[face.first] != parthenon::BoundaryFlag::user) continue;
      const std::string &bc_key = face.second.first;
      const IndexDomain domain = face.second.second;
      const std::string bc = fluid->Param<std::string>(bc_key);
      if (bc == "reflect") {
        ZeroBoundaryEMF(rc.get(), domain, coarse);
      } else if (bc == "polar") {
        PARTHENON_REQUIRE(pmesh->ndim < 3 ||
                              (pmb->boundary_flag[parthenon::BoundaryFace::inner_x3] ==
                                   parthenon::BoundaryFlag::periodic &&
                               pmb->boundary_flag[parthenon::BoundaryFace::outer_x3] ==
                                   parthenon::BoundaryFlag::periodic),
                          "Polar EMF averaging requires each block to span all of X3");
        AverageBoundaryEMF(rc.get(), domain, coarse);
      }
    }
  }
  return TaskStatus::complete;
}

TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain) {
  auto pmesh = md->GetMeshPointer();
  if (!pmesh->packages.Get("fluid")->Param<bool>("mhd")) return TaskStatus::complete;
  const int ndim = pmesh->ndim;

  auto emf =
      md->PackVariables(std::vector<std::string>{internal_variables::eemf::name()});
  auto dBdt = mdudt->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  const IndexRange block = IndexRange{0, emf.GetDim(5) - 1};
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

  IndexRange ib1 = md->GetBoundsI(domain, TE::F1);
  IndexRange jb1 = md->GetBoundsJ(domain, TE::F1);
  IndexRange kb1 = md->GetBoundsK(domain, TE::F1);
  pmb0->par_for(
      "b_ct::AddSource_F1", block.s, block.e, kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
      KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        const auto &G = dBdt.GetCoords(bl);
        Real rhs = 0.0;
        if (ndim > 1) {
          rhs +=
              -G.template Volume<TE::E3>(k, j + 1, i) * emf(bl, TE::E3, 0, k, j + 1, i) +
              G.template Volume<TE::E3>(k, j, i) * emf(bl, TE::E3, 0, k, j, i);
        }
        if (ndim > 2) {
          rhs +=
              G.template Volume<TE::E2>(k + 1, j, i) * emf(bl, TE::E2, 0, k + 1, j, i) -
              G.template Volume<TE::E2>(k, j, i) * emf(bl, TE::E2, 0, k, j, i);
        }
        dBdt(bl, TE::F1, 0, k, j, i) = rhs / G.template Volume<TE::F1>(k, j, i);
      });

  IndexRange ib2 = md->GetBoundsI(domain, TE::F2);
  IndexRange jb2 = md->GetBoundsJ(domain, TE::F2);
  IndexRange kb2 = md->GetBoundsK(domain, TE::F2);
  pmb0->par_for(
      "b_ct::AddSource_F2", block.s, block.e, kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
      KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        const auto &G = dBdt.GetCoords(bl);
        Real rhs =
            G.template Volume<TE::E3>(k, j, i + 1) * emf(bl, TE::E3, 0, k, j, i + 1) -
            G.template Volume<TE::E3>(k, j, i) * emf(bl, TE::E3, 0, k, j, i);
        if (ndim > 2) {
          rhs +=
              -G.template Volume<TE::E1>(k + 1, j, i) * emf(bl, TE::E1, 0, k + 1, j, i) +
              G.template Volume<TE::E1>(k, j, i) * emf(bl, TE::E1, 0, k, j, i);
        }
        dBdt(bl, TE::F2, 0, k, j, i) = rhs / G.template Volume<TE::F2>(k, j, i);
      });

  IndexRange ib3 = md->GetBoundsI(domain, TE::F3);
  IndexRange jb3 = md->GetBoundsJ(domain, TE::F3);
  IndexRange kb3 = md->GetBoundsK(domain, TE::F3);
  pmb0->par_for(
      "b_ct::AddSource_F3", block.s, block.e, kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
      KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
        const auto &G = dBdt.GetCoords(bl);
        Real rhs =
            -G.template Volume<TE::E2>(k, j, i + 1) * emf(bl, TE::E2, 0, k, j, i + 1) +
            G.template Volume<TE::E2>(k, j, i) * emf(bl, TE::E2, 0, k, j, i);
        if (ndim > 1) {
          rhs +=
              G.template Volume<TE::E1>(k, j + 1, i) * emf(bl, TE::E1, 0, k, j + 1, i) -
              G.template Volume<TE::E1>(k, j, i) * emf(bl, TE::E1, 0, k, j, i);
        }
        dBdt(bl, TE::F3, 0, k, j, i) = rhs / G.template Volume<TE::F3>(k, j, i);
      });

  return TaskStatus::complete;
}

TaskStatus AverageFaceField(MeshData<Real> *mc0, MeshData<Real> *mbase, const Real beta) {
  auto pmesh = mc0->GetMeshPointer();
  if (!pmesh->packages.Get("fluid")->Param<bool>("mhd")) {
    return TaskStatus::complete;
  }

  auto x = mc0->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  auto base = mbase->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  const IndexRange block = IndexRange{0, x.GetDim(5) - 1};
  auto pmb0 = mc0->GetBlockData(0)->GetBlockPointer();
  for (auto el : {TE::F1, TE::F2, TE::F3}) {
    IndexRange ib = mc0->GetBoundsI(IndexDomain::entire, el);
    IndexRange jb = mc0->GetBoundsJ(IndexDomain::entire, el);
    IndexRange kb = mc0->GetBoundsK(IndexDomain::entire, el);
    pmb0->par_for(
        "b_ct::AverageFaceField", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
          x(bl, el, 0, k, j, i) =
              beta * x(bl, el, 0, k, j, i) + (1.0 - beta) * base(bl, el, 0, k, j, i);
        });
  }
  return TaskStatus::complete;
}

TaskStatus UpdateFaceField(MeshData<Real> *mc0, MeshData<Real> *mdudt, const Real beta_dt,
                           MeshData<Real> *mc1) {
  auto pmesh = mc0->GetMeshPointer();
  if (!pmesh->packages.Get("fluid")->Param<bool>("mhd")) {
    return TaskStatus::complete;
  }

  auto x = mc0->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  auto dudt = mdudt->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  auto z = mc1->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  const IndexRange block = IndexRange{0, x.GetDim(5) - 1};
  auto pmb0 = mc0->GetBlockData(0)->GetBlockPointer();
  for (auto el : {TE::F1, TE::F2, TE::F3}) {
    IndexRange ib = mc0->GetBoundsI(IndexDomain::interior, el);
    IndexRange jb = mc0->GetBoundsJ(IndexDomain::interior, el);
    IndexRange kb = mc0->GetBoundsK(IndexDomain::interior, el);
    pmb0->par_for(
        "b_ct::UpdateFaceField", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i) {
          z(bl, el, 0, k, j, i) =
              x(bl, el, 0, k, j, i) + beta_dt * dudt(bl, el, 0, k, j, i);
        });
  }
  return TaskStatus::complete;
}

Real MaxDivB(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  if (!pmesh->packages.Get("fluid")->Param<bool>("mhd")) {
    return 0.0;
  }

  const int ndim = pmesh->ndim;

  auto Bf = md->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const IndexRange block = IndexRange{0, Bf.GetDim(5) - 1};

  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  Real max_divb = 0.0;
  Kokkos::Max<Real> max_reducer(max_divb);
  pmb0->par_reduce(
      "b_ct::MaxDivB", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int bl, const int k, const int j, const int i,
                    Real &local_result) {
        const auto &G = Bf.GetCoords(bl);
        const Real local_divb = std::abs(face_div(G, Bf(bl), ndim, k, j, i));
        if (local_divb > local_result) {
          local_result = local_divb;
        }
      },
      max_reducer);
  return max_divb;
}

TaskStatus CalcDivB(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd")) {
    return TaskStatus::complete;
  }
  const int ndim = pmb->pmy_mesh->ndim;

  auto Bf = rc->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  auto divb = rc->Get(diagnostic_variables::divb::name()).data;
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "b_ct::CalcDivB", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        divb(k, j, i) = face_div(coords, Bf, ndim, k, j, i);
      });
  return TaskStatus::complete;
}

} // namespace b_ct
