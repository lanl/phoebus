// © 2021-2022. Triad National Security, LLC. All rights reserved.
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

#include <cmath>

#include "fixup.hpp"

#include <defs.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "geometry/tetrads.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/radiation.hpp"
#include <parthenon/package.hpp>

using Microphysics::RadiationType;
using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;

namespace fixup {

template <typename T, class CLOSURE>
TaskStatus RadConservedToPrimitiveFixupImpl(T *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace ir = radmoment_internal;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;

  Mesh *pmesh = rc->GetMeshPointer();
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmesh->packages.Get("eos").get();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_rad = rad_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_rad) return TaskStatus::complete;

  const std::vector<std::string> vars({p::velocity::name(), p::ye::name(), c::ye::name(),
                                       pr::J::name(), pr::H::name(), cr::E::name(),
                                       cr::F::name(), ir::tilPi::name(),
                                       ir::c2pfail::name(), impl::fail::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  auto idx_pvel = imap.GetFlatIdx(p::velocity::name());
  int pye = imap[p::ye::name()].second; // negative if not present
  int cye = imap[c::ye::name()].second;
  auto idx_J = imap.GetFlatIdx(pr::J::name(), false);
  auto idx_H = imap.GetFlatIdx(pr::H::name(), false);
  auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
  auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
  int ifluidfail = imap[impl::fail::name()].first;
  int iradfail = imap[ir::c2pfail::name()].first;
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name(), false);

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total rad nfail: %i\n", nfail_total);
    IndexRange ibi = rc->GetBoundsI(IndexDomain::interior);
    IndexRange jbi = rc->GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = rc->GetBoundsK(IndexDomain::interior);
    nfail_total = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kbi.s, kbi.e, jbi.s, jbi.e, ibi.s, ibi.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total rad interior nfail: %i\n", nfail_total);
  }

  auto geom = Geometry::GetCoordinateSystem(rc);
  Bounds *bounds = fix_pkg->MutableParam<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer()->coords;

  const int nspec = idx_E.DimSize(1);
  const int ndim = pmesh->ndim;

  auto rad_c2p_failure_strategy =
      fix_pkg->Param<FAILURE_STRATEGY>("rad_c2p_failure_strategy");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadConToPrim::Solve fixup", DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real xi_max;
        Real garbage;
        bounds->GetRadiationCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                                     coords.Xc<3>(k, j, i), xi_max, garbage);

        // It is assumed that the fluid is already fixed up
        auto fail = [&](const int k, const int j, const int i) {
          return v(b, iradfail, k, j, i);
        };
        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = fail(k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              fail(k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += fail(k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 fail(k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += fail(k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   fail(k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };

        if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          const Real vel[] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                              v(b, idx_pvel(2), k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vel, gcov);
          Vec con_v({vel[0] / W, vel[1] / W, vel[2] / W});

          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);

          Real num_valid = v(b, iradfail, k, j, i - 1) + v(b, iradfail, k, j, i + 1);
          if (ndim > 1)
            num_valid += v(b, iradfail, k, j - 1, i) + v(b, iradfail, k, j + 1, i);
          if (ndim == 3)
            num_valid += v(b, iradfail, k - 1, j, i) + v(b, iradfail, k + 1, j, i);
          // Can't stencil outside of memory of this meshblock
          // if (num_valid > 0.5 &&
          //    rad_c2p_failure_strategy == FAILURE_STRATEGY::interpolate && i > ib.s &&
          //    i < ib.e - 1 && j > jb.s && j < jb.e - 1 && k > kb.s && k < kb.e - 1) {
          if (num_valid > 0.5 &&
              rad_c2p_failure_strategy == FAILURE_STRATEGY::interpolate) {
            const Real norm = 1.0 / num_valid;
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = fixup(idx_J(ispec), norm);
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = fixup(idx_H(ispec, ii), norm);
              }
            }
          } else {
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = 10. * robust::SMALL();
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = 0.; }
            }
          }

          for (int ispec = 0; ispec < nspec; ispec++) {
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                         v(b, idx_H(ispec, 2), k, j, i)};
            const Real xi =
                std::sqrt(g.contractCov3Vectors(cov_H, cov_H) -
                          std::pow(g.contractConCov3Vectors(con_v, cov_H), 2));
            if (xi > xi_max) {
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
            }
          }

          CLOSURE c(con_v, &g);
          for (int ispec = 0; ispec < nspec; ispec++) {
            Real E;
            Vec cov_F;
            Tens2 con_tilPi;
            Real J = v(b, idx_J(ispec), k, j, i);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i) * J,
                         v(b, idx_H(ispec, 1), k, j, i) * J,
                         v(b, idx_H(ispec, 2), k, j, i) * J};
            if (iTilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                con_tilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetConTilPiFromPrim(J, cov_H, &con_tilPi);
            }
            c.Prim2Con(J, cov_H, con_tilPi, &E, &cov_F);

            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * cov_F(ii); }
          }
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus RadConservedToPrimitiveFixup(T *rc) {
  Mesh *pmesh = rc->GetMeshPointer();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  } else {
    return TaskStatus::complete;
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return RadConservedToPrimitiveFixupImpl<T, radiation::ClosureM1<settings>>(rc);
  } else if (method == "moment_eddington") {
    return RadConservedToPrimitiveFixupImpl<T, radiation::ClosureEdd<settings>>(rc);
  } else if (method == "mocmc") {
    return RadConservedToPrimitiveFixupImpl<T, radiation::ClosureMOCMC<settings>>(rc);
  }
  return TaskStatus::complete;
}

template TaskStatus
RadConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
template <>
TaskStatus RadConservedToPrimitiveFixup<MeshData<Real>>(MeshData<Real> *md) {
  for (const auto &mbd : md->GetAllBlockData()) {
    RadConservedToPrimitiveFixup(mbd.get());
  }
  return TaskStatus::complete;
}

} // namespace fixup
