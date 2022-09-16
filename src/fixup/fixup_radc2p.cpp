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

#include <bvals/bvals_interfaces.hpp>
#include <defs.hpp>
#include <singularity-eos/eos/eos.hpp>

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

using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;
using singularity::RadiationType;
using singularity::neutrinos::Opacity;

namespace fixup {

template <typename T, class CLOSURE>
TaskStatus RadConservedToPrimitiveFixupImpl(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace ir = radmoment_internal;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;

  auto *pmb = rc->GetParentPointer().get();
  // IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  // IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  // IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_rad = rad_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_rad) return TaskStatus::complete;

  const std::vector<std::string> vars({p::density,
                                       c::density,
                                       p::velocity,
                                       c::momentum,
                                       p::energy,
                                       c::energy,
                                       p::bfield,
                                       p::ye,
                                       c::ye,
                                       p::pressure,
                                       p::temperature,
                                       p::gamma1,
                                       pr::J,
                                       pr::H,
                                       cr::E,
                                       cr::F,
                                       impl::cell_signal_speed,
                                       ir::tilPi,
                                       ir::c2pfail,
                                       impl::fail});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  auto idx_pvel = imap.GetFlatIdx(p::velocity);
  auto idx_cmom = imap.GetFlatIdx(c::momentum);
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  auto idx_J = imap.GetFlatIdx(pr::J, false);
  auto idx_H = imap.GetFlatIdx(pr::H, false);
  auto idx_E = imap.GetFlatIdx(cr::E, false);
  auto idx_F = imap.GetFlatIdx(cr::F, false);
  int ifluidfail = imap[impl::fail].first;
  int iradfail = imap[ir::c2pfail].first;
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);

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
    IndexRange ibi = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jbi = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
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

  // TODO(BRR) make this less ugly? Do this at all?
  // TODO(BRR) if (use_ghost_for_stencil_fixup) {} ?
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "C2P fail initialization", DevExecSpace(), 0, v.GetDim(5) - 1,
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ib.s || i > ib.e || j < jb.s || j > jb.e || k < kb.s || k > kb.e) {
          // Do not use ghost zones as data for averaging
          // TODO(BRR) need to allow ghost zones from neighboring blocks
          v(b, ifluidfail, k, j, i) = con2prim_robust::FailFlags::fail;
          v(b, iradfail, k, j, i) = radiation::FailFlags::fail;
        }
      });

  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  const int nspec = idx_E.DimSize(1);
  const int ndim = pmb->pmy_mesh->ndim;

  auto c2p_failure_strategy = fix_pkg->Param<FAILURE_STRATEGY>("c2p_failure_strategy");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadConToPrim::Solve fixup", DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // If fluid fail but rad success, recalculate rad c2p and set iradfail with
        // result, then still process if (fluid fail && rad fail) check.
        // Note that it is assumed that the fluid is already fixed up
        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = v(b, iradfail, k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              v(b, iradfail, k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v(b, iradfail, k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 v(b, iradfail, k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v(b, iradfail, k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   v(b, iradfail, k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };

        // if (v(b, ifluidfail, k, j, i) == con2prim_robust::FailFlags::fail ||
        //    v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {
        if (v(b, iradfail, k, j, i) == radiation::FailFlags::fail) {

          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          const Real vel[] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                              v(b, idx_pvel(2), k, j, i)};

          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);

          Real num_valid = v(b, iradfail, k, j, i - 1) + v(b, iradfail, k, j, i + 1);
          if (ndim > 1)
            num_valid += v(b, iradfail, k, j - 1, i) + v(b, iradfail, k, j + 1, i);
          if (ndim == 3)
            num_valid += v(b, iradfail, k - 1, j, i) + v(b, iradfail, k + 1, j, i);
          if (num_valid > 0.5 && c2p_failure_strategy == FAILURE_STRATEGY::interpolate) {
            const Real norm = 1.0 / num_valid;
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = fixup(idx_J(ispec), norm);
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = fixup(idx_H(ispec, ii), norm);
              }
            }
          } else {
            for (int ispec = 0; ispec < nspec; ispec++) {
              v(b, idx_J(ispec), k, j, i) = 1.e-100;
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = 0.; }
            }
          }

          Real xi_max = 0.99;
          for (int ispec = 0; ispec < nspec; ispec++) {
            //       v(b, idx_J(ispec), k, j, i) =
            //           std::max<Real>(v(b, idx_J(ispec), k, j, i), 1.e-10);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                         v(b, idx_H(ispec, 2), k, j, i)};
            Real xi = std::sqrt(g.contractCov3Vectors(cov_H, cov_H));
            if (xi > xi_max) {
              SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
            }
          }

          const Real W = phoebus::GetLorentzFactor(vel, gcov);
          Vec con_v({vel[0] / W, vel[1] / W, vel[2] / W});
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
              c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);
            }
            c.Prim2Con(J, cov_H, con_tilPi, &E, &cov_F);

            //        Real xi = std::sqrt(g.contractCov3Vectors(cov_H, cov_H)) / J;
            //            printf("Rad fixup [%i %i %i] J = %e cov_H = %e %e %e
            //            (xi = %e) E
            //            = %e cov_F = %e %e %e\n",
            //
            //          k,j,i,J,cov_H(0),cov_H(1),cov_H(2),xi,E,cov_F(0),cov_F(1),cov_F(2));

            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * cov_F(ii); }
          }
        }
      });

  // TODO(BRR) This is inefficient!
  ApplyFloors(rc);

  return TaskStatus::complete;
}

template <typename T>
TaskStatus RadConservedToPrimitiveFixup(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad_pkg = pm->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureM1<Vec, Tens2, settings>>(
        rc);
  } else if (method == "moment_eddington") {
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureEdd<Vec, Tens2, settings>>(
        rc);
  } else if (method == "mocmc") {
    return RadConservedToPrimitiveFixupImpl<
        T, radiation::ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return RadConservedToPrimitiveFixupImpl<T,
                                            radiation::ClosureEdd<Vec, Tens2, settings>>(
        rc);
  }
  return TaskStatus::fail;
}

template TaskStatus
RadConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace fixup
