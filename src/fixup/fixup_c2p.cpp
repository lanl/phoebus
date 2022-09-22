// © 2022. Triad National Security, LLC. All rights reserved.
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

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc, T *rc0) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  // IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  // IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  // IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *fluid_pkg = pmb->packages.Get("fluid").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();

  const std::vector<std::string> vars({p::density, c::density, p::velocity, c::momentum,
                                       p::energy, c::energy, p::bfield, p::ye, c::ye,
                                       p::pressure, p::temperature, p::gamma1,
                                       impl::cell_signal_speed, impl::fail});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  PackIndexMap imap0;
  auto v0 = rc0->PackVariables(vars, imap0);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  int ifail = imap[impl::fail].first;

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_fluid = fluid_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_fluid) return TaskStatus::complete;

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total nfail: %i\n", nfail_total);
    IndexRange ibi = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jbi = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    nfail_total = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetDim(5) - 1, kbi.s, kbi.e, jbi.s, jbi.e, ibi.s, ibi.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total interior nfail: %i\n", nfail_total);
  }

  const int ndim = pmb->pmy_mesh->ndim;

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  // TODO(BRR) make this less ugly (or do this at all?)
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
          v(b, ifail, k, j, i) = con2prim_robust::FailFlags::fail;
        }
      });

  auto c2p_failure_strategy = fix_pkg->Param<FAILURE_STRATEGY>("c2p_failure_strategy");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real eos_lambda[2]; // use last temp as initial guess
        eos_lambda[0] = 0.5;
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i));

        Real gamma_max, e_max;
        bounds.GetCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
                           gamma_max, e_max);

        // Need to account for not stenciling outside of ghost zones
        // bool is_outer_ghost_layer =
        //    (i == ib.s || i == ib.e || j == jb.s || j == jb.e || k == kb.s || k ==
        //    kb.e);

        auto fixup0 = [&](const int iv) {
          v(b, iv, k, j, i) = v0(b, iv, k, j, i - 1) + v0(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v0(b, iv, k, j - 1, i) + v0(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v0(b, iv, k - 1, j, i) + v0(b, iv, k + 1, j, i);
            }
          }
          return v(b, iv, k, j, i) / (2 * ndim);
        };
        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = v(b, ifail, k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              v(b, ifail, k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += v(b, ifail, k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 v(b, ifail, k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += v(b, ifail, k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   v(b, ifail, k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };
        if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
          Real num_valid = 0;
          num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
          if (ndim > 1) num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
          if (ndim == 3) num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);
          //}
          if (c2p_failure_strategy == FAILURE_STRATEGY::interpolate_previous) {
            v(b, prho, k, j, i) = fixup0(prho);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b, pv, k, j, i) = fixup0(pv);
            }
            v(b, peng, k, j, i) = fixup0(peng);

            if (pye > 0) v(b, pye, k, j, i) = fixup0(pye);
          } else {
            if (num_valid > 0.5 &&
                c2p_failure_strategy == FAILURE_STRATEGY::interpolate) {
              //            printf("[%i %i %i] num_valid: %e\n", k, j, i, num_valid);
              const Real norm = 1.0 / num_valid;
              v(b, prho, k, j, i) = fixup(prho, norm);
              for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
                v(b, pv, k, j, i) = fixup(pv, norm);
              }
              v(b, peng, k, j, i) = fixup(peng, norm);

              if (pye > 0) v(b, pye, k, j, i) = fixup(pye, norm);
            } else {
              //            printf("[%i %i %i] no valid: %e\n", k, j, i, num_valid);
              // printf("[%i %i %i] no valid c2p neighbors!\n", k, j, i);
              // No valid neighbors; set fluid mass/energy to near-zero and set primitive
              // velocities to zero

              // v(b, prho, k, j, i) = 1.e-20;
              // v(b, peng, k, j, i) = 1.e-20;
              v(b, prho, k, j, i) = 1.e-100;
              v(b, peng, k, j, i) = 1.e-100;

              // Safe value for ye
              if (pye > 0) {
                v(b, pye, k, j, i) = 0.5;
              }

              // Zero primitive velocities
              SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = 0.; }
            }
          }

          const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);

          // Clamp velocity now (for rad inversion)
          Real vpcon[3] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                           v(b, pvel_lo + 2, k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vpcon, gcov);
          if (W > gamma_max) {
            const Real rescale = std::sqrt(gamma_max * gamma_max - 1.) / (W * W - 1.);
            SPACELOOP(ii) { vpcon[ii] *= rescale; }
            SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = vpcon[ii]; }
          }

          // Update dependent primitives
          if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
          v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
              v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
              eos_lambda);
          v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
              v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
          v(b, gm1, k, j, i) =
              ratio(eos.BulkModulusFromDensityTemperature(v(b, prho, k, j, i),
                                                          v(b, tmp, k, j, i), eos_lambda),
                    v(b, prs, k, j, i));

          // Update conserved variables

          Real S[3];
          Real bcons[3];
          Real bp[3] = {0.0, 0.0, 0.0};
          if (pb_hi > 0) {
            bp[0] = v(b, pb_lo, k, j, i);
            bp[1] = v(b, pb_lo + 1, k, j, i);
            bp[2] = v(b, pb_hi, k, j, i);
          }
          Real ye_cons;
          Real ye_prim = 0.5;
          if (pye > 0) {
            ye_prim = v(b, pye, k, j, i);
          }
          Real sig[3];
          prim2con::p2c(v(b, prho, k, j, i), vpcon, bp, v(b, peng, k, j, i), ye_prim,
                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gcon, beta, alpha,
                        gdet, v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i), ye_cons,
                        sig);
          v(b, cmom_lo, k, j, i) = S[0];
          v(b, cmom_lo + 1, k, j, i) = S[1];
          v(b, cmom_hi, k, j, i) = S[2];
          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
          for (int m = slo; m <= shi; m++) {
            v(b, m, k, j, i) = sig[m - slo];
          }

          // TODO(BRR)
          // if not iradfail here, call rad c2p with updated velocity and set iradfail to result
        }
      });

  // TODO(BRR) This is inefficient!
  // ONLY FOR MHD!
  // ApplyFloors(rc);
  // ApplyFluidFloors(rc);

  // Need to update radiation primitives
  // TODO(BRR) This is very inefficient!
  // radiation::MomentCon2Prim(rc);

  // ApplyFloors(rc);

  return TaskStatus::complete;
}

template TaskStatus
ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                               MeshBlockData<Real> *rc0);

} // namespace fixup
