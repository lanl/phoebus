// © 2022-2023. Triad National Security, LLC. All rights reserved.
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
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/radiation.hpp"
#include <parthenon/package.hpp>

using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;

namespace fixup {

template <typename T, class CLOSURE>
TaskStatus ConservedToPrimitiveFixupImpl(T *rc) {
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
  StateDescriptor *fluid_pkg = pmesh->packages.Get("fluid").get();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();
  StateDescriptor *eos_pkg = pmesh->packages.Get("eos").get();

  const std::vector<std::string> vars({p::density::name(),
                                       c::density::name(),
                                       p::velocity::name(),
                                       c::momentum::name(),
                                       p::energy::name(),
                                       c::energy::name(),
                                       p::bfield::name(),
                                       p::ye::name(),
                                       c::ye::name(),
                                       p::pressure::name(),
                                       p::temperature::name(),
                                       p::gamma1::name(),
                                       impl::cell_signal_speed::name(),
                                       impl::fail::name(),
                                       ir::c2pfail::name(),
                                       ir::tilPi::name(),
                                       pr::J::name(),
                                       pr::H::name(),
                                       cr::E::name(),
                                       cr::F::name(),
                                       ir::xi::name(),
                                       ir::phi::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density::name()].first;
  const int crho = imap[c::density::name()].first;
  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int cmom_lo = imap[c::momentum::name()].first;
  const int cmom_hi = imap[c::momentum::name()].second;
  const int peng = imap[p::energy::name()].first;
  const int ceng = imap[c::energy::name()].first;
  const int prs = imap[p::pressure::name()].first;
  const int tmp = imap[p::temperature::name()].first;
  const int gm1 = imap[p::gamma1::name()].first;
  const int slo = imap[impl::cell_signal_speed::name()].first;
  const int shi = imap[impl::cell_signal_speed::name()].second;
  const int pb_lo = imap[p::bfield::name()].first;
  const int pb_hi = imap[p::bfield::name()].second;
  int pye = imap[p::ye::name()].second; // negative if not present
  int cye = imap[c::ye::name()].second;

  int ifail = imap[impl::fail::name()].first;
  int irfail = imap[ir::c2pfail::name()].first;

  auto idx_E = imap.GetFlatIdx(cr::E::name(), false);
  auto idx_F = imap.GetFlatIdx(cr::F::name(), false);
  auto idx_J = imap.GetFlatIdx(pr::J::name(), false);
  auto idx_H = imap.GetFlatIdx(pr::H::name(), false);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name(), false);
  auto iXi = imap.GetFlatIdx(ir::xi::name(), false);
  auto iPhi = imap.GetFlatIdx(ir::phi::name(), false);

  const bool rad_active = rad_pkg->Param<bool>("active");
  const int num_species = rad_active ? rad_pkg->Param<int>("num_species") : 0;

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
    IndexRange ibi = rc->GetBoundsI(IndexDomain::interior);
    IndexRange jbi = rc->GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = rc->GetBoundsK(IndexDomain::interior);
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

  const int ndim = pmesh->ndim;

  auto eos = eos_pkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  Bounds *pbounds = fix_pkg->MutableParam<Bounds>("bounds");
  Bounds bounds = *pbounds;

  Coordinates_t coords = rc->GetParentPointer()->coords;

  auto fluid_c2p_failure_strategy =
      fix_pkg->Param<FAILURE_STRATEGY>("fluid_c2p_failure_strategy");
  const auto c2p_failure_force_fixup_both =
      fix_pkg->Param<bool>("c2p_failure_force_fixup_both");

  PARTHENON_REQUIRE(!c2p_failure_force_fixup_both,
                    "As currently implemented this is a race condition!");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real eos_lambda[2]; // use last temp as initial guess
        eos_lambda[0] = 0.5;
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i));

        Real gamma_max, e_max;
        bounds.GetCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                           coords.Xc<3>(k, j, i), gamma_max, e_max);

        if (c2p_failure_force_fixup_both && rad_active) {
          if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail ||
              v(b, irfail, k, j, i) == radiation::FailFlags::fail) {
            v(b, ifail, k, j, i) = con2prim_robust::FailFlags::fail;
            v(b, irfail, k, j, i) = radiation::FailFlags::fail;
          }
        }

        // Need to account for not stenciling outside of ghost zones
        // bool is_outer_ghost_layer =
        //    (i == ib.s || i == ib.e || j == jb.s || j == jb.e || k == kb.s || k ==
        //    kb.e);

        auto fail = [&](const int k, const int j, const int i) {
          if (c2p_failure_force_fixup_both) {
            return v(b, ifail, k, j, i) * v(b, irfail, k, j, i);
          } else {
            return v(b, ifail, k, j, i);
          }
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
        // When using IndexDomain::entire, can't stencil from e.g. i = 0 because 0 - 1 =
        // -1 is a segfault
        if (v(b, ifail, k, j, i) == con2prim_robust::FailFlags::fail) {
          Real num_valid = 0;
          num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
          if (ndim > 1) num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
          if (ndim == 3) num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);

          // if (num_valid > 0.5 &&
          //    fluid_c2p_failure_strategy == FAILURE_STRATEGY::interpolate && i > ib.s &&
          //    i < ib.e - 1 && j > jb.s && j < jb.e - 1 && k > kb.s && k < kb.e - 1) {
          if (num_valid > 0.5 &&
              fluid_c2p_failure_strategy == FAILURE_STRATEGY::interpolate) {
            const Real norm = 1.0 / num_valid;
            v(b, prho, k, j, i) = fixup(prho, norm);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b, pv, k, j, i) = fixup(pv, norm);
            }
            v(b, peng, k, j, i) = fixup(peng, norm);

            if (pye > 0) v(b, pye, k, j, i) = fixup(pye, norm);

            v(b, prho, k, j, i) =
                std::max<Real>(v(b, prho, k, j, i), 100. * robust::SMALL());
            v(b, peng, k, j, i) =
                std::max<Real>(v(b, peng, k, j, i), 100. * robust::SMALL());
          } else {
            // No valid neighbors; set fluid mass/energy to near-zero and set primitive
            // velocities to zero

            v(b, prho, k, j, i) = 100. * robust::SMALL();
            v(b, peng, k, j, i) = 100. * robust::SMALL();

            // Safe value for ye
            if (pye > 0) {
              v(b, pye, k, j, i) = 0.5;
            }

            // Zero primitive velocities
            SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = 0.; }
          }

          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
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
          Real W = phoebus::GetLorentzFactor(vpcon, gcov);
          if (W > gamma_max) {
            const Real rescale = std::sqrt(gamma_max * gamma_max - 1.) / (W * W - 1.);
            SPACELOOP(ii) { vpcon[ii] *= rescale; }
            SPACELOOP(ii) { v(b, pvel_lo + ii, k, j, i) = vpcon[ii]; }
            W = gamma_max;
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
                        sdetgam, v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i),
                        ye_cons, sig);
          v(b, cmom_lo, k, j, i) = S[0];
          v(b, cmom_lo + 1, k, j, i) = S[1];
          v(b, cmom_hi, k, j, i) = S[2];
          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
          for (int m = slo; m <= shi; m++) {
            v(b, m, k, j, i) = sig[m - slo];
          }

          if (irfail >= 0) {
            // If rad c2p failed, we'll fix that up subsequently
            if (v(b, irfail, k, j, i) == radiation::FailFlags::success) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j,
                                                      i);
                Vec con_v{vpcon[0] / W, vpcon[1] / W, vpcon[2] / W};
                CLOSURE c(con_v, &g);

                Real E = v(b, idx_E(ispec), k, j, i) / sdetgam;
                Vec cov_F;
                SPACELOOP(ii) { cov_F(ii) = v(b, idx_F(ispec, ii), k, j, i) / sdetgam; }
                Tens2 con_tilPi;
                Real J;
                Vec cov_H;
                if (iTilPi.IsValid()) {
                  SPACELOOP2(ii, jj) {
                    con_tilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
                  }
                } else {
                  Real xi = 0.;
                  Real phi = M_PI;
                  // TODO(BRR) STORE_GUESS
                  c.GetConTilPiFromCon(E, cov_F, xi, phi, &con_tilPi);
                }

                c.Con2Prim(E, cov_F, con_tilPi, &J, &cov_H);

                v(b, idx_J(ispec), k, j, i) = J;
                SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = cov_H(ii) / J; }

                // Floors and ceilings will be applied subsequently by ApplyFloors task
              }
            }
          }
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc) {
  Mesh *pmesh = rc->GetMeshPointer();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureM1<settings>>(rc);
  } else if (method == "moment_eddington") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureEdd<settings>>(rc);
  } else if (method == "mocmc") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureMOCMC<settings>>(rc);
  } else {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureEdd<settings>>(rc);
  }
  return TaskStatus::complete;
}

template TaskStatus
ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
// template TaskStatus
template <>
TaskStatus ConservedToPrimitiveFixup<MeshData<Real>>(MeshData<Real> *md) {
  for (const auto &mbd : md->GetAllBlockData()) {
    ConservedToPrimitiveFixup(mbd.get());
  }
  return TaskStatus::complete;
}

} // namespace fixup
