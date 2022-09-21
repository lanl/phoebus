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
//#include "fixup_applier.hpp"

#include <bvals/bvals_interfaces.hpp>
#include <defs.hpp>
#include <singularity-eos/eos/eos.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/fluid.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
//#include "geometry/tetrads.hpp"
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

// If radiation source terms fail (probably due to a rootfind failing to converge) average
// the fluid and radiation primitive variables over good neighbors. Or if there are no
// good neighbors, set everything to the floors. Then call p2c on both fluid and
// radiation.
template <typename T, class CLOSURE>
TaskStatus SourceFixupImpl(T *rc, T *rc0) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;
  namespace ir = radmoment_internal;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();
  if (!rad_pkg->Param<bool>("active")) {
    return TaskStatus::complete;
  }
  bool enable_source_fixup = fix_pkg->Param<bool>("enable_source_fixup");
  if (!enable_source_fixup) {
    return TaskStatus::complete;
  }

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  const std::vector<std::string> vars(
      {p::density, c::density, p::velocity, c::momentum, p::energy, c::energy, p::bfield,
       p::ye, c::ye, p::pressure, p::temperature, p::gamma1, pr::J, pr::H, cr::E, cr::F,
       impl::cell_signal_speed, ir::srcfail, ir::tilPi});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  PackIndexMap imap0;
  auto v0 = rc0->PackVariables(vars, imap0);

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
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  int ifail = imap[ir::srcfail].first;
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);
  // TODO(BRR) get iTilPi for MOCMC

  bool report_source_fails = fix_pkg->Param<bool>("report_source_fails");
  if (report_source_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Source fixup failures", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b, ifail, k, j, i) == radiation::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total source nfail: %i\n", nfail_total);
  }

  const int ndim = pmb->pmy_mesh->ndim;

  auto geom = Geometry::GetCoordinateSystem(rc);
  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  auto num_species = rad_pkg->Param<int>("num_species");

  // TODO(BRR) make this less ugly
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Source fail initialization", DevExecSpace(), 0,
      v.GetDim(5) - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ib.s || i > ib.e || j < jb.s || j > jb.e || k < kb.s || k > kb.e) {
          // Do not use ghost zones as data for averaging
          // TODO(BRR) need to allow ghost zones from neighboring blocks
          v(b, ifail, k, j, i) = radiation::FailFlags::fail;
        }
      });

  auto src_failure_strategy = fix_pkg->Param<FAILURE_STRATEGY>("src_failure_strategy");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Source fixup", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        double gamma_max, e_max;
        bounds.GetCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
                           gamma_max, e_max);
        Real xi_max;
        bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                                    coords.x3v(k, j, i), xi_max);

        double eos_lambda[2]; // used for stellarcollapse eos and
                              // other EOS's that require root
                              // finding.
        eos_lambda[1] = std::log10(v(b, tmp, k, j, i)); // use last temp as initial guess

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

        if (v(b, ifail, k, j, i) == radiation::FailFlags::fail) {
          if (src_failure_strategy == FAILURE_STRATEGY::interpolate_previous) {
            v(b, prho, k, j, i) = fixup0(prho);
            v(b, peng, k, j, i) = fixup0(peng);
            SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = fixup0(idx_pvel(ii)); }
            if (pye > 0) {
              v(b, pye, k, j, i) = fixup0(pye);
            }
            if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
            v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                eos_lambda);
            v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
                v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
            v(b, gm1, k, j, i) =
                ratio(eos.BulkModulusFromDensityTemperature(
                          v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda),
                      v(b, prs, k, j, i));

            for (int ispec = 0; ispec < num_species; ispec++) {
              v(b, idx_J(ispec), k, j, i) = fixup0(idx_J(ispec));
              SPACELOOP(ii) {
                v(b, idx_H(ispec, ii), k, j, i) = fixup0(idx_H(ispec, ii));
              }
            }
          } else {

            Real num_valid = v(b, ifail, k, j, i - 1) + v(b, ifail, k, j, i + 1);
            if (ndim > 1)
              num_valid += v(b, ifail, k, j - 1, i) + v(b, ifail, k, j + 1, i);
            if (ndim == 3)
              num_valid += v(b, ifail, k - 1, j, i) + v(b, ifail, k + 1, j, i);

            if (num_valid > 0.5 &&
                src_failure_strategy == FAILURE_STRATEGY::interpolate) {
              const Real norm = 1.0 / num_valid;

              v(b, prho, k, j, i) = fixup(prho, norm);
              v(b, peng, k, j, i) = fixup(peng, norm);
              SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = fixup(idx_pvel(ii), norm); }
              if (pye > 0) {
                v(b, pye, k, j, i) = fixup(pye, norm);
              }
              if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
              v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                  v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                  eos_lambda);
              v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
                  v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
              v(b, gm1, k, j, i) =
                  ratio(eos.BulkModulusFromDensityTemperature(
                            v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda),
                        v(b, prs, k, j, i));

              for (int ispec = 0; ispec < num_species; ispec++) {
                v(b, idx_J(ispec), k, j, i) = fixup(idx_J(ispec), norm);
                SPACELOOP(ii) {
                  v(b, idx_H(ispec, ii), k, j, i) = fixup(idx_H(ispec, ii), norm);
                }
              }
            } else {
              // No valid neighbors; set to floors with zero spatial velocity
              //            printf("No valid source neighbors! %i %i %i\n", k, j, i);

              // double rho_floor, sie_floor;
              // bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i),
              //                 coords.x3v(k, j, i), rho_floor, sie_floor);
              // v(b, prho, k, j, i) = rho_floor;
              // v(b, peng, k, j, i) = rho_floor * sie_floor;
              v(b, prho, k, j, i) = 1.e-100;
              v(b, peng, k, j, i) = 1.e-100;

              // Zero primitive velocities
              SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = 0.; }

              // Auxiliary primitives
              // Safe value for ye
              if (pye > 0) {
                v(b, pye, k, j, i) = 0.5;
              }
              if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
              v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                  v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j, i)),
                  eos_lambda);
              v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
                  v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
              v(b, gm1, k, j, i) =
                  ratio(eos.BulkModulusFromDensityTemperature(
                            v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda),
                        v(b, prs, k, j, i));

              // Real ucon[4] = {0};
              Real vpcon[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                               v(b, idx_pvel(2), k, j, i)};
              // GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);
              // Geometry::Tetrads tetrads(ucon, gcov);

              for (int ispec = 0; ispec < num_species; ispec++) {
                // v(b, idx_J(ispec), k, j, i) = 1.e-10;
                v(b, idx_J(ispec), k, j, i) = 1.e-100;

                // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = Hcov[ii] / v(b,
                // idx_J(ispec), k, j, i);
                SPACELOOP(ii) {
                  v(b, idx_H(ispec, ii), k, j, i) = 0.;
                  // printf("Fail [%i %i %i] H[%i] = %e\n", k, j, i, ii,
                  //       v(b, idx_H(ispec, ii), k, j, i));
                }

                // TODO(BRR) is this wrong? Want zero flux in fluid frame, this isn't it
                // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = 0.; }
                // SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) = Mcon_coord[ii + 1] /
                // v(b, idx_J(ispec), k, j, i);
                //  printf("Fail [%i %i %i] H[%i] = %e\n", k,j,i,ii,v(b, idx_H(ispec, ii),
                //  k, j, i));
                //}
              }
            }
          }

          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);

          //          // Apply floors (redundant with case of no valid neighbors)
          //          double rho_floor, sie_floor;
          //          bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i),
          //          coords.x3v(k, j, i),
          //                           rho_floor, sie_floor);
          //          v(b, prho, k, j, i) = std::max<Real>(v(b, prho, k, j, i),
          //          rho_floor); v(b, peng, k, j, i) =
          //              std::max<Real>(v(b, peng, k, j, i), rho_floor * sie_floor);
          //
          //          if (pye > 0) eos_lambda[0] = v(b, pye, k, j, i);
          //          v(b, tmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
          //              v(b, prho, k, j, i), ratio(v(b, peng, k, j, i), v(b, prho, k, j,
          //              i)), eos_lambda);
          //          v(b, prs, k, j, i) = eos.PressureFromDensityTemperature(
          //              v(b, prho, k, j, i), v(b, tmp, k, j, i), eos_lambda);
          //          v(b, gm1, k, j, i) =
          //              ratio(eos.BulkModulusFromDensityTemperature(v(b, prho, k, j, i),
          //                                                          v(b, tmp, k, j, i),
          //                                                          eos_lambda),
          //                    v(b, prs, k, j, i));
          //

          // Clamp velocity now (for rad inversion)
          Real vpcon[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k, j, i),
                           v(b, idx_pvel(2), k, j, i)};
          Real W = phoebus::GetLorentzFactor(vpcon, gcov);
          if (W > gamma_max) {
            const Real rescale = std::sqrt(gamma_max * gamma_max - 1.) / (W * W - 1.);
            SPACELOOP(ii) { vpcon[ii] *= rescale; }
            SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) = vpcon[ii]; }
            W = gamma_max;
          }
          Vec con_v({vpcon[0] / W, vpcon[1] / W, vpcon[2] / W});

          typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);
          //
          //          Real gamma_max, e_max;
          //          bounds.GetCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
          //                             coords.x3v(k, j, i), gamma_max, e_max);
          //
          //          // Clamp fluid velocity
          //          Real con_vp[3] = {v(b, idx_pvel(0), k, j, i), v(b, idx_pvel(1), k,
          //          j, i),
          //                            v(b, idx_pvel(2), k, j, i)};
          //          Real W = phoebus::GetLorentzFactor(con_vp, gcov);
          //          if (W > gamma_max) {
          //            const Real rescale = std::sqrt((gamma_max * gamma_max - 1.) / (W *
          //            W
          //            - 1.)); SPACELOOP(ii) { v(b, idx_pvel(ii), k, j, i) *= rescale; }
          //          }
          //          const Real r = std::exp(coords.x1v(k, j, i));
          //          Real xi_max;
          //          bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j,
          //          i),
          //                                      coords.x3v(k, j, i), xi_max);
          //          const Real Jmin = 1.e-10;
          for (int ispec = 0; ispec < num_species; ispec++) {
            //            v(b, idx_J(ispec), k, j, i) =
            //                std::max<Real>(v(b, idx_J(ispec), k, j, i), Jmin);
            Vec cov_H = {v(b, idx_H(ispec, 0), k, j, i), v(b, idx_H(ispec, 1), k, j, i),
                         v(b, idx_H(ispec, 2), k, j, i)};
            //Vec con_H;
            //g.raise3Vector(cov_H, &con_H);
            //Real xi = 0.;
            //SPACELOOP(ii) { xi += cov_H(ii) * con_H(ii); }
            //xi = std::sqrt(xi);
            //if (xi > xi_max) {
            //  SPACELOOP(ii) { v(b, idx_H(ispec, ii), k, j, i) *= xi_max / xi; }
            //}
              const Real xi = std::sqrt(g.contractCov3Vectors(cov_H,cov_H) - std::pow(g.contractConCov3Vectors(con_v,cov_H),2));
          }

          //          // Update MHD conserved variables
          Real S[3];
          Real bcons[3];
          Real bp[3] = {0.0, 0.0, 0.0};
          if (pb_hi > 0) {
            bp[0] = v(b, pb_lo, k, j, i);
            bp[1] = v(b, pb_lo + 1, k, j, i);
            bp[2] = v(b, pb_hi, k, j, i);
          }
          Real ye_cons;
          Real ye_prim = 0.0;
          if (pye > 0) {
            ye_prim = v(b, pye, k, j, i);
          }
          Real sig[3];
          prim2con::p2c(v(b, prho, k, j, i), vpcon, bp, v(b, peng, k, j, i), ye_prim,
                        v(b, prs, k, j, i), v(b, gm1, k, j, i), gcov, gcon, beta, alpha,
                        sdetgam, v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i),
                        ye_cons, sig);
          v(b, idx_cmom(0), k, j, i) = S[0];
          v(b, idx_cmom(1), k, j, i) = S[1];
          v(b, idx_cmom(2), k, j, i) = S[2];
          if (pye > 0) v(b, cye, k, j, i) = ye_cons;
          for (int m = slo; m <= shi; m++) {
            v(b, m, k, j, i) = sig[m - slo];
          }

          // Update radiation conserved variables
          CLOSURE c(con_v, &g);
          for (int ispec = 0; ispec < num_species; ispec++) {
            Real E;
            Vec cov_F;
            Tens2 con_TilPi = {0};
            Real J = v(b, idx_J(ispec), k, j, i);
            Vec cov_H = {J * v(b, idx_H(ispec, 0), k, j, i),
                         J * v(b, idx_H(ispec, 1), k, j, i),
                         J * v(b, idx_H(ispec, 2), k, j, i)};
            if (iTilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                con_TilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetCovTilPiFromPrim(J, cov_H, &con_TilPi);
            }
            c.Prim2Con(J, cov_H, con_TilPi, &E, &cov_F);
            v(b, idx_E(ispec), k, j, i) = sdetgam * E;
            SPACELOOP(ii) { v(b, idx_F(ispec, ii), k, j, i) = sdetgam * cov_F(ii); }
          }
        }
      });

  // TODO(BRR) This is inefficient!
  //  ApplyFloors(rc);

  return TaskStatus::complete;
}

template <typename T>
TaskStatus SourceFixup(T *rc, T *rc0) {
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
    return SourceFixupImpl<T, radiation::ClosureM1<settings>>(rc, rc0);
  } else if (method == "moment_eddington") {
    return SourceFixupImpl<T, radiation::ClosureEdd<settings>>(rc, rc0);
  } else if (method == "mocmc") {
    return SourceFixupImpl<T, radiation::ClosureMOCMC<settings>>(rc, rc0);
  } else {
    // TODO(BRR) default to Eddington closure, check that rad floors are unused for
    // Monte Carlo/cooling function
    PARTHENON_REQUIRE(!enable_rad_floors,
                      "Rad floors not supported with cooling function/Monte Carlo!");
    return SourceFixupImpl<T, radiation::ClosureEdd<settings>>(rc, rc0);
  }
  return TaskStatus::fail;
}

template TaskStatus SourceFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                                     MeshBlockData<Real> *rc0);

} // namespace fixup
