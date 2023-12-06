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

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/root_find.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

#include "fixup/fixup.hpp"
#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"

namespace radiation {

using fixup::Bounds;
using Microphysics::Opacities;
using Microphysics::EOS::EOS;

template <typename CLOSURE>
class SourceResidual4 {
 public:
  KOKKOS_FUNCTION
  SourceResidual4(const EOS &eos, const Opacities &opacities, const Real rho,
                  const Real Ye, const Real bprim[3], const RadiationType species,
                  /*const*/ Tens2 &conTilPi, const Real (&gcov)[4][4],
                  const Real (&gammacon)[3][3], const Real alpha, const Real beta[3],
                  const Real sdetgam, typename CLOSURE::LocalGeometryType &g,
                  Real (&U_mhd_0)[4], Real (&U_rad_0)[4], const int &k, const int &j,
                  const int &i)
      : eos_(eos), opacities_(opacities), rho_(rho), bprim_(&(bprim[0])),
        species_(species), conTilPi_(conTilPi), gcov_(&gcov), gammacon_(&gammacon),
        alpha_(alpha), beta_(&(beta[0])), sdetgam_(sdetgam), g_(g), U_mhd_0_(&U_mhd_0),
        U_rad_0_(&U_rad_0), k_(k), j_(j), i_(i) {
    lambda_[0] = Ye;
    lambda_[1] = 0.;
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateMHDConserved(Real P_mhd[4], Real U_mhd[4]) {
    Real Pg = eos_.PressureFromDensityInternalEnergy(rho_, P_mhd[0] / rho_, lambda_);
    Real gam1 =
        eos_.BulkModulusFromDensityInternalEnergy(rho_, P_mhd[0] / rho_, lambda_) / Pg;
    Real D;
    Real bcons[3];
    Real ye_cons;
    prim2con::p2c(rho_, &(P_mhd[1]), bprim_, P_mhd[0], lambda_[0], Pg, gam1, *gcov_,
                  *gammacon_, beta_, alpha_, sdetgam_, D, &(U_mhd[1]), bcons, U_mhd[0],
                  ye_cons);
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateRadConserved(Real U_mhd[4], Real U_rad[4]) {
    for (int n = 0; n < 4; n++) {
      U_rad[n] = (*U_rad_0_)[n] - (U_mhd[n] - (*U_mhd_0_)[n]);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateRadConservedFromRadPrim(Real P_rad[4], Real U_rad[4]) {
    PARTHENON_FAIL("Not implemented!");
  }

  KOKKOS_INLINE_FUNCTION
  ClosureStatus CalculateRadPrimitive(Real P_mhd[4], Real U_rad[4], Real P_rad[4]) {
    const Real E = U_rad[0] / sdetgam_;
    Vec cov_F{U_rad[1] / sdetgam_, U_rad[2] / sdetgam_, U_rad[3] / sdetgam_};
    Vec cov_H;
    Real W = 0.;
    SPACELOOP2(ii, jj) { W += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] * P_mhd[jj + 1]; }
    W = std::sqrt(1. + W);
    Vec con_v{P_mhd[1] / W, P_mhd[2] / W, P_mhd[3] / W};
    CLOSURE c(con_v, &g_);
    // TODO(BRR) Accept separately calculated con_tilPi as an option
    // TODO(BRR) Store xi, phi guesses
    Real xi;
    Real phi;
    c.GetConTilPiFromCon(E, cov_F, xi, phi, &conTilPi_);
    auto status = c.Con2Prim(E, cov_F, conTilPi_, &(P_rad[0]), &cov_H);
    SPACELOOP(ii) { P_rad[ii + 1] = cov_H(ii); }
    return status;
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateSource(Real P_mhd[4], Real P_rad[4], Real S[4]) {
    Real Tg = eos_.TemperatureFromDensityInternalEnergy(rho_, P_mhd[0] / rho_, lambda_);
    Real JBB = opacities_.EnergyDensityFromTemperature(Tg, species_);
    Real kappaJ =
        opacities_.RosselandMeanAbsorptionCoefficient(rho_, Tg, lambda_[0], species_);
    Real kappaH = kappaJ + opacities_.RosselandMeanAbsorptionCoefficient(
                               rho_, Tg, lambda_[0], species_);
    // TODO(BRR) this is arguably cheating, arguably not. Should include dt though
    // kappaH * dt < 1 / eps
    kappaH = std::min<Real>(kappaH, 1.e5);
    kappaJ = std::min<Real>(kappaJ, 1.e5);
    Real W = 0.;
    SPACELOOP2(ii, jj) { W += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] * P_mhd[jj + 1]; }
    W = std::sqrt(1. + W);
    Real cov_v[3] = {0};
    SPACELOOP2(ii, jj) { cov_v[ii] += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] / W; }
    Real vdH = 0.;
    SPACELOOP(ii) { vdH += P_mhd[ii + 1] / W * P_rad[ii + 1]; }

    S[0] = alpha_ * sdetgam_ * (kappaJ * W * (JBB - P_rad[0]) - kappaH * vdH);
    SPACELOOP(ii) {
      S[ii + 1] = alpha_ * sdetgam_ *
                  (kappaJ * W * cov_v[ii] * (JBB - P_rad[0]) - kappaH * P_rad[1 + ii]);
    }
  }

 private:
  const EOS &eos_;
  const Opacities &opacities_;
  const Real rho_;
  const Real *bprim_;
  const RadiationType species_;
  Real lambda_[2];
  Tens2 &conTilPi_;

  const Real (*gcov_)[4][4];
  const Real (*gammacon_)[3][3];
  const Real alpha_;
  const Real *beta_;
  const Real sdetgam_;

  typename CLOSURE::LocalGeometryType &g_;

  const Real (*U_mhd_0_)[4];
  const Real (*U_rad_0_)[4];

  const int &k_, &j_, &i_;
};

class InteractionTResidual {
 public:
  KOKKOS_FUNCTION
  InteractionTResidual(const EOS &eos, const Opacities &opacities, const Real &rho,
                       const Real &ug0, const Real &Ye, const Real J0[3],
                       const int &nspec, const RadiationType species[3], const Real &dtau,
                       const Real &alpha_max)
      : eos_(eos), opacities_(opacities), rho_(rho), ug0_(ug0), Ye_(Ye), nspec_(nspec),
        dtau_(dtau), alpha_max_(alpha_max) {
    for (int ispec = 0; ispec < nspec; ++ispec) {
      J0_[ispec] = J0[ispec];
      species_[ispec] = species[ispec];
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real T) {
    Real lambda[2] = {Ye_, 0.0};

    Real J0_tot = 0.;
    Real dJ_tot = 0.;

    for (int ispec = 0; ispec < nspec_; ++ispec) {
      J0_tot += J0_[ispec];
      Real kappa =
          opacities_.RosselandMeanAbsorptionCoefficient(rho_, T, Ye_, species_[ispec]);
      kappa = std::min<Real>(kappa, alpha_max_);

      const Real JBB = opacities_.EnergyDensityFromTemperature(T, species_[ispec]);

      dJ_tot += (J0_[ispec] + dtau_ * kappa * JBB) / (1. + dtau_ * kappa) - J0_[ispec];
    }

    const Real ug1 = rho_ * eos_.InternalEnergyFromDensityTemperature(rho_, T, lambda);

    return ((ug1 - ug0_) + (dJ_tot)) / (ug0_ + J0_tot);
  }

 private:
  const EOS &eos_;
  const Opacities &opacities_;
  const Real &rho_;
  const Real &ug0_;
  const Real &Ye_;
  Real J0_[MaxNumRadiationSpecies];
  const int &nspec_;
  const Real &dtau_;      // Proper time
  const Real &alpha_max_; // Maximum optical depth per timestep
  RadiationType species_[MaxNumRadiationSpecies];
};

template <class T, class CLOSURE>
TaskStatus MomentFluidSourceImpl(T *rc, Real dt, bool update_fluid) {
  PARTHENON_REQUIRE(USE_VALENCIA, "Covariant MHD formulation not supported!");

  auto *pmb = rc->GetParentPointer();
  StateDescriptor *fluid_pkg = pmb->packages.Get("fluid").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto eos = eos_pkg->Param<EOS>("d.EOS");
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{c::density,
                                c::energy::name(),
                                c::momentum,
                                c::ye::name(),
                                cr::E,
                                cr::F,
                                c::bfield::name(),
                                p::density::name(),
                                p::temperature,
                                p::energy::name(),
                                p::ye::name(),
                                p::velocity::name(),
                                p::pressure,
                                p::gamma1,
                                p::bfield::name(),
                                pr::J,
                                pr::H,
                                ir::kappaJ,
                                ir::kappaH,
                                ir::JBB,
                                ir::tilPi};

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);
  auto idx_tilPi = imap.GetFlatIdx(ir::tilPi, false);
  auto pv = imap.GetFlatIdx(p::velocity::name());

  int prho = imap[p::density::name()].first;
  int peng = imap[p::energy::name()].first;
  int pT = imap[p::temperature].first;
  int pprs = imap[p::pressure].first;
  int pgm1 = imap[p::gamma1].first;
  int pYe = imap[p::ye::name()].first;
  int pb_lo = imap[p::bfield::name()].first;
  int cb_lo = imap[c::bfield::name()].first;

  int crho = imap[c::density].first;
  int ceng = imap[c::energy::name()].first;
  int cmom_lo = imap[c::momentum].first;
  int cmom_hi = imap[c::momentum].second;
  int cye = imap[c::ye::name()].first;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  const parthenon::Coordinates_t &coords = pmb->coords;

  const auto &opacities = opac->Param<Opacities>("opacities");

  const auto src_solver = rad->Param<SourceSolver>("src_solver");
  const auto src_use_oned_backup = rad->Param<bool>("src_use_oned_backup");
  const auto src_rootfind_eps = rad->Param<Real>("src_rootfind_eps");
  const auto src_rootfind_tol = rad->Param<Real>("src_rootfind_tol");
  const auto src_rootfind_maxiter = rad->Param<int>("src_rootfind_maxiter");
  const auto oned_fixup_strategy = rad->Param<OneDFixupStrategy>("oned_fixup_strategy");

  const Real c2p_tol = update_fluid ? fluid_pkg->Param<Real>("c2p_tol") : 0.;
  const int c2p_max_iter = update_fluid ? fluid_pkg->Param<int>("c2p_max_iter") : 0;
  const Real c2p_floor_scale_fac =
      update_fluid ? fluid_pkg->Param<Real>("c2p_floor_scale_fac") : 0.;
  const bool c2p_fail_on_floors = true;
  const bool c2p_fail_on_ceilings = true;
  auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter,
                                                c2p_floor_scale_fac, c2p_fail_on_floors,
                                                c2p_fail_on_ceilings);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        // Geometry
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);
        Tens2 con_gamma;
        geom.MetricInverse(CellLocation::Cent, iblock, k, j, i, con_gamma.data);
        Real cov_g[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, iblock, k, j, i, cov_g);
        Real beta[3];
        geom.ContravariantShift(CellLocation::Cent, iblock, k, j, i, beta);
        Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
        Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, iblock, k, j, i);

        // Bounds
        Real xi_max;
        Real tau_max;
        bounds.GetRadiationCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                                    coords.Xc<3>(k, j, i), xi_max, tau_max);
        const Real alpha_max = tau_max / (alpha * dt);

        Real rho = v(iblock, prho, k, j, i);
        Real ug = v(iblock, peng, k, j, i);
        Real Tg = v(iblock, pT, k, j, i);
        Real Ye = pYe > 0 ? v(iblock, pYe, k, j, i) : 0.5;
        Real bprim[3] = {0};
        if (pb_lo > -1) {
          Real bprim[3] = {v(iblock, pb_lo, k, j, i), v(iblock, pb_lo + 1, k, j, i),
                           v(iblock, pb_lo + 2, k, j, i)};
        }
        Real lambda[2] = {Ye, 0.};
        Real con_vp[3] = {v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                          v(iblock, pv(2), k, j, i)};

        Real dE[MaxNumRadiationSpecies];
        Vec cov_dF[MaxNumRadiationSpecies];

        bool success = false;
        if (src_solver == SourceSolver::zerod) {
          PARTHENON_FAIL("zerod solver temporarily disabled!");
          success = true;
        } else if (src_solver == SourceSolver::oned) {
        oned_solver_begin:
          for (int ispec = 0; ispec < num_species; ispec++) {

            const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
            Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};

            Real J0[3] = {v(iblock, idx_J(0), k, j, i), 0, 0};
            PARTHENON_REQUIRE(num_species == 1, "Multiple species not implemented");

            const Real dtau = alpha * dt / W; // Elapsed time in fluid frame

            // Rootfind over fluid temperature in fluid rest frame
            root_find::RootFind root_find(src_rootfind_maxiter);
            InteractionTResidual res(eos, opacities, rho, ug, Ye, J0, num_species,
                                     species_d, dtau, alpha_max);
            root_find::RootFindStatus status;
            Real T1 = root_find.regula_falsi(res, 1.e-2 * v(iblock, pT, k, j, i),
                                             1.e2 * v(iblock, pT, k, j, i),
                                             src_rootfind_tol * v(iblock, pT, k, j, i),
                                             v(iblock, pT, k, j, i), &status);
            Real rebracketing_fac = 10.;
            int n_rebracketing_tries = 0;
            constexpr int MAX_REBRACKETING_TRIES = 4;
            while (status == root_find::RootFindStatus::failure &&
                   n_rebracketing_tries < MAX_REBRACKETING_TRIES) {
              T1 = root_find.regula_falsi(
                  res, 1.e-1 / rebracketing_fac * v(iblock, pT, k, j, i),
                  1.e1 * rebracketing_fac * v(iblock, pT, k, j, i),
                  src_rootfind_tol * v(iblock, pT, k, j, i), v(iblock, pT, k, j, i),
                  &status);

              n_rebracketing_tries++;
              rebracketing_fac *= 10.;
            }

            if (status == root_find::RootFindStatus::failure) {
              PARTHENON_DEBUG_WARN("1D source rootfind failure!");
              success = false;
            }

            CLOSURE c(con_v, &g);

            Real Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
            Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam};

            // Treat the Eddington tensor explicitly for now
            Real J = v(iblock, idx_J(ispec), k, j, i);
            Vec cov_H{{
                J * v(iblock, idx_H(ispec, 0), k, j, i),
                J * v(iblock, idx_H(ispec, 1), k, j, i),
                J * v(iblock, idx_H(ispec, 2), k, j, i),
            }};
            Tens2 con_tilPi;

            if (idx_tilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                con_tilPi(ii, jj) = v(iblock, idx_tilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              c.GetConTilPiFromPrim(J, cov_H, &con_tilPi);
            }

            Real JBB = opacities.EnergyDensityFromTemperature(T1, species_d[ispec]);
            Real kappaJ = opacities.RosselandMeanAbsorptionCoefficient(rho, T1, Ye,
                                                                       species_d[ispec]);
            kappaJ = std::min<Real>(kappaJ, alpha_max);
            Real kappaH = opacities.RosselandMeanScatteringCoefficient(rho, T1, Ye,
                                                                       species_d[ispec]);
            kappaH = std::min<Real>(kappaH, alpha_max);
            kappaH += kappaJ;
            Real tauJ = alpha * dt * kappaJ;
            Real tauH = alpha * dt * kappaH;

            if (status == root_find::RootFindStatus::success) {
              c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ, tauH,
                                   &(dE[ispec]), &(cov_dF[ispec]));

              Estar += dE[ispec];
              SPACELOOP(ii) cov_Fstar(ii) += cov_dF[ispec](ii);

              // Check result for sanity (note that this doesn't check fluid energy
              // density)
              auto c2p_status = c.Con2Prim(Estar, cov_Fstar, con_tilPi, &J, &cov_H);
              if (c2p_status == ClosureStatus::success) {
                success = true;
              } else {
                PARTHENON_DEBUG_WARN("closure failure!");
                success = false;
              }
            } else {
              PARTHENON_DEBUG_WARN("success is false!");
              success = false;
            }

            if (!success && oned_fixup_strategy == OneDFixupStrategy::ignore_dJ) {
              // Retry the solver update but without updating internal energy
              Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
              SPACELOOP(ii) {
                cov_Fstar(ii) = v(iblock, idx_F(ispec, ii), k, j, i) / sdetgam;
              }

              Real JBB = opacities.EnergyDensityFromTemperature(v(iblock, pT, k, j, i),
                                                                species_d[ispec]);
              // Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
              Real kappaJ = opacities.RosselandMeanAbsorptionCoefficient(
                  rho, T1, Ye, species_d[ispec]);
              Real kappaH = opacities.RosselandMeanScatteringCoefficient(
                                rho, T1, Ye, species_d[ispec]) +
                            kappaJ;
              tauJ = alpha * dt * kappaJ;
              tauH = alpha * dt * kappaH;
              c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ, tauH,
                                   &(dE[ispec]), &(cov_dF[ispec]));

              // Check xi
              Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam + dE[ispec];
              SPACELOOP(ii) {
                cov_Fstar(ii) =
                    v(iblock, idx_F(ispec, ii), k, j, i) / sdetgam + cov_dF[ispec](ii);
              }
              auto c2p_status = c.Con2Prim(Estar, cov_Fstar, con_tilPi, &J, &cov_H);

              const Real xi =
                  std::sqrt(g.contractCov3Vectors(cov_H, cov_H) -
                            std::pow(g.contractConCov3Vectors(con_v, cov_H), 2)) /
                  J;
              if (c2p_status == ClosureStatus::success) {
                printf("c2p ignore dJ success!\n");
                success = true;
              } else {
                printf("c2p ignore dJ failure!\n");
                success = false;
                break;
              }
            } else if (!success && oned_fixup_strategy == OneDFixupStrategy::ignore_all) {
              dE[ispec] = 0.;
              SPACELOOP(ii) { cov_dF[ispec](ii) = 0.; }
              success = true;
            } else {
              break; // Don't bother with other species if this failed
            }
          } // ispec
        } else if (src_solver == SourceSolver::fourd) {
          // TODO(BRR) generalize to multiple species
          int ispec = 0;

          // Rootfind via Newton's method over 4-momentum with some fixups inside the
          // rootfind.
          Real P_mhd_guess[4] = {ug, v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                                 v(iblock, pv(2), k, j, i)};
          Real U_mhd_0[4];
          const Real Pg0 = eos.PressureFromDensityInternalEnergy(rho, ug / rho, lambda);
          const Real gam1 =
              eos.BulkModulusFromDensityInternalEnergy(rho, ug / rho, lambda) / Pg0;
          Real D;
          Real bcons[3];
          Real ye_cons;
          prim2con::p2c(rho, &(P_mhd_guess[1]), bprim, P_mhd_guess[0], lambda[0], Pg0,
                        gam1, cov_g, con_gamma.data, beta, alpha, sdetgam, D,
                        &(U_mhd_0[1]), bcons, U_mhd_0[0], ye_cons);
          Real U_rad_0[4] = {
              v(iblock, idx_E(ispec), k, j, i), v(iblock, idx_F(ispec, 0), k, j, i),
              v(iblock, idx_F(ispec, 1), k, j, i), v(iblock, idx_F(ispec, 2), k, j, i)};

          // TODO(BRR) instead update con_TilPi each substep?
          // TODO(BRR) update v in closure each substep?
          Vec con_v{{P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]}};
          const Real W = phoebus::GetLorentzFactor(con_v.data, cov_gamma.data);
          SPACELOOP(ii) { con_v(ii) /= W; }
          CLOSURE c(con_v, &g);
          Tens2 con_tilPi = {0};
          if (idx_tilPi.IsValid()) {
            SPACELOOP2(ii, jj) {
              con_tilPi(ii, jj) = v(iblock, idx_tilPi(ispec, ii, jj), k, j, i);
            }
          } else {
            Real J = v(iblock, idx_J(ispec), k, j, i);
            Vec covH{{v(iblock, idx_F(ispec, 0), k, j, i),
                      v(iblock, idx_F(ispec, 1), k, j, i),
                      v(iblock, idx_F(ispec, 2), k, j, i)}};
            c.GetConTilPiFromPrim(J, covH, &con_tilPi);
          }
          SourceResidual4<CLOSURE> srm(eos, opacities, rho, Ye, bprim, species_d[ispec],
                                       con_tilPi, cov_g, con_gamma.data, alpha, beta,
                                       sdetgam, g, U_mhd_0, U_rad_0, k, j, i);

          // Memory for evaluating Jacobian via finite differences
          Real P_rad_m[4];
          Real P_rad_p[4];
          Real U_mhd_m[4];
          Real U_mhd_p[4];
          Real U_rad_m[4];
          Real U_rad_p[4];
          Real dS_m[4];
          Real dS_p[4];

          Real U_mhd_guess[4];
          Real U_rad_guess[4];
          Real P_rad_guess[4];
          Real dS_guess[4];

          // Initialize residuals
          Real resid[4];
          srm.CalculateMHDConserved(P_mhd_guess, U_mhd_guess);
          srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
          auto status = srm.CalculateRadPrimitive(P_mhd_guess, U_rad_guess, P_rad_guess);
          srm.CalculateSource(P_mhd_guess, P_rad_guess, dS_guess);
          for (int n = 0; n < 4; n++) {
            resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
          }

          Real err = 1.e100;
          int niter = 0;
          bool bad_guess = false;
          do {
            // Numerically calculate Jacobian
            Real jac[4][4] = {0};

            // Minimum non-zero magnitude from P_mhd_guess
            Real P_mhd_mag_min = robust::LARGE();
            for (int m = 0; m < 4; m++) {
              if (std::fabs(P_mhd_guess[m]) > 0.) {
                P_mhd_mag_min = std::min<Real>(P_mhd_mag_min, std::fabs(P_mhd_guess[m]));
              }
            }

            bool bad_guess_m = false;
            bool bad_guess_p = false;
            for (int m = 0; m < 4; m++) {
              Real P_mhd_m[4] = {P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2],
                                 P_mhd_guess[3]};
              Real P_mhd_p[4] = {P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2],
                                 P_mhd_guess[3]};
              const Real fd_step = std::max(src_rootfind_eps * P_mhd_mag_min,
                                            src_rootfind_eps * std::fabs(P_mhd_guess[m]));
              P_mhd_m[m] -= fd_step;
              P_mhd_p[m] += fd_step;

              srm.CalculateMHDConserved(P_mhd_m, U_mhd_m);
              srm.CalculateRadConserved(U_mhd_m, U_rad_m);
              auto status_m = srm.CalculateRadPrimitive(P_mhd_m, U_rad_m, P_rad_m);
              srm.CalculateSource(P_mhd_m, P_rad_m, dS_m);
              if (status_m == ClosureStatus::failure) {
                bad_guess_m = true;
              }

              srm.CalculateMHDConserved(P_mhd_p, U_mhd_p);
              srm.CalculateRadConserved(U_mhd_p, U_rad_p);
              auto status_p = srm.CalculateRadPrimitive(P_mhd_p, U_rad_p, P_rad_p);
              srm.CalculateSource(P_mhd_p, P_rad_p, dS_p);
              if (status_p == ClosureStatus::failure) {
                bad_guess_p = true;
              }

              for (int n = 0; n < 4; n++) {
                Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                jac[n][m] = (fp - fm) / (P_mhd_p[m] - P_mhd_m[m]);
              }
            }

            // Fail or repair if jacobian evaluation was pathological
            if (bad_guess_m == true && bad_guess_p == true) {
              // If both + and - finite difference support points are bad, this
              // interaction fails
              bad_guess = true;
              break;
            } else if (bad_guess_m == true) {
              // If only - finite difference support point is bad, do one-sided
              // difference with + support point
              for (int m = 0; m < 4; m++) {
                Real P_mhd_p[4] = {P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2],
                                   P_mhd_guess[3]};
                P_mhd_p[m] += std::max(src_rootfind_eps * P_mhd_mag_min,
                                       src_rootfind_eps * std::fabs(P_mhd_p[m]));
                srm.CalculateMHDConserved(P_mhd_p, U_mhd_p);
                srm.CalculateRadConserved(U_mhd_p, U_rad_p);
                auto status_p = srm.CalculateRadPrimitive(P_mhd_p, U_rad_p, P_rad_p);
                srm.CalculateSource(P_mhd_p, P_rad_p, dS_p);
                PARTHENON_DEBUG_REQUIRE(status_p == ClosureStatus::success,
                                        "This inversion should have already worked!");
                for (int n = 0; n < 4; n++) {
                  Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
                  Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                  jac[n][m] = (fp - fguess) / (P_mhd_p[m] - P_mhd_guess[m]);
                }
              }
            } else if (bad_guess_p == true) {
              // If only + finite difference support point is bad, do one-sided
              // difference with - support point
              for (int m = 0; m < 4; m++) {
                Real P_mhd_m[4] = {P_mhd_guess[0], P_mhd_guess[1], P_mhd_guess[2],
                                   P_mhd_guess[3]};
                P_mhd_m[m] -= std::max(src_rootfind_eps * P_mhd_mag_min,
                                       src_rootfind_eps * std::fabs(P_mhd_m[m]));
                srm.CalculateMHDConserved(P_mhd_m, U_mhd_m);
                srm.CalculateRadConserved(U_mhd_m, U_rad_m);
                auto status_m = srm.CalculateRadPrimitive(P_mhd_m, U_rad_m, P_rad_m);
                srm.CalculateSource(P_mhd_m, P_rad_m, dS_m);
                PARTHENON_DEBUG_REQUIRE(status_m == ClosureStatus::success,
                                        "This inversion should have already worked!");
                for (int n = 0; n < 4; n++) {
                  Real fguess = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
                  Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
                  jac[n][m] = (fguess - fm) / (P_mhd_guess[m] - P_mhd_m[m]);
                }
              }
            }

            Real jacinv[4][4];
            LinearAlgebra::matrixInverse4x4(jac, jacinv);

            if (bad_guess == false) {

              // Save energies before update in case we need to rescale step
              double ug0 = P_mhd_guess[0];
              double ur0 = P_rad_guess[0];

              // Update guess
              for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                  P_mhd_guess[m] -= jacinv[m][n] * resid[n];
                }
              }

              srm.CalculateMHDConserved(P_mhd_guess, U_mhd_guess);
              srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
              status = srm.CalculateRadPrimitive(P_mhd_guess, U_rad_guess, P_rad_guess);
              srm.CalculateSource(P_mhd_guess, P_rad_guess, dS_guess);
              if (status == ClosureStatus::failure) {
                bad_guess = true;
              }

              if (bad_guess) {
                // Try reducing the step size so xi < ximax, ug > 0, J > 0
                Vec cov_H{P_rad_guess[1] / P_rad_guess[0],
                          P_rad_guess[2] / P_rad_guess[0],
                          P_rad_guess[3] / P_rad_guess[0]};
                Vec con_v{P_mhd_guess[1], P_mhd_guess[2], P_mhd_guess[3]};
                Real J = P_rad_guess[0];
                const Real xi =
                    std::sqrt(g.contractCov3Vectors(cov_H, cov_H) -
                              std::pow(g.contractConCov3Vectors(con_v, cov_H), 2)) /
                    J;

                constexpr Real umin = 100. * robust::SMALL();
                constexpr Real Jmin = 100. * robust::SMALL();

                Real scaling_factor = 0.;
                if (xi > xi_max) {
                  scaling_factor = std::max<Real>(scaling_factor, xi_max / xi);
                }
                if (P_mhd_guess[0] < umin) {
                  scaling_factor = std::max<Real>(scaling_factor,
                                                  (ug0 - umin) / (ug0 - P_mhd_guess[0]));
                }
                if (P_rad_guess[0] < Jmin) {
                  scaling_factor = std::max<Real>(scaling_factor,
                                                  (ur0 - Jmin) / (ur0 - P_rad_guess[0]));
                }
                if (!(scaling_factor > robust::SMALL() && scaling_factor >= 1.)) {
                  // Nonsensical scaling factor, maybe negative P_rad_guess[0]
                  printf("bad scaling factor! breaking [%i %i %i]\n", k, j, i);
                  bad_guess = true;
                  break;
                }
                // Nudge it a bit
                scaling_factor *= 0.5;
                for (int m = 0; m < 4; m++) {
                  for (int n = 0; n < 4; n++) {
                    P_mhd_guess[m] += (1. - scaling_factor) * jacinv[m][n] * resid[n];
                  }
                }
                srm.CalculateMHDConserved(P_mhd_guess, U_mhd_guess);
                srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
                status = srm.CalculateRadPrimitive(P_mhd_guess, U_rad_guess, P_rad_guess);
                srm.CalculateSource(P_mhd_guess, P_rad_guess, dS_guess);
                if (status == ClosureStatus::success) {
                  bad_guess = false;
                }
              }
            }

            // If guess was invalid, break and mark this zone as a failure
            if (bad_guess) {
              break;
            }

            // Update residuals
            for (int n = 0; n < 4; n++) {
              resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
              if (std::isnan(resid[n])) {
                bad_guess = true;
                break;
              }
            }
            if (bad_guess == true) {
              break;
            }

            // Calculate error
            err = robust::SMALL();
            Real max_divisor = robust::SMALL();
            for (int n = 0; n < 4; n++) {
              max_divisor = std::max<Real>(max_divisor, std::fabs(U_mhd_guess[n]) +
                                                            std::fabs(U_mhd_0[n]) +
                                                            std::fabs(dt * dS_guess[n]));
            }
            for (int n = 0; n < 4; n++) {
              Real suberr = std::fabs(resid[n]) / max_divisor;
              if (suberr > err) {
                err = suberr;
              }
            }

            niter++;
            if (niter == src_rootfind_maxiter) {
              break;
            }
          } while (err > src_rootfind_tol);

          if (niter == src_rootfind_maxiter || err > src_rootfind_tol ||
              std::isnan(U_rad_guess[0]) || std::isnan(U_rad_guess[1]) ||
              std::isnan(U_rad_guess[2]) || std::isnan(U_rad_guess[3]) || bad_guess) {
            success = false;
          } else {
            success = true;
            dE[ispec] = (U_rad_guess[0] - v(iblock, idx_E(ispec), k, j, i)) / sdetgam;
            SPACELOOP(ii) {
              cov_dF[ispec](ii) =
                  (U_rad_guess[ii + 1] - v(iblock, idx_F(ispec, ii), k, j, i)) / sdetgam;
            }
          }

          if (success == false) {
            // Optionally fall back to oned solver if fourd encounters an issue
            if (src_use_oned_backup) {
              goto oned_solver_begin;
            }
          }
        } // SourceSolver

        // TODO(BRR) Check this logic for multiple species
        for (int ispec = 0; ispec < num_species; ispec++) {

          if (success == true) {
            v(iblock, idx_E(ispec), k, j, i) += sdetgam * dE[ispec];
            SPACELOOP(ii) {
              v(iblock, idx_F(ispec, ii), k, j, i) += sdetgam * cov_dF[ispec](ii);
            }
          }

          Tens2 conTilPi = {0};
          Real E = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
          Vec covF;
          Real J;
          Vec covH;

          bool successful_prim_recovery = false;

          if (update_fluid) {
            if (cye > 0) {
              v(iblock, cye, k, j, i) -= sdetgam * 0.0;
            }
            v(iblock, ceng, k, j, i) -= sdetgam * dE[ispec];
            SPACELOOP(ii) {
              v(iblock, cmom_lo + ii, k, j, i) -= sdetgam * cov_dF[ispec](ii);
            }

            // Do GRMHD inversion and check that it works
            auto mhd_c2p_status = invert(geom, eos, coords, k, j, i);
            if (mhd_c2p_status == con2prim_robust::ConToPrimStatus::failure) {
              successful_prim_recovery = false;
            } else {
              // Set con_v
              Vec con_v{{v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                         v(iblock, pv(2), k, j, i)}};
              const Real W = phoebus::GetLorentzFactor(con_v.data, cov_gamma.data);
              SPACELOOP(ii) { con_v(ii) /= W; }

              // Do rad inversion and check that it works
              CLOSURE c(con_v, &g, {ClosureCon2PrimStrategy::frail});

              // Update pi
              SPACELOOP(ii) { covF(ii) = v(iblock, idx_F(ispec, ii), k, j, i) / sdetgam; }
              if (idx_tilPi.IsValid()) {
                SPACELOOP2(ii, jj) {
                  conTilPi(ii, jj) = v(iblock, idx_tilPi(ispec, ii, jj), k, j, i);
                }
              } else {
                Real xi = 0.0;
                Real phi = M_PI;
                // TODO(BRR) Remove STORE_GUESS parameter and instead check if closure
                // type is M1?
                // if (STORE_GUESS) {
                //  xi = v(iblock, iXi(ispec), k, j, i);
                //  phi = 1.0001 * v(iblock, iPhi(ispec), k, j, i);
                // }
                c.GetConTilPiFromCon(E, covF, xi, phi, &conTilPi);
                // if (STORE_GUESS) {
                //  v(iblock, iXi(ispec), k, j, i) = xi;
                //  v(iblock, iPhi(ispec), k, j, i) = phi;
                //}
              }

              auto rad_c2p_status = c.Con2Prim(E, covF, conTilPi, &J, &covH);

              if (rad_c2p_status == ClosureStatus::success) {
                successful_prim_recovery = true;
                // Store rad prims
                v(iblock, idx_J(ispec), k, j, i) = J;
                SPACELOOP(ii) { v(iblock, idx_H(ispec, ii), k, j, i) = covH(ii) / J; }
              } else {
                successful_prim_recovery = false;
              }
            }

          } else {
            // No fluid update; just do rad inversion and check that it works

            const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
            Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};
            CLOSURE c(con_v, &g, {ClosureCon2PrimStrategy::frail});
            // Update pi
            SPACELOOP(ii) { covF(ii) = v(iblock, idx_F(ispec, ii), k, j, i) / sdetgam; }
            if (idx_tilPi.IsValid()) {
              SPACELOOP2(ii, jj) {
                conTilPi(ii, jj) = v(iblock, idx_tilPi(ispec, ii, jj), k, j, i);
              }
            } else {
              Real xi = 0.0;
              Real phi = M_PI;
              // TODO(BRR) Remove STORE_GUESS parameter and instead check if closure type
              // is M1?
              // if (STORE_GUESS) {
              //  xi = v(iblock, iXi(ispec), k, j, i);
              //  phi = 1.0001 * v(iblock, iPhi(ispec), k, j, i);
              // }
              c.GetConTilPiFromCon(E, covF, xi, phi, &conTilPi);
              // if (STORE_GUESS) {
              //  v(iblock, iXi(ispec), k, j, i) = xi;
              //  v(iblock, iPhi(ispec), k, j, i) = phi;
              //}
            }

            auto status = c.Con2Prim(E, covF, conTilPi, &J, &covH);
            if (status == ClosureStatus::success) {
              // If it worked, set prims
              v(iblock, idx_J(ispec), k, j, i) = J;
              SPACELOOP(ii) { v(iblock, idx_H(ispec, ii), k, j, i) = covH(ii) / J; }
              successful_prim_recovery = true;
            } else {
              successful_prim_recovery = false;
            }
          }

          if (!successful_prim_recovery) {
            // Source update failed. Try again with smaller kappas? Set to floors? Zero
            // source update?

            if (update_fluid) {
              // Also set con_v for the closure
              v(iblock, prho, k, j, i) = 10. * robust::SMALL();
              v(iblock, peng, k, j, i) = 10. * robust::SMALL();
              SPACELOOP(ii) { v(iblock, pv(ii), k, j, i) = 0.; }
              // TODO(BRR) use Ye here!
              Real ye = 0.5;
              if (pYe >= 0) {
                ye = v(iblock, pYe, k, j, i);
              }
              Real eos_lambda[2] = {ye, 0};
              v(iblock, pprs, k, j, i) = eos.PressureFromDensityInternalEnergy(
                  v(iblock, prho, k, j, i),
                  robust::ratio(v(iblock, peng, k, j, i), v(iblock, prho, k, j, i)),
                  eos_lambda);
              v(iblock, pT, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
                  v(iblock, prho, k, j, i),
                  robust::ratio(v(iblock, peng, k, j, i), v(iblock, prho, k, j, i)),
                  eos_lambda);
              v(iblock, pgm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                  v(iblock, prho, k, j, i), v(iblock, pT, k, j, i), eos_lambda);
              Real vp[3] = {v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                            v(iblock, pv(2), k, j, i)};
              Real bp[3];
              if (pb_lo >= 0) {
                SPACELOOP(ii) { bp[ii] = v(iblock, pb_lo + ii, k, j, i); }
              }
              Real S[3];
              Real Bcons[3];
              Real yecons;
              prim2con::p2c(v(iblock, prho, k, j, i), vp, bp, v(iblock, peng, k, j, i),
                            eos_lambda[0], v(iblock, pprs, k, j, i),
                            v(iblock, pgm1, k, j, i), cov_g, con_gamma.data, beta, alpha,
                            sdetgam, v(iblock, crho, k, j, i), S, Bcons,
                            v(iblock, ceng, k, j, i), yecons);
              // TODO(BRR) set cons ye!
              if (cye >= 0) {
                v(iblock, cye, k, j, i) = yecons;
              }
              SPACELOOP(ii) { v(iblock, cmom_lo + ii, k, j, i) = S[ii]; }
            }

            Vec con_v{{v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                       v(iblock, pv(2), k, j, i)}};
            const Real W = phoebus::GetLorentzFactor(con_v.data, cov_gamma.data);
            SPACELOOP(ii) { con_v(ii) /= W; }

            CLOSURE c(con_v, &g, {ClosureCon2PrimStrategy::frail});

            v(iblock, idx_J(ispec), k, j, i) = robust::SMALL();
            J = v(iblock, idx_J(ispec), k, j, i);
            SPACELOOP(ii) {
              v(iblock, idx_H(ispec, ii), k, j, i) = 0.;
              covH(ii) = v(iblock, idx_H(ispec, ii), k, j, i) * J;
            }
            c.Prim2Con(J, covH, conTilPi, &E, &covF);
            v(iblock, idx_E(ispec), k, j, i) = E * sdetgam;
            SPACELOOP(ii) { v(iblock, idx_F(ispec, ii), k, j, i) = covF(ii) * sdetgam; }
          }
        }
      });
  return TaskStatus::complete;
}
template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) {
  Mesh *pm = rc->GetMeshPointer();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentFluidSourceImpl<T, ClosureM1<settings>>(rc, dt, update_fluid);
  } else if (method == "moment_eddington") {
    return MomentFluidSourceImpl<T, ClosureEdd<settings>>(rc, dt, update_fluid);
  } else if (method == "mocmc") {
    return MomentFluidSourceImpl<T, ClosureMOCMC<settings>>(rc, dt, update_fluid);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real,
                                                           bool);

} // namespace radiation
