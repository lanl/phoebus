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

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include <singularity-eos/eos/eos.hpp>

#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/root_find.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

#include "fluid/prim2con.hpp"

namespace radiation {

using namespace singularity::neutrinos;
using singularity::EOS;

template <typename CLOSURE>
class SourceResidual4 {
 public:
  KOKKOS_FUNCTION
  SourceResidual4(const EOS &eos, const Opacity &opac, const MeanOpacity &mopac,
                  const Real rho, const Real Ye, const Real bprim[3],
                  const RadiationType species, const Tens2 &conTilPi,
                  const Real (&gcov)[4][4], const Real (&gammacon)[3][3],
                  const Real alpha, const Real beta[3], const Real sdetgam,
                  const Real scattering_fraction, typename CLOSURE::LocalGeometryType &g,
                  Real (&U_mhd_0)[4], Real (&U_rad_0)[4], const int &k, const int &j,
                  const int &i)
      : eos_(eos), opac_(opac), mopac_(mopac), rho_(rho), bprim_(&(bprim[0])),
        species_(species), conTilPi_(conTilPi), gcov_(&gcov), gammacon_(&gammacon),
        alpha_(alpha), beta_(&(beta[0])), sdetgam_(sdetgam),
        scattering_fraction_(scattering_fraction), g_(g), U_mhd_0_(&U_mhd_0),
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
      // TODO(BRR) different for non-valencia
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateRadConservedFromRadPrim(Real P_rad[4], Real U_rad[4]) {}

  KOKKOS_INLINE_FUNCTION
  ClosureStatus CalculateRadPrimitive(Real P_mhd[4], Real U_rad[4], Real P_rad[4]) {
    Vec cov_F{U_rad[1] / sdetgam_, U_rad[2] / sdetgam_, U_rad[3] / sdetgam_};
    Vec cov_H;
    Real W = 0.;
    SPACELOOP2(ii, jj) { W += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] * P_mhd[jj + 1]; }
    W = std::sqrt(1. + W);
    Vec con_v{P_mhd[1] / W, P_mhd[2] / W, P_mhd[3] / W};
    CLOSURE c(con_v, &g_);
    auto status = c.Con2Prim(U_rad[0] / sdetgam_, cov_F, conTilPi_, &(P_rad[0]), &cov_H);
    SPACELOOP(ii) { P_rad[ii + 1] = cov_H(ii); }
    return status;
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateSource(Real P_mhd[4], Real P_rad[4], Real S[4]) {
    Real Tg = eos_.TemperatureFromDensityInternalEnergy(rho_, P_mhd[0] / rho_, lambda_);
    Real JBB = opac_.EnergyDensityFromTemperature(Tg, species_);
    Real kappaH =
        mopac_.RosselandMeanAbsorptionCoefficient(rho_, Tg, lambda_[0], species_);
    // TODO(BRR) this is arguably cheating, arguably not. Should include dt though
    // kappaH * dt < 1 / eps
    kappaH = std::min<Real>(kappaH, 1.e5);
    Real kappaJ = (1. - scattering_fraction_) * kappaH;
    // printf("Tg: %e JBB: %e kappaH: %e kappaJ: %e\n", Tg, JBB, kappaH, kappaJ);
    Real W = 0.;
    SPACELOOP2(ii, jj) { W += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] * P_mhd[jj + 1]; }
    W = std::sqrt(1. + W);
    Real cov_v[3] = {0};
    SPACELOOP2(ii, jj) {
      cov_v[ii] += (*gcov_)[ii + 1][jj + 1] * P_mhd[ii + 1] * P_mhd[jj + 1] / (W * W);
    }
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
  const Opacity &opac_;
  const MeanOpacity &mopac_;
  const Real rho_;
  const Real *bprim_;
  const RadiationType species_;
  Real lambda_[2];
  const Tens2 &conTilPi_;

  const Real (*gcov_)[4][4];
  const Real (*gammacon_)[3][3];
  const Real alpha_;
  const Real *beta_;
  const Real sdetgam_;
  const Real scattering_fraction_;

  typename CLOSURE::LocalGeometryType &g_;

  const Real (*U_mhd_0_)[4];
  const Real (*U_rad_0_)[4];

  const int &k_, &j_, &i_;
};

class InteractionTResidual {
 public:
  KOKKOS_FUNCTION
  InteractionTResidual(const EOS &eos, const Opacity &opacity,
                       const MeanOpacity &mean_opacity, const Real &rho, const Real &ug0,
                       const Real &Ye, const Real J0[3], const int &nspec,
                       const RadiationType species[3], const Real &scattering_fraction,
                       const Real &dtau)
      : eos_(eos), opacity_(opacity), mean_opacity_(mean_opacity), rho_(rho), ug0_(ug0),
        Ye_(Ye), nspec_(nspec), scattering_fraction_(scattering_fraction), dtau_(dtau) {
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
      const Real kappa =
          (1. - scattering_fraction_) *
          mean_opacity_.RosselandMeanAbsorptionCoefficient(rho_, T, Ye_, species_[ispec]);

      const Real JBB = opacity_.EnergyDensityFromTemperature(T, species_[ispec]);

      dJ_tot += (J0_[ispec] + dtau_ * kappa * JBB) / (1. + dtau_ * kappa) - J0_[ispec];
    }

    const Real ug1 = rho_ * eos_.InternalEnergyFromDensityTemperature(rho_, T, lambda);

    return ((ug1 - ug0_) + (dJ_tot)) / (ug0_ + J0_tot);
  }

 private:
  const EOS &eos_;
  const Opacity &opacity_;
  const MeanOpacity &mean_opacity_;
  const Real &rho_;
  const Real &ug0_;
  const Real &Ye_;
  Real J0_[3];
  const int &nspec_;
  const Real &scattering_fraction_;
  const Real &dtau_; // Proper time
  RadiationType species_[3];
};

template <class T>
class ReconstructionIndexer {
 public:
  KOKKOS_INLINE_FUNCTION
  ReconstructionIndexer(const T &v, const int chunk_size, const int offset,
                        const int block = 0)
      : v_(v), chunk_size_(chunk_size), offset_(offset), block_(block) {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int idir, const int ivar, const int k, const int j,
                   const int i) const {
    const int idx = idir * chunk_size_ + ivar + offset_;
    return v_(block_, idx, k, j, i);
  }

 private:
  const T &v_;
  const int ntot_ = 1;
  const int chunk_size_;
  const int offset_;
  const int block_;
};

template <class T, class CLOSURE, bool STORE_GUESS>
TaskStatus MomentCon2PrimImpl(T *rc) {
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  std::vector<std::string> variables{
      cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity, ir::xi, ir::phi, ir::c2pfail};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    variables.push_back(ir::tilPi);
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  // TODO(BRR) Should be able to just get an invalid iTilPi back from imap when MOCMC off
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }
  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(2);

  auto iXi = imap.GetFlatIdx(ir::xi);
  auto iPhi = imap.GetFlatIdx(ir::phi);

  auto ifail = imap[ir::c2pfail].first;

  auto geom = Geometry::GetCoordinateSystem(rc);
  const Real pi = acos(-1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Con2Prim", DevExecSpace(), 0,
      v.GetDim(5) - 1,  // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e,       // z-loop
      jb.s, jb.e,       // y-loop
      ib.s, ib.e,       // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        const Real isdetgam = 1.0 / geom.DetGamma(CellLocation::Cent, b, k, j, i);

        const Real vp[3] = {v(b, pv(0), k, j, i), v(b, pv(1), k, j, i),
                            v(b, pv(2), k, j, i)};
        const Real W = phoebus::GetLorentzFactor(vp, cov_gamma.data);
        Vec con_v{{v(b, pv(0), k, j, i) / W, v(b, pv(1), k, j, i) / W,
                   v(b, pv(2), k, j, i) / W}};

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);
        CLOSURE c(con_v, &g);

        Real J;
        Vec covH;
        Tens2 conTilPi;
        Real E = v(b, cE(ispec), k, j, i) * isdetgam;
        Vec covF = {{v(b, cF(ispec, 0), k, j, i) * isdetgam,
                     v(b, cF(ispec, 1), k, j, i) * isdetgam,
                     v(b, cF(ispec, 2), k, j, i) * isdetgam}};

        if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          Real xi = 0.0;
          Real phi = pi;
          if (STORE_GUESS) {
            xi = v(b, iXi(ispec), k, j, i);
            phi = 1.0001 * v(b, iPhi(ispec), k, j, i);
          }
          c.GetCovTilPiFromCon(E, covF, xi, phi, &conTilPi);
          if (STORE_GUESS) {
            v(b, iXi(ispec), k, j, i) = xi;
            v(b, iPhi(ispec), k, j, i) = phi;
          }
        }
        auto status = c.Con2Prim(E, covF, conTilPi, &J, &covH);
        if (i == 48 && j == 63) {
          printf("[%i %i %i] rad fail here? %i\n", k, j, i,
                 status == ClosureStatus::failure);
          printf("J = %e covH = %e %e %e\n", J, covH(0), covH(1), covH(2));
          printf("dirB.s: %i dirB.e: %i\n", dirB.s, dirB.e);
        }
        if (std::isnan(J) || std::isnan(covH(0)) || std::isnan(covH(1)) ||
            std::isnan(covH(2))) {
          printf("k: %i j: %i i: %i ispec: %i\n", k, j, i, ispec);
          printf("E: %e covF: %e %e %e\n", E, covF(0), covF(1), covF(2));
          printf("J: %e covH: %e %e %e\n", J, covH(0), covH(1), covH(2));
          printf("con_v: %e %e %e\n", con_v(0), con_v(1), con_v(2));
          SPACELOOP2(ii, jj) printf("conTilPi[%i %i] = %e\n", ii, jj, conTilPi(ii, jj));
          PARTHENON_FAIL("Radiation Con2Prim NaN.");
        }

        //        if (ispec == 0 && i == 64 && j > 43 && j < 48) {
        //          printf("[%i %i %i] E = %e F = %e %e %e J = %e H = %e %e %e\n", k, j,
        //          i, E,
        //                 covF(0), covF(1), covF(2), J, covH(0), covH(1), covH(2));
        //        }

        v(b, pJ(ispec), k, j, i) = J;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          v(b, pH(ispec, idir), k, j, i) = robust::ratio(covH(idir), J);
          //  covH(idir) /
          //  J; // Used the scaled value of the rest frame flux for reconstruction
        }

        v(b, ifail, k, j, i) =
            (status == ClosureStatus::success ? FailFlags::success : FailFlags::fail);
        // if (v(b, ifail, k, j, i) == FailFlags::fail) {
        if (status != ClosureStatus::success) {
          // printf("fail! %i %i %i J = %e H = %e %e %e\n", k, j, i, J, covH(0), covH(1),
          // covH(2)); printf("fail! %i %i %i\n", k, j, i);
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentCon2Prim(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");

  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentCon2PrimImpl<T, ClosureM1<Vec, Tens2, settings>, true>(rc);
  } else if (method == "moment_eddington") {
    return MomentCon2PrimImpl<T, ClosureEdd<Vec, Tens2, settings>, false>(rc);
  } else if (method == "mocmc") {
    return MomentCon2PrimImpl<T, ClosureMOCMC<Vec, Tens2, settings>, false>(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown");
  }
  return TaskStatus::fail;
}
// template TaskStatus MomentCon2Prim<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MomentCon2Prim<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE>
TaskStatus MomentPrim2ConImpl(T *rc, IndexDomain domain) {
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(domain);
  IndexRange jb = pm->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pm->cellbounds.GetBoundsK(domain);

  std::vector<std::string> variables{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    variables.push_back(ir::tilPi);
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(2);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Prim2Con", DevExecSpace(), 0,
      v.GetDim(5) - 1,  // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e,       // z-loop
      jb.s, jb.e,       // y-loop
      ib.s, ib.e,       // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        // Set up the background
        const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, 0, b, j, i);

        const Real con_vp[3] = {v(b, pv(0), k, j, i), v(b, pv(1), k, j, i),
                                v(b, pv(2), k, j, i)};
        const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
        Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};

        CLOSURE c(con_v, &g);

        Real E;
        Vec covF;
        Tens2 conTilPi;
        Real J = v(b, pJ(ispec), k, j, i);
        Vec covH = {{v(b, pH(ispec, 0), k, j, i) * J, v(b, pH(ispec, 1), k, j, i) * J,
                     v(b, pH(ispec, 2), k, j, i) * J}};

        if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          c.GetCovTilPiFromPrim(J, covH, &conTilPi);
        }

        c.Prim2Con(J, covH, conTilPi, &E, &covF);

        v(b, cE(ispec), k, j, i) = sdetgam * E;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) {
          v(b, cF(ispec, idir), k, j, i) = sdetgam * covF(idir);
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentPrim2Con(T *rc, IndexDomain domain) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentPrim2ConImpl<T, ClosureM1<Vec, Tens2, settings>>(rc, domain);
  } else if (method == "moment_eddington") {
    return MomentPrim2ConImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc, domain);
  } else if (method == "mocmc") {
    return MomentPrim2ConImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc, domain);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}

template TaskStatus MomentPrim2Con<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                        IndexDomain);

// template <class T>
// TaskStatus ReconstructEdgeStates(T *rc) {
//}

template <class T>
TaskStatus ReconstructEdgeStates(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  using namespace PhoebusReconstruction;

  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  bool eddington_known = false;
  if (method == "mocmc") {
    eddington_known = true;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = (pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = (pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = (pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  VariablePack<Real> ql_base =
      rc->PackVariables(std::vector<std::string>{ir::ql}, imap_ql);
  VariablePack<Real> qr_base =
      rc->PackVariables(std::vector<std::string>{ir::qr}, imap_qr);
  std::vector<std::string> variables = {pr::J, pr::H};
  if (eddington_known) {
    variables.push_back(ir::tilPi);
  }
  variables.push_back(ir::dJ);
  VariablePack<Real> v = rc->PackVariables(variables, imap);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (eddington_known) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  ParArrayND<Real> ql_v = rc->Get(ir::ql_v).data;
  ParArrayND<Real> qr_v = rc->Get(ir::qr_v).data;
  VariablePack<Real> v_vel =
      rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
  auto qIdx = imap_ql.GetFlatIdx(ir::ql);

  const int nspec = qIdx.DimSize(1);
  int nrecon = 4 * nspec;
  if (eddington_known) {
    nrecon = (4 + 9) * nspec; // TODO(BRR) 6 instead of 9 for conTilPi by symmetry
  }

  const int offset = imap_ql[ir::ql].first;

  const int nblock = ql_base.GetDim(5);
  const int ndim = pmb->pmy_mesh->ndim;
  auto &coords = pmb->coords;

  // TODO temp
  auto idx_ql = imap_ql.GetFlatIdx(ir::ql);
  auto idx_qr = imap_qr.GetFlatIdx(ir::qr);

  // TODO(JCD): par_for_outer doesn't have a 4d loop pattern which is needed for blocks
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(), 0, 0, 0,
      nrecon + 2, kb.s - dk, kb.e + dk, jb.s - dj, jb.e + dj,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int n, const int k, const int j) {
        const int b = 0; // this will be replaced by the block arg to the lambda
        ReconstructionIndexer<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionIndexer<VariablePack<Real>> qr(qr_base, nrecon, offset, b);

        const VariablePack<Real> &var = (n < nrecon ? v : v_vel);
        const int var_id = n % nrecon;
        Real *pv = &var(b, var_id, k, j, 0);
        Real *pvim1 = pv - 1;
        Real *pvip1 = pv + 1;
        Real *pvjm1 = &var(b, var_id, k, j - dj, 0);
        Real *pvjp1 = &var(b, var_id, k, j + dj, 0);
        Real *pvkm1 = &var(b, var_id, k - dk, j, 0);
        Real *pvkp1 = &var(b, var_id, k + dk, j, 0);
        Real *vi_l, *vi_r, *vj_l, *vj_r, *vk_l, *vk_r;
        if (n < nrecon) {
          vi_l = &ql(0, n, k, j, 1);
          vi_r = &qr(0, n, k, j, 0);
          vj_l = &ql(1 % ndim, n, k, j + dj, 0);
          vj_r = &qr(1 % ndim, n, k, j, 0);
          vk_l = &ql(2 % ndim, n, k + dk, j, 0);
          vk_r = &qr(2 % ndim, n, k, j, 0);
        } else {
          vi_l = &ql_v(0, var_id, k, j, 1);
          vi_r = &qr_v(0, var_id, k, j, 0);
          vj_l = &ql_v(1 % ndim, var_id, k, j + dj, 0);
          vj_r = &qr_v(1 % ndim, var_id, k, j, 0);
          vk_l = &ql_v(2 % ndim, var_id, k + dk, j, 0);
          vk_r = &qr_v(2 % ndim, var_id, k, j, 0);
        }

        // TODO(JCD): do we want to enable other recon methods like weno5?
        // x-direction
        // TODO(BRR) ib.s - 1 means Jl = 0 in first active zone
        ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvim1, pv, pvip1, vi_l,
                                   vi_r);
        // ReconLoop<PiecewiseLinear>(member, ib.s - 2, ib.e + 1, pvim1, pv, pvip1, vi_l,
        //                          vi_r);
        // y-direction
        if (ndim > 1)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvjm1, pv, pvjp1, vj_l, vj_r);

        // if (j > 50 && j < 70 && n >= nrecon) {
        //  printf("[%i] pv: %e %e %e vl: %e vr: %e\n", n, pvjm1[75], pv[75], pvjp1[75],
        //  vj_l[75], vj_r[75]);
        //}
        // z-direction
        if (ndim > 2)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvkm1, pv, pvkp1, vk_l, vk_r);

        /*ReconLoop<PiecewiseConstant>(member, ib.s - 1, ib.e + 1, pv, vi_l,
                                   vi_r);
        // y-direction
        if (ndim > 1)
          ReconLoop<PiecewiseConstant>(member, ib.s, ib.e, pv, vj_l, vj_r);
        // z-direction
        if (ndim > 2)
          ReconLoop<PiecewiseConstant>(member, ib.s, ib.e, pv, vk_l, vk_r);
          */

        // Calculate spatial derivatives of J at zone faces for diffusion limit
        //    x-->
        //    +---+---+
        //    | a | b |
        //    +---+---+
        //    | c Q d |
        //  ^ +---+---+
        //  | | e | f |
        //  y +---+---+
        //
        //  dJ/dx (@ Q) = (d - c)/dx
        //  dJ/dy (@ Q) = (a + b - e - f)/(4*dy)
        if (n < nspec) {
          const Real idx = 1.0 / coords.Dx(X1DIR, k, j, 0);
          const Real idx4 = 0.25 * idx;
          const Real idy = 1.0 / coords.Dx(X2DIR, k, j, 0);
          const Real idy4 = 0.25 * idy;
          const Real idz = 1.0 / coords.Dx(X3DIR, k, j, 0);
          const Real idz4 = 0.25 * idz;
          Real *J = &v(b, idx_J(n), k, j, 0);
          Real *Jjm1 = &v(b, idx_J(n), k, j - dj, 0);
          Real *Jjp1 = &v(b, idx_J(n), k, j + dj, 0);
          Real *Jkm1 = &v(b, idx_J(n), k - dk, j, 0);
          Real *Jkp1 = &v(b, idx_J(n), k + dk, j, 0);
          Real *Jkp1jm1 = &v(b, idx_J(n), k + dk, j - dj, 0);
          Real *Jkm1jm1 = &v(b, idx_J(n), k - dk, j - dj, 0);
          Real *Jkm1jp1 = &v(b, idx_J(n), k - dk, j + dj, 0);
          // x-direction faces
          Real *dJdx = &v(b, idx_dJ(n, 0, 0), k, j, 0);
          Real *dJdy = &v(b, idx_dJ(n, 1, 0), k, j, 0);
          Real *dJdz = &v(b, idx_dJ(n, 2, 0), k, j, 0);
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e + 1, [&](const int i) {
                dJdx[i] = (J[i] - J[i - 1]) * idx;
                dJdy[i] = (Jjp1[i] + Jjp1[i - 1] - Jjm1[i] - Jjm1[i - 1]) * idy4;
                dJdz[i] = (Jkp1[i] + Jkp1[i - 1] - Jkm1[i] - Jkm1[i - 1]) * idz4;
              });
          if (ndim > 1) {
            // y-direction faces
            dJdx = &v(b, idx_dJ(n, 0, 1), k, j, 0);
            dJdy = &v(b, idx_dJ(n, 1, 1), k, j, 0);
            dJdz = &v(b, idx_dJ(n, 2, 1), k, j, 0);
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e, [&](const int i) {
                  dJdx[i] = (J[i + 1] + Jjm1[i + 1] - J[i - 1] - Jjm1[i - 1]) * idx4;
                  dJdy[i] = (J[i] - Jjm1[i]) * idy;
                  dJdz[i] = (Jkp1[i] + Jkp1jm1[i] - Jkm1[i] - Jkm1jm1[i]) * idz4;
                });
          }
          if (ndim > 2) {
            // z-direction faces
            dJdx = &v(b, idx_dJ(n, 0, 2), k, j, 0);
            dJdy = &v(b, idx_dJ(n, 1, 2), k, j, 0);
            dJdz = &v(b, idx_dJ(n, 2, 2), k, j, 0);
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e, [&](const int i) {
                  dJdx[i] = (J[i + 1] + Jkm1[i + 1] - J[i - 1] - Jkm1[i - 1]) * idx4;
                  dJdy[i] = (Jjp1[i] + Jkm1jp1[i] - Jjm1[i] - Jkm1jm1[i]) * idy4;
                  dJdz[i] = (J[i] - Jkm1[i]) * idz;
                });
          }
        }
      });

  return TaskStatus::complete;
}
template TaskStatus ReconstructEdgeStates<MeshBlockData<Real>>(MeshBlockData<Real> *);

// This really only works for MeshBlockData right now since fluxes don't have a block
// index
template <class T, class CLOSURE>
TaskStatus CalculateFluxesImpl(T *rc) {
  // printf("skipping radiation fluxes\n");
  // return TaskStatus::complete;
  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = (pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = (pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = (pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  std::vector<std::string> vars{ir::ql, ir::qr, ir::ql_v, ir::qr_v, ir::dJ, ir::kappaH};
  std::vector<std::string> flxs{cr::E, cr::F};

  auto v = rc->PackVariablesAndFluxes(vars, flxs, imap);

  auto idx_qlv = imap.GetFlatIdx(ir::ql_v);
  auto idx_qrv = imap.GetFlatIdx(ir::qr_v);
  auto idx_ql = imap.GetFlatIdx(ir::ql);
  auto idx_qr = imap.GetFlatIdx(ir::qr);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);

  auto idx_Ff = imap.GetFlatIdx(cr::F);
  auto idx_Ef = imap.GetFlatIdx(cr::E);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  // const int nblock = 1; //v.GetDim(5);

  auto geom = Geometry::GetCoordinateSystem(rc);

  const Real kappaH_min = 1.e-20;

  auto &coords = pmb->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Fluxes", DevExecSpace(), X1DIR,
      pmb->pmy_mesh->ndim, // Loop over directions
      // 0, nblock-1, // Loop over reconstructed variables
      kb.s - dk, kb.e + dk, // z-loop
      jb.s - dj, jb.e + dj, // y-loop
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir_in, const int k, const int j, const int i) {
        const int idir = idir_in - 1;

        const int koff = (idir_in == 3 ? 1 : 0);
        const int joff = (idir_in == 2 ? 1 : 0);
        const int ioff = (idir_in == 1 ? 1 : 0);

        CellLocation face;
        switch (idir) {
        case (0):
          face = CellLocation::Face1;
          break;
        case (1):
          face = CellLocation::Face2;
          break;
        case (2):
          face = CellLocation::Face3;
          break;
        }

        Vec con_beta;
        Tens2 cov_gamma;
        geom.Metric(face, k, j, i, cov_gamma.data);
        geom.ContravariantShift(face, k, j, i, con_beta.data);
        const Real sdetgam = geom.DetGamma(face, k, j, i);
        typename CLOSURE::LocalGeometryType g(geom, face, 0, k, j, i);

        const Real dx = coords.Dx(idir_in, k, j, i) * sqrt(cov_gamma(idir, idir));

        for (int ispec = 0; ispec < num_species; ++ispec) {

          const Real &Jl = v(idx_ql(ispec, 0, idir), k, j, i);
          const Real &Jr = v(idx_qr(ispec, 0, idir), k, j, i);
          const Vec Hl = {Jl * v(idx_ql(ispec, 1, idir), k, j, i),
                          Jl * v(idx_ql(ispec, 2, idir), k, j, i),
                          Jl * v(idx_ql(ispec, 3, idir), k, j, i)};
          const Vec Hr = {Jr * v(idx_qr(ispec, 1, idir), k, j, i),
                          Jr * v(idx_qr(ispec, 2, idir), k, j, i),
                          Jr * v(idx_qr(ispec, 3, idir), k, j, i)};

          const Real con_vpl[3] = {v(idx_qlv(0, idir), k, j, i),
                                   v(idx_qlv(1, idir), k, j, i),
                                   v(idx_qlv(2, idir), k, j, i)};
          const Real con_vpr[3] = {v(idx_qrv(0, idir), k, j, i),
                                   v(idx_qrv(1, idir), k, j, i),
                                   v(idx_qrv(2, idir), k, j, i)};
          const Real Wl = phoebus::GetLorentzFactor(con_vpl, cov_gamma.data);
          const Real Wr = phoebus::GetLorentzFactor(con_vpr, cov_gamma.data);
          Vec con_vl{{con_vpl[0] / Wl, con_vpl[1] / Wl, con_vpl[2] / Wl}};
          Vec con_vr{{con_vpr[0] / Wr, con_vpr[1] / Wr, con_vpr[2] / Wr}};

          Vec cov_dJ{{v(idx_dJ(ispec, 0, idir), k, j, i),
                      v(idx_dJ(ispec, 1, idir), k, j, i),
                      v(idx_dJ(ispec, 2, idir), k, j, i)}};

          //          if (idir == 1 && i == 64 && j > 43 && j < 48 && ispec == 0) {
          //            printf("[%i %i %i] Jl: %e Jr: %e\n", k, j, i, Jl, Jr);
          //            printf("[%i %i %i] vl: %e %e %e vr: %e %e %e\n", k, j, i,
          //            con_vl(0),
          //                   con_vl(1), con_vl(2), con_vr(0), con_vr(1), con_vr(2));
          //          }

          // Calculate the geometric mean of the opacity on either side of the interface,
          // this is necessary for handling the asymptotic limit near sharp surfaces
          Real kappaH = sqrt((v(idx_kappaH(ispec), k, j, i) *
                              v(idx_kappaH(ispec), k - koff, j - joff, i - ioff)));

          const Real a = tanh(ratio(1.0, std::pow(std::abs(kappaH * dx), 1)));

          // Calculate the observer frame quantities on either side of the interface
          /// TODO: (LFR) Add other contributions to the asymptotic flux
          Vec HasymL = -cov_dJ / (3 * kappaH + 3 * kappaH_min);
          Vec HasymR = HasymL;
          CLOSURE cl(con_vl, &g);
          CLOSURE cr(con_vr, &g);
          Real El, Er;
          Tens2 con_tilPil, con_tilPir;
          Vec covFl, conFl, conFl_asym;
          Vec covFr, conFr, conFr_asym;
          Tens2 Pl, Pl_asym; // P^i_j on the left side of the interface
          Tens2 Pr, Pr_asym; // P^i_j on the right side of the interface

          // Fluxes in the asymptotic limit
          if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            // Use reconstructed values of tilPi
            SPACELOOP2(ii, jj) {
              con_tilPil(ii, jj) = v(idx_ql(ispec, 4 + ii + 3 * jj, idir), k, j, i);
              con_tilPir(ii, jj) = v(idx_qr(ispec, 4 + ii + 3 * jj, idir), k, j, i);
            }
          } else {
            cl.GetCovTilPiFromPrim(Jl, HasymL, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, HasymR, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, HasymL, con_tilPil, &conFl_asym, &Pl_asym);
          cr.getFluxesFromPrim(Jr, HasymR, con_tilPir, &conFr_asym, &Pr_asym);

          // Regular fluxes
          if (!programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            // Recalculate Eddington if using J, H
            cl.GetCovTilPiFromPrim(Jl, Hl, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, Hr, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, Hl, con_tilPil, &conFl, &Pl);
          cr.getFluxesFromPrim(Jr, Hr, con_tilPir, &conFr, &Pr);
          cl.Prim2Con(Jl, Hl, con_tilPil, &El, &covFl);
          cr.Prim2Con(Jr, Hr, con_tilPir, &Er, &covFr);

          // Mix the fluxes by the Peclet number
          // TODO: (LFR) Make better choices
          const Real speed = a * 1.0 + (1 - a) * std::max(sqrt(cl.v2), sqrt(cr.v2));
          conFl = a * conFl + (1 - a) * conFl_asym;
          conFr = a * conFr + (1 - a) * conFr_asym;
          Pl = a * Pl + (1 - a) * Pl_asym;
          Pr = a * Pr + (1 - a) * Pr_asym;

          // Correct the fluxes with the shift terms
          conFl(idir) -= con_beta(idir) * El;
          conFr(idir) -= con_beta(idir) * Er;

          SPACELOOP(ii) {
            Pl(idir, ii) -= con_beta(idir) * covFl(ii);
            Pr(idir, ii) -= con_beta(idir) * covFr(ii);
          }

          // Calculate the numerical flux using LLF
          v.flux(idir_in, idx_Ef(ispec), k, j, i) =
              0.5 * sdetgam * (conFl(idir) + conFr(idir) + speed * (El - Er));
          //          if (std::isnan(v.flux(idir_in, idx_Ef(ispec), k, j, i))) {
          //            printf("NAN flux! %i %i %i F: %e Fl Fr: %e %e speed: %e El Er: %e
          //            %e\n", k, j,
          //                   i, v.flux(idir_in, idx_Ef(ispec), k, j, i), conFl(idir),
          //                   conFr(idir), speed, El, Er);
          //            printf("Jl: %e Jr: %e\n", Jl, Jr);
          //            printf("Hl: %e %e %e Hr: %e %e %e\n", Hl(0), Hl(1), Hl(2), Hr(0),
          //            Hr(1),
          //                   Hr(2));
          //            printf("F asym: %e %e\n", conFl_asym(idir), conFr_asym(idir));
          //            PARTHENON_FAIL("nan flux");
          //          }
          //          if (idir == 1 && i == 64 && j > 43 && j < 48 && ispec == 0) {
          //            printf("[%i %i %i] flux[0]: %e\n", k, j, i,
          //                   v.flux(idir_in, idx_Ef(ispec), k, j, i));
          //            printf("El: %e Er: %e\n", El, Er);
          //          }

          /*if (j == 97 && (i == 4 || i == 5) && idir == 0) {
            printf("[%i %i %i] flux[0]: %e\n",
              k,j,i,v.flux(idir_in, idx_Ef(ispec), k, j, i));
          }
          if ((j == 97 || j == 98) && i == 4 && idir == 1) {
            printf("[%i %i %i] flux[1]: %e\n",
              k,j,i,v.flux(idir_in, idx_Ef(ispec), k, j, i));
          }*/

          //          if (j == 97 && i < 6 && ispec == 0 && idir == 0) {
          //            printf("[%i %i %i] Jl: %e Jr: %e Hl: %e %e %e Hr: %e %e %e flux:
          //            %e\n", k, j,
          //                   i, Jl, Jr, Hl(0), Hl(1), Hl(2), Hr(0), Hr(1), Hr(2),
          //                   v.flux(idir_in, idx_Ef(ispec), k, j, i));
          //          }

          SPACELOOP(ii) {
            v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) =
                0.5 * sdetgam *
                (Pl(idir, ii) + Pr(idir, ii) + speed * (covFl(ii) - covFr(ii)));
          }
          if (sdetgam < std::numeric_limits<Real>::min() * 10) {
            v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.0;
            SPACELOOP(ii) v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) = 0.0;
          }
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus CalculateFluxes(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateFluxesImpl<T, ClosureM1<Vec, Tens2, settings>>(rc);
  } else if (method == "moment_eddington") {
    return CalculateFluxesImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc);
  } else if (method == "mocmc") {
    return CalculateFluxesImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateFluxes<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE>
TaskStatus CalculateGeometricSourceImpl(T *rc, T *rc_src) {
  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  // printf("skipping radiation geometric sources\n");
  // return TaskStatus::complete;

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace p = fluid_prim;
  PackIndexMap imap;
  std::vector<std::string> vars{cr::E, cr::F, pr::J, pr::H, p::velocity};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    vars.push_back(ir::tilPi);
  }
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(p::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  PackIndexMap imap_src;
  std::vector<std::string> vars_src{cr::E, cr::F};
  auto v_src = rc_src->PackVariables(vars_src, imap_src);
  auto idx_E_src = imap_src.GetFlatIdx(cr::E);
  auto idx_F_src = imap_src.GetFlatIdx(cr::F);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_E.DimSize(1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::GeometricSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        // Set up the background state
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);
        const Real con_vp[3] = {v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                                v(iblock, pv(2), k, j, i)};
        const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
        Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, iblock, k, j, i);
        CLOSURE c(con_v, &g);

        Real alp = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
        Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
        Vec con_beta;
        geom.ContravariantShift(CellLocation::Cent, k, j, i, con_beta.data);
        Real beta2 = 0.0;
        SPACELOOP2(ii, jj) beta2 += con_beta(ii) * con_beta(jj) * cov_gamma(ii, jj);

        Real dlnalp[ND];
        Real Gamma[ND][ND][ND];
        geom.GradLnAlpha(CellLocation::Cent, iblock, k, j, i, dlnalp);
        geom.ConnectionCoefficient(CellLocation::Cent, iblock, k, j, i, Gamma);

        // Get the gradient of the shift from the Christoffel symbols of the first kind
        // Get the extrinsic curvature from the Christoffel symbols of the first kind
        // All indices are covariant
        Tens2 dbeta, K;
        const Real iFac = 1.0 / (alp + beta2 / alp);
        SPACELOOP2(ii, jj) {
          dbeta(ii, jj) = Gamma[ii + 1][jj + 1][0] + Gamma[ii + 1][0][jj + 1];
          K(ii, jj) = Gamma[ii + 1][0][jj + 1];
          SPACELOOP(kk) K(ii, jj) -= Gamma[ii + 1][kk + 1][jj + 1] * con_beta(kk);
          K(ii, jj) *= iFac;
        }

        for (int ispec = 0; ispec < nspec; ++ispec) {
          Real E = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
          Real J = v(iblock, idx_J(ispec), k, j, i);
          Vec covF{{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                    v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                    v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam}};
          Vec covH{{J * v(iblock, idx_H(ispec, 0), k, j, i),
                    J * v(iblock, idx_H(ispec, 1), k, j, i),
                    J * v(iblock, idx_H(ispec, 2), k, j, i)}};
          Vec conF;
          g.raise3Vector(covF, &conF);
          Tens2 conP, con_tilPi;

          if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            SPACELOOP2(ii, jj) {
              con_tilPi(ii, jj) = v(iblock, iTilPi(ispec, ii, jj), k, j, i);
            }
          } else {
            c.GetCovTilPiFromPrim(J, covH, &con_tilPi);
          }
          c.getConPFromPrim(J, covH, con_tilPi, &conP);

          Real srcE = 0.0;
          SPACELOOP2(ii, jj) srcE += K(ii, jj) * conP(ii, jj);
          SPACELOOP(ii) srcE -= dlnalp[ii + 1] * conF(ii);
          srcE *= alp;

          Vec srcF{0, 0, 0};
          SPACELOOP(ii) {
            SPACELOOP(jj) { srcF(ii) += covF(jj) * dbeta(ii, jj); }
            srcF(ii) -= alp * E * dlnalp[ii + 1];
            SPACELOOP2(jj, kk) {
              srcF(ii) += alp * conP(jj, kk) * Gamma[jj + 1][kk + 1][ii + 1];
            }
          }
          v_src(iblock, idx_E_src(ispec), k, j, i) = sdetgam * srcE;
          SPACELOOP(ii) {
            v_src(iblock, idx_F_src(ispec, ii), k, j, i) = sdetgam * srcF(ii);
          }
          //          if (j == 97 && i == 4) {
          //            printf("sources[%i]: %e %e %e %e\n", ispec,
          //                   v_src(iblock, idx_E_src(ispec), k, j, i),
          //                   v_src(iblock, idx_F_src(ispec, 0), k, j, i),
          //                   v_src(iblock, idx_F_src(ispec, 1), k, j, i),
          //                   v_src(iblock, idx_F_src(ispec, 2), k, j, i));
          //          }
        }
      });
  return TaskStatus::complete;
}
template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateGeometricSourceImpl<T, ClosureM1<Vec, Tens2, settings>>(rc, rc_src);
  } else if (method == "moment_eddington") {
    return CalculateGeometricSourceImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc, rc_src);
  } else if (method == "mocmc") {
    return CalculateGeometricSourceImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc,
                                                                               rc_src);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateGeometricSource<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                                  MeshBlockData<Real> *);

// TODO(BRR) Unsure how else to get radiation parameters from MeshData
template <>
TaskStatus MomentFluidSource(MeshData<Real> *rc, Real dt, bool update_fluid) {
  for (int n = 0; n < rc->NumBlocks(); n++) {
    MomentFluidSource(rc->GetBlockData(n).get(), dt, update_fluid);
  }
  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  // printf("SKIPPING FLUID SOURCE\n");
  // return TaskStatus::complete;
  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E,          cr::F,     c::bfield,  p::density,
                                p::temperature, p::energy, p::ye,      p::velocity,
                                p::bfield,      pr::J,     pr::H,      ir::kappaJ,
                                ir::kappaH,     ir::JBB,   ir::srcfail};
  // printf("skipping fluid update\n");
  // update_fluid = false;
  //  if (update_fluid) {
  vars.push_back(c::energy);
  vars.push_back(c::momentum);
  vars.push_back(c::ye);
  //  }

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);
  auto ifail = imap[ir::srcfail].first;
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int peng = imap[p::energy].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;
  int pb_lo = imap[p::bfield].first;
  int cb_lo = imap[c::bfield].first;

  int ceng(-1), cmom_lo(-1), cmom_hi(-1), cye(-1);
  // if (update_fluid) {
  ceng = imap[c::energy].first;
  cmom_lo = imap[c::momentum].first;
  cmom_hi = imap[c::momentum].second;
  cye = imap[c::ye].first;
  //}

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  // TODO(BRR) This updates all neutrino species (including contributions to fluid)
  // regardless of whether they are active
  // int nspec = idx_E.DimSize(1);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  const auto &d_mean_opacity = opac->Param<MeanOpacity>("d.mean_opacity");
  // Mainly for testing purposes, probably should be able to do this with the opacity code
  // itself
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  constexpr int izone = -1; // 33;
  constexpr int jzone = -1; // 26;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        // TODO(BRR) turn this into a loop
        const int ispec = 0;

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
        // TODO(BRR) go beyond Eddington
        typename ClosureEdd<Vec, Tens2>::LocalGeometryType g(geom, CellLocation::Cent,
                                                             iblock, k, j, i);

        // Write out rootfind explicitly to include fixups

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

        Real Pguess[4] = {ug, v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                          v(iblock, pv(2), k, j, i)};
        Real U_mhd_0[4];
        const Real Pg0 = eos.PressureFromDensityInternalEnergy(rho, ug / rho, lambda);
        const Real gam1 =
            eos.BulkModulusFromDensityInternalEnergy(rho, ug / rho, lambda) / Pg0;
        Real D;
        Real bcons[3];
        Real ye_cons;
        prim2con::p2c(rho, &(Pguess[1]), bprim, Pguess[0], lambda[0], Pg0, gam1, cov_g,
                      con_gamma.data, beta, alpha, sdetgam, D, &(U_mhd_0[1]), bcons,
                      U_mhd_0[0], ye_cons);
        Real U_rad_0[4] = {
            v(iblock, idx_E(ispec), k, j, i), v(iblock, idx_F(ispec, 0), k, j, i),
            v(iblock, idx_F(ispec, 1), k, j, i), v(iblock, idx_F(ispec, 2), k, j, i)};

        //           if (i == 32 && j == 32 && ispec == 0) {
        //            Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
        //                rho, Tg, Ye, species_d[ispec]);
        //            Real tau = alpha * dt * kappa;
        //            printf("tau[32, 32] = %e\n", tau);
        //           }

        // TODO(BRR) go beyond Eddington
        Tens2 conTilPi = {0};
        SourceResidual4<ClosureEdd<Vec, Tens2>> srm(
            eos, d_opacity, d_mean_opacity, rho, Ye, bprim, species_d[ispec], conTilPi,
            cov_g, con_gamma.data, alpha, beta, sdetgam, scattering_fraction, g, U_mhd_0,
            U_rad_0, k, j, i);

        // TODO(BRR) use Htilde instead of H = J*Htilde for rad prims because mhd prims
        // are like 0 - 1?

        // Memory for evaluating Jacobian via finite differences
        Real P_rad_m[4];
        Real P_rad_p[4];
        Real U_mhd_m[4];
        Real U_mhd_p[4];
        Real U_rad_m[4];
        Real U_rad_p[4];
        Real dS_m[4];
        Real dS_p[4];

        // Alias existing allocated memory to save space
        Real *U_mhd_guess = &(U_mhd_p[0]);
        Real *U_rad_guess = &(U_rad_p[0]);
        Real *P_rad_guess = &(P_rad_p[0]);
        Real *dS_guess = &(dS_p[0]);
        //        if (i == 64 && j == 64) {
        //        printf("Initial cons mhd rad:\n");
        //        printf("mhd: %e %e %e %e\n", v(iblock, ceng, k, j, i), v(iblock,
        //        cmom_lo, k, j, i),
        //          v(iblock, cmom_lo+1, k, j, i), v(iblock, cmom_lo+2, k, j, i));
        //        printf("rad: %e %e %e %e\n", v(iblock, idx_E(0), k, j, i), v(iblock,
        //        idx_F(ispec, 0), k, j, i),
        //          v(iblock, idx_F(ispec, 1), k, j, i), v(iblock, idx_F(ispec, 2), k, j,
        //          i));
        //        }

        // Initialize residuals
        Real resid[4];
        srm.CalculateMHDConserved(Pguess, U_mhd_guess);
        srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
        auto status = srm.CalculateRadPrimitive(Pguess, U_rad_guess, P_rad_guess);
        if (i == 44 && j == 36) {
          printf("[%i %i %i] rho = %e ug = %e Tg = %e Prad0: %e %e %e %e\n", k, j, i, rho,
                 ug, Tg, P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3]);
        }
        srm.CalculateSource(Pguess, P_rad_guess, dS_guess);
        for (int n = 0; n < 4; n++) {
          // TODO(BRR) different for non-valencia
          resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
        }
        //        if (i == 20 && j == 64) {
        //          printf("Pmhd0: %e %e %e %e\n", Pguess[0], Pguess[1], Pguess[2],
        //          Pguess[3]); printf("Umhd0: %e %e %e %e\n", U_mhd_guess[0],
        //          U_mhd_guess[1], U_mhd_guess[2],
        //                 U_mhd_guess[3]);
        //          printf("Prad0: %e %e %e %e\n", P_rad_guess[0], P_rad_guess[1],
        //          P_rad_guess[2],
        //                 P_rad_guess[3]);
        //          printf("Urad0: %e %e %e %e\n", U_rad_guess[0], U_rad_guess[1],
        //          U_rad_guess[2],
        //                 U_rad_guess[3]);
        //        }
        //        if (i == 64 && j == 64) {
        //          printf("Pmhd Umhd Prad Urad:\n");
        //          for (int n = 0; n < 4; n++) {
        //            printf("%e %e %e %e\n", Pguess[n], U_mhd_guess[n], P_rad_guess[n],
        //            U_rad_guess[n]);
        //          }
        //        }

        bool success = false;
        // TODO(BRR) These will need to be per-species later
        Real dE;
        Vec cov_dF;

        Real err = 1.e100;
        constexpr Real TOL = 1.e-8;
        constexpr int max_iter = 100;
        int niter = 0;
        do {
          // Numerically calculate Jacobian
          Real jac[4][4] = {0};
          constexpr Real EPS = 1.e-8;

          // Minimum non-zero magnitude from Pguess
          Real P_mhd_mag_min = robust::LARGE();
          for (int m = 0; m < 4; m++) {
            if (std::fabs(Pguess[m]) > 0.) {
              P_mhd_mag_min = std::min<Real>(P_mhd_mag_min, std::fabs(Pguess[m]));
            }
          }

          bool bad_guess = false;
          for (int m = 0; m < 4; m++) {
            Real P_mhd_m[4] = {Pguess[0], Pguess[1], Pguess[2], Pguess[3]};
            Real P_mhd_p[4] = {Pguess[0], Pguess[1], Pguess[2], Pguess[3]};
            //            // TODO(BRR) Use minimum non-zero magnitude from Pguess[]
            //            Real P_mhd_mag_min = std::max<Real>(
            //                std::max<Real>(std::max<Real>(std::fabs(Pguess[0]),
            //                std::fabs(Pguess[1])),
            //                               std::fabs(Pguess[2])),
            //                std::fabs(Pguess[3]));
            //            printf("Pguess: %e %e %e %e P_mhd_mag_min: %e\n",
            //              Pguess[0], Pguess[1], Pguess[2], Pguess[3], P_mhd_mag_min);
            P_mhd_m[m] -= std::max(EPS * P_mhd_mag_min, EPS * std::fabs(P_mhd_m[m]));
            P_mhd_p[m] += std::max(EPS * P_mhd_mag_min, EPS * std::fabs(P_mhd_p[m]));
            //            printf("m: %i delta: %e (%e %e)\n", m,
            // i              std::max(EPS * P_mhd_mag_min, EPS * std::fabs(P_mhd_m[m])),
            //              EPS * P_mhd_mag_min,
            //              EPS * std::fabs(P_mhd_m[m]));

            srm.CalculateMHDConserved(P_mhd_m, U_mhd_m);
            srm.CalculateRadConserved(U_mhd_m, U_rad_m);
            status = srm.CalculateRadPrimitive(P_mhd_m, U_rad_m, P_rad_m);
            srm.CalculateSource(P_mhd_m, P_rad_m, dS_m);
            if (status == ClosureStatus::failure) {
              printf("bad guess! P_rad_m = %e %e %e %e\n", P_rad_m[0], P_rad_m[1],
               P_rad_m[2], P_rad_m[3]);
              bad_guess = true;
            }

            srm.CalculateMHDConserved(P_mhd_p, U_mhd_p);
            srm.CalculateRadConserved(U_mhd_p, U_rad_p);
            status = srm.CalculateRadPrimitive(P_mhd_p, U_rad_p, P_rad_p);
            srm.CalculateSource(P_mhd_p, P_rad_p, dS_p);
            if (status == ClosureStatus::failure) {
              printf("bad guess! P_rad_p = %e %e %e %e\n", P_rad_p[0], P_rad_p[1],
               P_rad_p[2], P_rad_p[3]);
              bad_guess = true;
            }

            for (int n = 0; n < 4; n++) {
              // TODO(BRR) -dt*dS for non-valencia
              Real fp = U_mhd_p[n] - U_mhd_0[n] + dt * dS_p[n];
              Real fm = U_mhd_m[n] - U_mhd_0[n] + dt * dS_m[n];
              jac[n][m] = (fp - fm) / (P_mhd_p[m] - P_mhd_m[m]);
            }
          }

          Real jacinv[4][4];
          LinearAlgebra::Invert4x4Matrix(jac, jacinv);

          // Calculate residual (unneeded, use value from end of last step?)
          //          srm.CalculateMHDConserved(Pguess, U_mhd_guess);
          //          srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
          //          status = srm.CalculateRadPrimitive(Pguess, U_rad_guess,
          //          P_rad_guess); srm.CalculateSource(Pguess, P_rad_guess, dS_guess); if
          //          (status == ClosureStatus::failure) {
          //            bad_guess = true;
          //          }

          if (i == izone && j == jzone) {
            printf("\n[%i]\n", niter);
            printf("Pmhd0: %e %e %e %e\n", Pguess[0], Pguess[1], Pguess[2], Pguess[3]);
            printf("Umhd0: %e %e %e %e\n", U_mhd_guess[0], U_mhd_guess[1], U_mhd_guess[2],
                   U_mhd_guess[3]);
            printf("Prad0: %e %e %e %e\n", P_rad_guess[0], P_rad_guess[1], P_rad_guess[2],
                   P_rad_guess[3]);
            printf("Urad0: %e %e %e %e\n", U_rad_guess[0], U_rad_guess[1], U_rad_guess[2],
                   U_rad_guess[3]);
            printf("resid: %e %e %e %e\n", resid[0], resid[1], resid[2], resid[3]);
          }

          if (bad_guess == false) {

            // Update guess
            for (int m = 0; m < 4; m++) {
              for (int n = 0; n < 4; n++) {
                Pguess[m] -= jacinv[m][n] * resid[n];
              }
            }

            // TODO(BRR) check that this guess is sane i.e. ug > 0 J > 0 xi < 1, if not,
            // recalculate the guess with some rescaled jacinv*resid

            if (i == izone && j == jzone) {
              printf("Pmhdg: %e %e %e %e\n", Pguess[0], Pguess[1], Pguess[2], Pguess[3]);
            }
            srm.CalculateMHDConserved(Pguess, U_mhd_guess);
            srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
            status = srm.CalculateRadPrimitive(Pguess, U_rad_guess, P_rad_guess);
            srm.CalculateSource(Pguess, P_rad_guess, dS_guess);
            if (i == izone && j == jzone) {
              printf("\n[%i]\n", niter);
              printf("Pmhdg: %e %e %e %e\n", Pguess[0], Pguess[1], Pguess[2], Pguess[3]);
              printf("Umhdg: %e %e %e %e\n", U_mhd_guess[0], U_mhd_guess[1],
                     U_mhd_guess[2], U_mhd_guess[3]);
              printf("Pradg: %e %e %e %e\n", P_rad_guess[0], P_rad_guess[1],
                     P_rad_guess[2], P_rad_guess[3]);
              printf("Uradg: %e %e %e %e\n", U_rad_guess[0], U_rad_guess[1],
                     U_rad_guess[2], U_rad_guess[3]);
            }
            if (status == ClosureStatus::failure) {
               printf("bad guess Pguess: %e %e %e %e Pr: %e %e %e %e\n", Pguess[0],
               Pguess[1], Pguess[2], Pguess[3], P_rad_guess[0], P_rad_guess[1],
               P_rad_guess[2], P_rad_guess[3]); printf("[%i %i %i]\n", k, j, i);
               printf("rho: %e T: %e ug: %e\n", rho, Tg, ug);
              bad_guess = true;
              // exit(-1);
            }

            // Try reducing the size of the step
            if (bad_guess) {
              double xi = 0.1;
              for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                  Pguess[m] += (1. - xi) * jacinv[m][n] * resid[n];
                }
              }
              srm.CalculateMHDConserved(Pguess, U_mhd_guess);
              srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
              status = srm.CalculateRadPrimitive(Pguess, U_rad_guess, P_rad_guess);
              srm.CalculateSource(Pguess, P_rad_guess, dS_guess);
              if (status == ClosureStatus::success) {
                bad_guess = false;
              }
              // printf("after fix? bad guess = %i\n", static_cast<int>(bad_guess));
            }

            // Try reducing the size of the step again
            if (bad_guess) {
              double xi = 0.01;
              for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                  Pguess[m] += (1. - xi) * jacinv[m][n] * resid[n];
                }
              }
              srm.CalculateMHDConserved(Pguess, U_mhd_guess);
              srm.CalculateRadConserved(U_mhd_guess, U_rad_guess);
              status = srm.CalculateRadPrimitive(Pguess, U_rad_guess, P_rad_guess);
              srm.CalculateSource(Pguess, P_rad_guess, dS_guess);
              if (status == ClosureStatus::success) {
                bad_guess = false;
              }
              // printf("after fix? bad guess = %i\n", static_cast<int>(bad_guess));
            }
          }

          // If guess was invalid, break and mark this zone as a failure
          if (bad_guess) {
            break;
          }

          // Update residuals
          for (int n = 0; n < 4; n++) {
            resid[n] = U_mhd_guess[n] - U_mhd_0[n] + dt * dS_guess[n];
            if (i == izone && j == jzone) {
              printf("resid[%i] = %e (%e - %e + %e) (dt = %e dS = %e)\n", n, resid[n],
                     U_mhd_guess[n], U_mhd_0[n], dt * dS_guess[n], dt, dS_guess[n]);
            }
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
            Real suberr =
                // std::fabs(resid[n]) / (std::fabs(U_mhd_guess[n]) +
                // std::fabs(U_mhd_0[n]) +
                //                       std::fabs(dt * dS_guess[n]));
                std::fabs(resid[n]) / max_divisor;
            //            if (i == 25 && j == 101) {
            //              printf("suberr[%i] = %e (%e / %e + %e + %e) dt = %e\n", n,
            //              suberr,
            //                     std::fabs(resid[n]), std::fabs(U_mhd_guess[n]),
            //                     std::fabs(U_mhd_0[n]), std::fabs(dt * dS_guess[n]),
            //                     dt);
            //            }
            if (suberr > err) {
              err = suberr;
            }
          }

          if (i == izone && j == jzone) {
            printf("[%i] err: %e TOL: %e\n", niter, err, TOL);
          }

          niter++;
          if (niter == max_iter) {
            break;
          }
        } while (err > TOL);

                if (niter == 0) {
                  printf("[%i %i %i]? %i err = %e (%i) J0 = %e ug0 = %e rho0 = %e\n", k, j, i,
                         static_cast<int>(success), err, niter, v(iblock, idx_J(0), k,
                         j, i), v(iblock, peng, k, j, i), v(iblock, prho, k, j, i));
                  exit(-1);
                }
        if (niter == max_iter || err > TOL || std::isnan(U_rad_guess[0]) ||
            std::isnan(U_rad_guess[1]) || std::isnan(U_rad_guess[2]) ||
            std::isnan(U_rad_guess[3])) {
          success = false;
          printf("[%i %i %i] FAILURE niter = %i err = %e U_rad_guess: %e %e %e %e\n", k,
                 j, i, niter, err, U_rad_guess[0], U_rad_guess[1], U_rad_guess[2],
                 U_rad_guess[3]);
          //          exit(-1);
          if (i == izone && j == jzone) {
            exit(-1);
          }
        } else {
          success = true;
          dE = (U_rad_guess[0] - v(iblock, idx_E(ispec), k, j, i)) / sdetgam;
          SPACELOOP(ii) {
            cov_dF(ii) =
                (U_rad_guess[ii + 1] - v(iblock, idx_F(ispec, ii), k, j, i)) / sdetgam;
          }
          //        if (i == 64 && j == 64) {
          //         printf("first: dE: %e cov_dF: %e %e %e\n", dE, cov_dF(0), cov_dF(1),
          //         cov_dF(2));
          //        }
        }

        //        if (i == 5 && j == 4) {
        //          printf("EXIT!\n");
        //          exit(-1);
        //        }

        // FORCING USE OF 1D ROOTFIND!
        // success = false;

        // If 4D rootfind failed, try operator splitting the energy and momentum updates.
        // TODO(BRR) Only do this if J << ug?
        // if (success == false) {
        // Need to use this update if update fluid is false because 4D rootfind updates
        // the fluid state internally
        if (update_fluid == false) {
          const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
          Vec con_v{{con_vp[0] / W, con_vp[1] / W, con_vp[2] / W}};

          Real J0[3] = {v(iblock, idx_J(0), k, j, i), 0, 0};
          PARTHENON_REQUIRE(num_species == 1, "Multiple species not implemented");

          const Real dtau = alpha * dt / W; // Elapsed time in fluid frame

          // Rootfind over fluid temperature in fluid rest frame
          root_find::RootFind root_find;
          InteractionTResidual res(eos, d_opacity, d_mean_opacity, rho, ug, Ye, J0,
                                   num_species, species_d, scattering_fraction, dtau);
          const Real T1 =
              root_find.secant(res, 0, 1.e3 * v(iblock, pT, k, j, i),
                               1.e-8 * v(iblock, pT, k, j, i), v(iblock, pT, k, j, i));

          /// TODO: (LFR) Move beyond Eddington for this update
          ClosureEdd<Vec, Tens2> c(con_v, &g);
          // for (int ispec = 0; ispec < num_species; ++ispec)
          {

            Real Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
            Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam};

            // Treat the Eddington tensor explicitly for now
            Real &J = v(iblock, idx_J(ispec), k, j, i);
            Vec cov_H{{
                J * v(iblock, idx_H(ispec, 0), k, j, i),
                J * v(iblock, idx_H(ispec, 1), k, j, i),
                J * v(iblock, idx_H(ispec, 2), k, j, i),
            }};
            Tens2 con_tilPi;

            c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);

            Real JBB = d_opacity.EnergyDensityFromTemperature(T1, species_d[ispec]);
            Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
                rho, T1, Ye, species_d[ispec]);
            Real tauJ = alpha * dt * (1. - scattering_fraction) * kappa;
            Real tauH = alpha * dt * kappa;

            c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ, tauH, &dE,
                                 &cov_dF);
          }

          if (v(iblock, idx_E(ispec), k, j, i) - sdetgam * dE > 0.) {
            // TODO(BRR) also check H^2 < J?
            success = true;
          } else {
            success = false;
          }
        }

        // If sequence of source update methods was successful, apply update to conserved
        // quantities. If unsuccessful, mark zone for fixup.
        //        if (i == 25 && j == 101) {
        //          printf("sucess? %i\n", success);
        //        }
        if (success == true) {
          // if (i > 60 && i < 80) {
          //  printf("[%i] dE: %e dF: %e %e %e update_fluid: %i\n", i,
          //    dE, cov_dF(0), cov_dF(1), cov_dF(2), update_fluid);
          // }
          v(iblock, ifail, k, j, i) = FailFlags::success;
          //          if (i == 25 && j == 101) {
          //            printf("dE: %e dF: %e %e %e\n", dE, cov_dF(0), cov_dF(1),
          //            cov_dF(2));
          //          }

          v(iblock, idx_E(ispec), k, j, i) += sdetgam * dE;
          SPACELOOP(ii) { v(iblock, idx_F(ispec, ii), k, j, i) += sdetgam * cov_dF(ii); }
          if (update_fluid) {
            if (cye > 0) {
              v(iblock, cye, k, j, i) -= sdetgam * 0.0;
            }
#if USE_VALENCIA
            v(iblock, ceng, k, j, i) -= sdetgam * dE;
#else
            PARTHENON_FAIL("This update for non-Valencia energy eqn is not correct");
            v(iblock, ceng, k, j, i) += alpha * sdetgam * dE;
#endif
            SPACELOOP(ii) { v(iblock, cmom_lo + ii, k, j, i) -= sdetgam * cov_dF(ii); }
          }
        } else {
          // std::stringstream msg;
          // msg << "Source update failure at [" << k << " " << j << " " << i << "]";
          // PARTHENON_FAIL(msg);
          v(iblock, ifail, k, j, i) = FailFlags::fail;
          // TODO(BRR) if failure just dont do source
          // v(iblock, ifail, k, j, i) = FailFlags::success;
        }
      });
  return TaskStatus::complete;
}

// template TaskStatus MomentFluidSource<MeshData<Real>>(MeshData<Real> *, Real, bool);
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real,
                                                           bool);

template <class T>
TaskStatus MomentCalculateOpacities(T *rc) {
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  const bool rad_mocmc_active = (rad->Param<std::string>("method") == "mocmc");
  if (rad_mocmc_active) {
    // For MOCMC, opacities are calculated by averaging over samples in interaction call
    return TaskStatus::complete;
  }

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{p::density, p::temperature, p::ye,  p::velocity,
                                ir::kappaJ, ir::kappaH,     ir::JBB};

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Mainly for testing purposes, probably should be able to do this with the opacity
  // code itself
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  // Get the device opacity object
  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  const auto &d_mean_opacity = opac->Param<MeanOpacity>("d.mean_opacity");

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_kappaJ.DimSize(1);

  /// TODO: (LFR) Fix this junk
  RadiationType dev_species[3] = {species[0], species[1], species[2]};

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::CalculateOpacities", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        for (int ispec = 0; ispec < nspec; ++ispec) {
          const Real rho = v(iblock, prho, k, j, i);
          const Real Temp = v(iblock, pT, k, j, i);
          const Real Ye = pYe > 0 ? v(iblock, pYe, k, j, i) : 0.5;

          Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
              rho, Temp, Ye, dev_species[ispec]);
          Real JBB = d_opacity.EnergyDensityFromTemperature(Temp, dev_species[ispec]);

          v(iblock, idx_JBB(ispec), k, j, i) = JBB;
          v(iblock, idx_kappaJ(ispec), k, j, i) = kappa * (1.0 - scattering_fraction);
          v(iblock, idx_kappaH(ispec), k, j, i) = kappa;
        }
      });

  return TaskStatus::complete;
}
template TaskStatus MomentCalculateOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *);
} // namespace radiation
