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

//#include "fixup/fixup.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/root_find.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"

namespace radiation {

using namespace singularity::neutrinos;
using fixup::Bounds;
using singularity::EOS;

template <typename CLOSURE>
class SourceResidual4 {
 public:
  KOKKOS_FUNCTION
  SourceResidual4(const EOS &eos, const Opacity &opac, const MeanOpacity &mopac,
                  const Real rho, const Real Ye, const Real bprim[3],
                  const RadiationType species, /*const*/ Tens2 &conTilPi,
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
    // Use gamma_max from code?
    if (W > 100) {
      printf("W = %e! [%i %i %i]\n", W, k_, j_, i_);
      return ClosureStatus::failure;
    }
    CLOSURE c(con_v, &g_);
    // TODO(BRR) Accept separately calculated con_tilPi as an option
    // TODO(BRR) Store xi, phi guesses
    Real xi;
    Real phi;
    c.GetCovTilPiFromCon(E, cov_F, xi, phi, &conTilPi_);
    auto status = c.Con2Prim(E, cov_F, conTilPi_, &(P_rad[0]), &cov_H);
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
  const Opacity &opac_;
  const MeanOpacity &mopac_;
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
  Real J0_[MaxNumRadiationSpecies];
  const int &nspec_;
  const Real &scattering_fraction_;
  const Real &dtau_; // Proper time
  RadiationType species_[MaxNumRadiationSpecies];
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
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  std::vector<std::string> variables{
      cr::E,  cr::F,   pr::J,       pr::H,    fluid_prim::velocity,
      ir::xi, ir::phi, ir::c2pfail, ir::tilPi};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);
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

        if (iTilPi.IsValid()) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          Real xi = 0.0;
          Real phi = pi;
          // TODO(BRR) Remove STORE_GUESS parameter and instead check if closure type is
          // M1?
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

        v(b, pJ(ispec), k, j, i) = J;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          // Use the scaled value of the rest frame flux for reconstruction
          v(b, pH(ispec, idir), k, j, i) = robust::ratio(covH(idir), J);
        }

//        if (i >= 128 && j == 118) {
//          printf("[%i %i %i] c2p: %e %e %e %e success? %i\n", k, j, i,
//                 v(b, pJ(ispec), k, j, i),
//                 v(b, pH(ispec, 0), k, j, i) * v(b, pJ(ispec), k, j, i),
//                 v(b, pH(ispec, 1), k, j, i) * v(b, pJ(ispec), k, j, i),
//                 v(b, pH(ispec, 2), k, j, i) * v(b, pJ(ispec), k, j, i),
//                 static_cast<int>(status == ClosureStatus::success));
//        }

        v(b, ifail, k, j, i) =
            (status == ClosureStatus::success ? FailFlags::success : FailFlags::fail);
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentCon2Prim(T *rc) {
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
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);

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
        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);

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

        if (iTilPi.IsValid()) {
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

template <class T>
TaskStatus ReconstructEdgeStates(T *rc) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  using namespace PhoebusReconstruction;

  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);

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
  std::vector<std::string> variables = {pr::J, pr::H, ir::tilPi};
  variables.push_back(ir::dJ);
  VariablePack<Real> v = rc->PackVariables(variables, imap);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);

  ParArrayND<Real> ql_v = rc->Get(ir::ql_v).data;
  ParArrayND<Real> qr_v = rc->Get(ir::qr_v).data;
  VariablePack<Real> v_vel =
      rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
  auto qIdx = imap_ql.GetFlatIdx(ir::ql);

  const int nspec = qIdx.DimSize(1);
  int nrecon = 4 * nspec;
  if (iTilPi.IsValid()) {
    nrecon = (4 + 9) * nspec; // TODO(BRR) 6 instead of 9 for conTilPi by symmetry
  }

  const int offset = imap_ql[ir::ql].first;

  const int nblock = ql_base.GetDim(5);
  const int ndim = pmb->pmy_mesh->ndim;
  auto &coords = pmb->coords;

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
          if (n == 0) {
          }
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
        ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvim1, pv, pvip1, vi_l,
                                   vi_r);
//        if (member.team_rank() == 0 && n == 0 && j == 118) {
//          for (int i = ib.e - 5; i <= ib.e + 1; i++) {
//            printf("[%i %i %i] (J) vi_l = %e vi_r = %e\n",
//              k, j, i, vi_l[i], vi_r[i]);
//            }
//        }
        // y-direction
        if (ndim > 1)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvjm1, pv, pvjp1, vj_l, vj_r);

        // z-direction
        if (ndim > 2)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvkm1, pv, pvkp1, vk_l, vk_r);

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
//          for (int i = 128; i <= ib.e + 4; i++) {
//          if (member.team_rank() == 0 && n == 0 && j == 118 && i > 128) {
//            printf("[%i %i %i] J recon: %e\n", k,j,i,v(b, idx_J(n), k, j, i));
//          }
//          }
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
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad_pkg = pmb->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();

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

  auto num_species = rad_pkg->Param<int>("num_species");

  // const int nblock = 1; //v.GetDim(5);

  auto geom = Geometry::GetCoordinateSystem(rc);

  auto bounds = fix_pkg->Param<Bounds>("bounds");

  // TODO(BRR) add to radiation floors
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
        Real X[4];
        X[1] = (face == CellLocation::Face1 ? coords.x1f(k, j, i) : coords.x1v(k, j, i));
        X[2] = (face == CellLocation::Face2 ? coords.x2f(k, j, i) : coords.x2v(k, j, i));
        X[3] = (face == CellLocation::Face3 ? coords.x3f(k, j, i) : coords.x3v(k, j, i));

        Real xi_max;
        // bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
        //                            coords.x3v(k, j, i), xi_max);
        xi_max = 0.99;
        // TODO(BRR) compilaiton bug?

        Vec con_beta;
        Tens2 cov_gamma;
        geom.Metric(face, k, j, i, cov_gamma.data);
        geom.ContravariantShift(face, k, j, i, con_beta.data);
        const Real sdetgam = geom.DetGamma(face, k, j, i);
        Tens2 con_gamma;
        geom.MetricInverse(face, k, j, i, con_gamma.data);
        const Real alpha = geom.Lapse(face, k, j, i);
        typename CLOSURE::LocalGeometryType g(geom, face, 0, k, j, i);

        const Real dx = coords.Dx(idir_in, k, j, i) * sqrt(cov_gamma(idir, idir));

        Real con_vpl[3] = {v(idx_qlv(0, idir), k, j, i),
                           v(idx_qlv(1, idir), k, j, i),
                           v(idx_qlv(2, idir), k, j, i)};
        Real con_vpr[3] = {v(idx_qrv(0, idir), k, j, i),
                           v(idx_qrv(1, idir), k, j, i),
                           v(idx_qrv(2, idir), k, j, i)};
        const Real Wl = phoebus::GetLorentzFactor(con_vpl, cov_gamma.data);
        const Real Wr = phoebus::GetLorentzFactor(con_vpr, cov_gamma.data);

        Real Wceiling, garbage;
        bounds.GetCeilings(X[1], X[2], X[3], Wceiling, garbage);
        if (Wl > Wceiling) {
          const Real rescale = std::sqrt(Wceiling * Wceiling - 1.) / (Wl * Wl - 1.);
          SPACELOOP(ii) {
            con_vpl[ii] *= rescale;
          }
        }
        if (Wr > Wceiling) {
          const Real rescale = std::sqrt(Wceiling * Wceiling - 1.) / (Wr * Wr - 1.);
          SPACELOOP(ii) {
            con_vpr[ii] *= rescale;
          }
        }

        Vec con_vl{{con_vpl[0] / Wl, con_vpl[1] / Wl, con_vpl[2] / Wl}};
        Vec con_vr{{con_vpr[0] / Wr, con_vpr[1] / Wr, con_vpr[2] / Wr}};

        // Signal speeds (assume (i.e. somewhat overestimate, esp. for large opt. depth)
        // cs_rad = 1)
        const Real sigp = alpha * std::sqrt(con_gamma(idir, idir)) - con_beta(idir);
        const Real sigm = -alpha * std::sqrt(con_gamma(idir, idir)) - con_beta(idir);
        const Real asym_sigl = alpha * con_vl(idir) - con_beta(idir);
        const Real asym_sigr = alpha * con_vr(idir) - con_beta(idir);

        // TODO(BRR) implement
        //Real Jfloor;
        //bounds.GetRadiationFloors(X[1], X[2], X[3], Jfloor);

        for (int ispec = 0; ispec < num_species; ++ispec) {
          // TODO(BRR) Use floors
          const Real Jl = std::max<Real>(v(idx_ql(ispec, 0, idir), k, j, i), 1.e-10);
          const Real Jr = std::max<Real>(v(idx_qr(ispec, 0, idir), k, j, i), 1.e-10);
//          if (j == 118 && i > 128) {
//          printf("Jl = %e (ispec: %i idir: %i) (%i %i %i %i)\n", Jl, ispec, idir, idx_ql(ispec, 0, idir), k, j, i);
//          }
          Vec Hl = {Jl * v(idx_ql(ispec, 1, idir), k, j, i),
                    Jl * v(idx_ql(ispec, 2, idir), k, j, i),
                    Jl * v(idx_ql(ispec, 3, idir), k, j, i)};
          Vec Hr = {Jr * v(idx_qr(ispec, 1, idir), k, j, i),
                    Jr * v(idx_qr(ispec, 2, idir), k, j, i),
                    Jr * v(idx_qr(ispec, 3, idir), k, j, i)};
          // TODO(BRR) Why does clamping xi here cause explosions?
          Real xil = std::sqrt(g.contractCov3Vectors(Hl, Hl)) / Jl;
          Real xir = std::sqrt(g.contractCov3Vectors(Hr, Hr)) / Jr;
          // TODO(BRR) for j = 4, j = 68 zones, xir/xil for idir = 1 are like 1e10 --
          // seems like BCs are not copying data correctly
          if (xil > xi_max) {
            SPACELOOP(ii) { Hl(ii) *= xi_max / xil; }
          }
          if (xir > xi_max) {
            SPACELOOP(ii) { Hr(ii) *= xi_max / xir; }
          }

          Vec cov_dJ{{v(idx_dJ(ispec, 0, idir), k, j, i),
                      v(idx_dJ(ispec, 1, idir), k, j, i),
                      v(idx_dJ(ispec, 2, idir), k, j, i)}};

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

//          if (j == 118 && i > 128) {
//            printf("[%i %i %i] Jl: %e Hl: %e %e %e Jr: %e Hr: %e %e %e\n",
//              k, j, i, Jl, Hl(0), Hl(1), Hl(2), Jr, Hr(0), Hr(1), Hr(2));
//          }

          // Mix the fluxes by the Peclet number
          // TODO: (LFR) Make better choices
          conFl = a * conFl + (1 - a) * conFl_asym;
          conFr = a * conFr + (1 - a) * conFr_asym;
          Pl = a * Pl + (1 - a) * Pl_asym;
          Pr = a * Pr + (1 - a) * Pr_asym;

          const Real rad_sigspeed = std::max<Real>(std::fabs(sigm), std::fabs(sigp));
          const Real asym_sigspeed =
              std::max<Real>(std::fabs(asym_sigl), std::fabs(asym_sigr));
          const Real sigspeed = a * rad_sigspeed + (1. - a) * asym_sigspeed;

          // Correct the fluxes with the shift terms
          conFl(idir) -= con_beta(idir) * El;
          conFr(idir) -= con_beta(idir) * Er;

          SPACELOOP(ii) {
            Pl(idir, ii) -= con_beta(idir) * covFl(ii);
            Pr(idir, ii) -= con_beta(idir) * covFr(ii);
          }

          // Calculate the numerical flux using LLF
          v.flux(idir_in, idx_Ef(ispec), k, j, i) =
              0.5 * sdetgam * (conFl(idir) + conFr(idir) + sigspeed * (El - Er));

          SPACELOOP(ii) {
            v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) =
                0.5 * sdetgam *
                (Pl(idir, ii) + Pr(idir, ii) + sigspeed * (covFl(ii) - covFr(ii)));
          }
//          if (i == 4 && j == 127) {
//            printf("flux [%i %i %i][%i] = %e %e %e %e a = %e Jl = %e Jr = %e Fl: %e %e "
//                   "%e Fr: %e %e %e kappaH: %e sigspeed: %e\n",
//                   k, j, i, idir, v.flux(idir_in, idx_Ef(ispec), k, j, i),
//                   v.flux(idir_in, idx_Ff(ispec, 0), k, j, i),
//                   v.flux(idir_in, idx_Ff(ispec, 1), k, j, i),
//                   v.flux(idir_in, idx_Ff(ispec, 2), k, j, i), a, Jl, covFl(0), covFl(1),
//                   covFl(2), Jr, covFr(0), covFr(1), covFr(2), kappaH, sigspeed);
//          }
          if (sdetgam < robust::SMALL()) {
            v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.0;
            SPACELOOP(ii) v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) = 0.0;
          }
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus CalculateFluxes(T *rc) {
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
  auto *pmb = rc->GetParentPointer().get();

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace p = fluid_prim;
  PackIndexMap imap;
  std::vector<std::string> vars{cr::E, cr::F, pr::J, pr::H, p::velocity, ir::tilPi};
  vars.push_back(diagnostic_variables::r_src_terms);
#if SET_FLUX_SRC_DIAGS
  vars.push_back(diagnostic_variables::r_src_terms);
#endif
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(p::velocity);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi, false);
  auto idx_diag = imap.GetFlatIdx(diagnostic_variables::r_src_terms, false);

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
        Real con_g[4][4] = {0};
        geom.SpacetimeMetricInverse(CellLocation::Cent, iblock, k, j, i, con_g);
        Real dlnalp[ND];
        Real Gamma[ND][ND][ND];
        geom.GradLnAlpha(CellLocation::Cent, iblock, k, j, i, dlnalp);
        geom.ConnectionCoefficient(CellLocation::Cent, iblock, k, j, i, Gamma);

        Real con_u[4];
        con_u[0] = W / alp;
        SPACELOOP(ii) { con_u[ii + 1] = W * con_v(ii) - con_beta(ii) / alp; }

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
          // Tens2 conP, con_tilPi;

          Real con_H4[4] = {0};
          SPACELOOP2(ii, jj) { con_H4[ii + 1] += g.con_gamma(ii, jj) * covH(jj); }

          Real conTilPi[4][4] = {0};
          if (iTilPi.IsValid()) {
            SPACELOOP2(ii, jj) {
              // con_tilPi(ii, jj) = v(iblock, iTilPi(ispec, ii, jj), k, j, i);
              conTilPi[ii + 1][jj + 1] = v(iblock, iTilPi(ispec, ii, jj), k, j, i);
            }
          } else {
            Tens2 con_tilPi{0};
            c.GetCovTilPiFromPrim(J, covH, &con_tilPi);
            SPACELOOP2(ii, jj) { conTilPi[ii + 1][jj + 1] = con_tilPi(ii, jj); }
          }

          Real con_T[4][4];
          SPACETIMELOOP2(mu, nu) {
            Real conh_munu = con_g[mu][nu] + con_u[mu] * con_u[nu];
            con_T[mu][nu] = (con_u[mu] * con_u[nu] + conh_munu / 3.) * J;
            con_T[mu][nu] += con_u[mu] * con_H4[nu] + con_u[nu] * con_H4[mu];
            con_T[mu][nu] += J * conTilPi[mu][nu];
          }
          // c.getConPFromPrim(J, covH, con_tilPi, &conP);

          // TODO(BRR) add Pij contribution

          Real srcE = 0.0;
          SPACETIMELOOP(mu) {
            srcE += con_T[mu][0] * dlnalp[mu];
            SPACETIMELOOP(nu) {
              Real Gamma_udd = 0.;
              SPACETIMELOOP(lam) { Gamma_udd += con_g[0][lam] * Gamma[lam][nu][mu]; }
              srcE -= con_T[mu][nu] * Gamma_udd;
            }
          }
          srcE *= alp * alp;

          Vec srcF{0, 0, 0};
          SPACELOOP(ii) {
            SPACETIMELOOP2(mu, nu) { srcF(ii) += con_T[mu][nu] * Gamma[mu][nu][ii + 1]; }
            srcF(ii) *= alp;
          }

          v_src(iblock, idx_E_src(ispec), k, j, i) = sdetgam * srcE;
          SPACELOOP(ii) {
            v_src(iblock, idx_F_src(ispec, ii), k, j, i) = sdetgam * srcF(ii);
          }

//          if (j == 127 && i == 4) {
//            printf("geosrc [%i %i %i]: %e %e %e %e\n", k, j, i,
//                   v_src(iblock, idx_E_src(ispec), k, j, i),
//                   v_src(iblock, idx_F_src(ispec, 0), k, j, i),
//                   v_src(iblock, idx_F_src(ispec, 1), k, j, i),
//                   v_src(iblock, idx_F_src(ispec, 2), k, j, i));
//          }
#if SET_FLUX_SRC_DIAGS
          v(iblock, idx_diag(ispec, 0), k, j, i) =
              v_src(iblock, idx_E_src(ispec), k, j, i);
          v(iblock, idx_diag(ispec, 1), k, j, i) =
              v_src(iblock, idx_F_src(ispec, 0), k, j, i);
          v(iblock, idx_diag(ispec, 2), k, j, i) =
              v_src(iblock, idx_F_src(ispec, 1), k, j, i);
          v(iblock, idx_diag(ispec, 3), k, j, i) =
              v_src(iblock, idx_F_src(ispec, 2), k, j, i);
#endif
        }
      });
  return TaskStatus::complete;
}

template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src) {
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

template <class T, class CLOSURE>
TaskStatus MomentFluidSourceImpl(T *rc, Real dt, bool update_fluid) {
  printf("%s:%i:%s\n", __FILE__, __LINE__, __func__);
  PARTHENON_REQUIRE(USE_VALENCIA, "Covariant MHD formulation not supported!");

  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *fluid_pkg = pmb->packages.Get("fluid").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{c::energy, c::momentum, c::ye,       cr::E,
                                cr::F,     c::bfield,   p::density,  p::temperature,
                                p::energy, p::ye,       p::velocity, p::bfield,
                                pr::J,     pr::H,       ir::kappaJ,  ir::kappaH,
                                ir::JBB,   ir::tilPi,   ir::srcfail};

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
  auto ifail = imap[ir::srcfail].first;
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int peng = imap[p::energy].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;
  int pb_lo = imap[p::bfield].first;
  int cb_lo = imap[c::bfield].first;

  int ceng = imap[c::energy].first;
  int cmom_lo = imap[c::momentum].first;
  int cmom_hi = imap[c::momentum].second;
  int cye = imap[c::ye].first;

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

  // const Real c2p_tol = fluid_pkg->Param<Real>("c2p_tol");
  // const int c2p_max_iter = fluid_pkg->Param<int>("c2p_max_iter");
  // auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter);
  auto coords = pmb->coords;

  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  const auto &d_mean_opacity = opac->Param<MeanOpacity>("d.mean_opacity");
  // Mainly for testing purposes, probably should be able to do this with the opacity code
  // itself
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  const auto src_solver = rad->Param<SourceSolver>("src_solver");
  const auto src_use_oned_backup = rad->Param<bool>("src_use_oned_backup");
  const auto src_rootfind_eps = rad->Param<Real>("src_rootfind_eps");
  const auto src_rootfind_tol = rad->Param<Real>("src_rootfind_tol");
  const auto src_rootfind_maxiter = rad->Param<int>("src_rootfind_maxiter");

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
        bounds.GetRadiationCeilings(coords.x1v(k, j, i), coords.x2v(k, j, i),
                                    coords.x3v(k, j, i), xi_max);

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

//         if (j == 118 && i == 131) {
//           int ispec = 0;
//          const Real W = phoebus::GetLorentzFactor(con_vp, cov_gamma.data);
//          Vec Hcov{v(iblock, idx_H(ispec, 0), k, j, i),
//                   v(iblock, idx_H(ispec, 1), k, j, i),
//                   v(iblock, idx_H(ispec, 2), k, j, i)};
//          Real xi = std::sqrt(g.contractCov3Vectors(Hcov, Hcov));
//          printf("Initial [%i %i %i] ug = %e vp = %e %e %e J = %e Hi = %e %e %e W = %e "
//                 "xi = %e\n",
//                 k, j, i, ug, con_vp[0], con_vp[1], con_vp[2],
//                 v(iblock, idx_J(ispec), k, j, i), v(iblock, idx_H(ispec, 0), k, j, i),
//                 v(iblock, idx_H(ispec, 1), k, j, i), v(iblock, idx_H(ispec, 2), k, j,
//                 i), W, xi);
//        }

        // bool success = false;
        // TODO(BRR) These will need to be per-species later
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
            InteractionTResidual res(eos, d_opacity, d_mean_opacity, rho, ug, Ye, J0,
                                     num_species, species_d, scattering_fraction, dtau);
            root_find::RootFindStatus status;
            const Real T1 = root_find.secant(res, 0, 1.e3 * v(iblock, pT, k, j, i),
                                             1.e-8 * v(iblock, pT, k, j, i),
                                             v(iblock, pT, k, j, i), &status);
            if (status == root_find::RootFindStatus::failure) {
              success = false;
              break;
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
              c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);
            }

            Real JBB = d_opacity.EnergyDensityFromTemperature(T1, species_d[ispec]);
            Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
                rho, T1, Ye, species_d[ispec]);
            Real tauJ = alpha * dt * (1. - scattering_fraction) * kappa;
            Real tauH = alpha * dt * kappa;

            c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ, tauH,
                                 &(dE[ispec]), &(cov_dF[ispec]));

            // Check xi
            Estar += dE[ispec];
            SPACELOOP(ii) cov_Fstar(ii) += cov_dF[ispec](ii);
            c.Con2Prim(Estar, cov_Fstar, con_tilPi, &J, &cov_H);
            Real xi = std::sqrt(g.contractCov3Vectors(cov_H, cov_H)) / J;

            if (v(iblock, idx_E(ispec), k, j, i) - sdetgam * dE[ispec] > 0. &&
                xi <= xi_max) {
              // TODO(BRR) also check H^2 < J?
              success = true;
            } else {
              success = false;
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
            c.GetCovTilPiFromPrim(J, covH, &con_tilPi);
          }
          SourceResidual4<CLOSURE> srm(eos, d_opacity, d_mean_opacity, rho, Ye, bprim,
                                       species_d[ispec], con_tilPi, cov_g, con_gamma.data,
                                       alpha, beta, sdetgam, scattering_fraction, g,
                                       U_mhd_0, U_rad_0, k, j, i);

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
                Real xi = 0.;
                SPACELOOP2(ii, jj) {
                  xi += con_gamma(ii, jj) * P_rad_guess[ii + 1] * P_rad_guess[jj + 1];
                }
                xi = std::sqrt(xi) / P_rad_guess[0];

                constexpr Real umin = 1.e-20; // TODO(BRR) use floors
                constexpr Real Jmin = 1.e-20; // TODO(BRR) needs to be a bit smaller than
                                              // Jr floor! otherwise scaling fac is 0

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
                printf("nan resid[%i]! %i %i %i Umhdg: %e %e %e %e dSg: %e %e %e %e\n", n,
                       k, j, i, U_mhd_guess[0], U_mhd_guess[1], U_mhd_guess[2],
                       U_mhd_guess[3], dS_guess[0], dS_guess[1], dS_guess[2],
                       dS_guess[3]);
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

//          if (i == 4 && j == 127) {
//            printf(
//                "[%i %i %i] cons = %e %e %e %e dcons = %e %e %e %e (sdg = %e)\n", k, j, i,
//                v(iblock, idx_E(ispec), k, j, i), v(iblock, idx_F(ispec, 0), k, j, i),
//                v(iblock, idx_F(ispec, 1), k, j, i), v(iblock, idx_F(ispec, 2), k, j, i),
//                dE[ispec] * sdetgam, cov_dF[ispec](0) * sdetgam,
//                cov_dF[ispec](1) * sdetgam, cov_dF[ispec](2) * sdetgam, sdetgam);
//            printf("[%i %i %i] radguess: %e %e %e %e niter = %i err = %e\n", k, j, i,
//                   P_rad_guess[0], P_rad_guess[1], P_rad_guess[2], P_rad_guess[3], niter,
//                   err);
//          }

          // if (j == 64) {
          //  printf("[%i %i %i] rho: %e Tg: %e ug: %e Prad0: %e %e %e %e Pradg: %e %e
          //  %e %e\n", k, j, i, rho, Tg,ug, v(iblock, idx_J(ispec), k, j, i), v(iblock,
          //  idx_F(ispec, 0), k, j, i), v(iblock, idx_F(ispec, 1), k, j, i), v(iblock,
          //  idx_F(ispec, 2), k, j, i), P_rad_guess[0], P_rad_guess[1], P_rad_guess[2],
          //  P_rad_guess[3]);
          //}
          //

          if (success == false) {
            printf("[%i %i %i] src failed err: %e niter: %i rho: %e Pmhd0: %e %e %e %e "
                   "Prad0: %e %e %e %e\n",
                   k, j, i, err, niter, rho, v(iblock, peng, k, j, i),
                   v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                   v(iblock, pv(2), k, j, i), v(iblock, idx_J(ispec), k, j, i),
                   v(iblock, idx_H(ispec, 0), k, j, i),
                   v(iblock, idx_H(ispec, 1), k, j, i),
                   v(iblock, idx_H(ispec, 2), k, j, i));
            // Fall back to oned solver if fourd encounters an issue
            if (src_use_oned_backup) {
              goto oned_solver_begin;
            }
          }
        } // SourceSolver

        if (success == true) {
          for (int ispec = 0; ispec < num_species; ispec++) {
            v(iblock, ifail, k, j, i) = FailFlags::success;

            v(iblock, idx_E(ispec), k, j, i) += sdetgam * dE[ispec];
            SPACELOOP(ii) {
              v(iblock, idx_F(ispec, ii), k, j, i) += sdetgam * cov_dF[ispec](ii);
            }
            if (update_fluid) {
              if (cye > 0) {
                v(iblock, cye, k, j, i) -= sdetgam * 0.0;
              }
              v(iblock, ceng, k, j, i) -= sdetgam * dE[ispec];
              SPACELOOP(ii) {
                v(iblock, cmom_lo + ii, k, j, i) -= sdetgam * cov_dF[ispec](ii);
              }
            }
          }
        } else {
          v(iblock, ifail, k, j, i) = FailFlags::fail;
        }
      });
  return TaskStatus::complete;
}
template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentFluidSourceImpl<T, ClosureM1<Vec, Tens2, settings>>(rc, dt,
                                                                     update_fluid);
  } else if (method == "moment_eddington") {
    return MomentFluidSourceImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc, dt,
                                                                      update_fluid);
  } else if (method == "mocmc") {
    return MomentFluidSourceImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc, dt,
                                                                        update_fluid);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
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
