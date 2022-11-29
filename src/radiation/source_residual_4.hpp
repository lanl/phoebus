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

#ifndef RADIATION_SOURCE_RESIDUAL_4_
#define RADIATION_SOURCE_RESIDUAL_4_

#include "fluid/prim2con.hpp"

namespace radiation {

template <typename CLOSURE>
class SourceResidual4 {
  using EOS = singularity::EOS;
  using Opacities = Microphysics::Opacities;

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
    Real Pg = eos_.PressureFromDensityInternalEnergy(rho_, robust::ratio(P_mhd[0], rho_),
                                                     lambda_);
    Real gam1 = robust::ratio(eos_.BulkModulusFromDensityInternalEnergy(
                                  rho_, robust::ratio(P_mhd[0], rho_), lambda_),
                              Pg);
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

} // namespace radiation

#endif // RADIATION_SOURCE_RESIDUAL_4_
