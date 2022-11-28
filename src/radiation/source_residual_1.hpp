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

#ifndef RADIATION_SOURCE_RESIDUAL_1_
#define RADIATION_SOURCE_RESIDUAL_1_

#include "phoebus_utils/variables.hpp"

#include <singularity-eos/eos/eos.hpp>
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

namespace radiation {

/// Class for performing 1D rootfind to get updated fluid temperature due to absorption
/// opacity.
// TODO(BRR) Add inelastic scattering
template <typename MBD, typename VP, typename CLOSURE>
class SourceResidual1 {
  using Opacity = singularity::neutrinos::Opacity;
  using MeanOpacity = singularity::neutrinos::MeanOpacity;
  using EOS = singularity::EOS;

 public:
  KOKKOS_FUNCTION
  SourceResidual1(const EOS &eos, const CLOSURE closure,
                  const OpacityAverager<CLOSURE> &opac_avgr,
                  const MOCMCInteractions<MBD, VP, CLOSURE> &mocmc_int,
                  const FrequencyInfo &freq_info, const Opacity &opacity,
                  const MeanOpacity &mean_opacity, const Real &rho, const Real &ug0,
                  const Real &Ye, const Real J0[3], const int &nspec,
                  const RadiationType species[3], const Real &scattering_fraction,
                  const Real &dtau, const VarAccessor2D &Inu0, const VarAccessor2D &Inu1,
                  const int &iblock, const int &k, const int &j, const int &i)
      : eos_(eos), closure_(closure), opac_avgr_(opac_avgr), mocmc_int_(mocmc_int),
        freq_info_(freq_info), opacity_(opacity), mean_opacity_(mean_opacity), rho_(rho),
        ug0_(ug0), Ye_(Ye), nspec_(nspec), scattering_fraction_(scattering_fraction),
        dtau_(dtau), Inu0_(Inu0), Inu1_(Inu1), iblock_(iblock), k_(k), j_(j), i_(i) {
    for (int ispec = 0; ispec < nspec; ++ispec) {
      J0_[ispec] = J0[ispec];
      PARTHENON_DEBUG_REQUIRE(!std::isnan(J0_[ispec]) && J0_[ispec] > robust::SMALL(),
                              "Invalid value of J0!");
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

      // Update Inu to n+1 if using MOCMC
      if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
        for (int n = 0; n < freq_info_.GetNumBins(); n++) {
          const Real nu = freq_info_.GetBinCenterNu(n);
          const Real Jnu =
              opacity_.EmissivityPerNu(rho_, T, Ye_, species_[ispec], nu) / (4. * M_PI);
          const Real kappaJ = opacity_.AngleAveragedAbsorptionCoefficient(
              rho_, T, Ye_, species_[ispec], nu);
          Inu1_(ispec, n) = (nu * Jnu + robust::ratio(Inu0_(ispec, n), dtau_)) /
                            (nu * kappaJ + robust::ratio(1., dtau_));
        }
      }

      Real kappaJ = opac_avgr_.GetAveragedAbsorptionOpacity(rho_, T, Ye_, ispec);
      kappaJ = (1. - scattering_fraction_) * kappaJ;
      // TODO(BRR) remove scattering_fraction

      const Real JBB = opacity_.EnergyDensityFromTemperature(T, species_[ispec]);

      dJ_tot += (J0_[ispec] + dtau_ * kappaJ * JBB) / (1. + dtau_ * kappaJ) - J0_[ispec];
    }

    const Real ug1 = rho_ * eos_.InternalEnergyFromDensityTemperature(rho_, T, lambda);

    const Real residual = ((ug1 - ug0_) + (dJ_tot)) / (ug0_ + J0_tot);
    PARTHENON_DEBUG_REQUIRE(!std::isnan(residual), "NAN residual!");

    return residual;
  }

 private:
  const EOS &eos_;
  const CLOSURE &closure_;
  const OpacityAverager<CLOSURE> &opac_avgr_;
  const MOCMCInteractions<MBD, VP, CLOSURE> &mocmc_int_;
  const FrequencyInfo &freq_info_;
  const Opacity &opacity_;
  const MeanOpacity &mean_opacity_;
  const Real &rho_;
  const Real &ug0_;
  const Real &Ye_;
  Real J0_[MaxNumRadiationSpecies];
  const int &nspec_;
  const Real &scattering_fraction_;
  const Real &dtau_; // Proper time
  const VarAccessor2D &Inu0_;
  const VarAccessor2D &Inu1_;
  RadiationType species_[MaxNumRadiationSpecies];
  const int &iblock_;
  const int &k_;
  const int &j_;
  const int &i_;
};

} // namespace radiation

#endif // RADIATION_SOURCE_RESIDUAL_1_
