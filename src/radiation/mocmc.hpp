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

#include "microphysics/opac_phoebus/opac_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/frequency_info.hpp"
#include "radiation/radiation.hpp"

#ifndef RADIATION_MOCMC_HPP_
#define RADIATION_MOCMC_HPP_

namespace radiation {

template <typename MBD, typename VP, typename CLOSURE>
class MOCMCInteractions {
  using FlatIdx = parthenon::vpack_types::FlatIdx;
  using Opacities = Microphysics::Opacities;
  using RadiationType = singularity::RadiationType;

 public:
  /// Load MOCMC swarm and relevant swarm data if MOCMC is being used. Otherwise don't
  /// do anything. Also assume that a VariablePack with Inu0 and Inu1 already exists --
  /// just pass those in directly
  MOCMCInteractions(const MBD *rc, VP &v, const FlatIdx &idx_Inu0,
                    const FlatIdx &idx_Inu1, const FrequencyInfo &freq_info,
                    const int &num_species,
                    const RadiationType species[MaxNumRadiationSpecies])
      : rc_(rc), v_(v), idx_Inu0_(idx_Inu0), idx_Inu1_(idx_Inu1), freq_info_(freq_info),
        num_species_(num_species) {

    if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
      auto *pmb = rc->GetParentPointer().get();
      auto &swarm = pmb->swarm_data.Get()->Get("mocmc");

      ncov_ = swarm->template Get<Real>("ncov").Get();
      mu_lo_ = swarm->template Get<Real>("mu_lo").Get();
      mu_hi_ = swarm->template Get<Real>("mu_hi").Get();
      phi_lo_ = swarm->template Get<Real>("phi_lo").Get();
      phi_hi_ = swarm->template Get<Real>("phi_hi").Get();
      Inuinv_ = swarm->template Get<Real>("Inuinv").Get();
      swarm_d_ = swarm->GetDeviceContext();

      for (int ispec = 0; ispec < num_species_; ispec++) {
        species_[ispec] = species[ispec];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateInu0(const int iblock, const int ispec, const int k, const int j,
                     const int i, const Real ucon[4]) const {
    if constexpr (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
      for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
        v_(iblock, idx_Inu0_(ispec, nbin), k, j, i) = 0.;
      }

      const int nsamp = swarm_d_.GetParticleCountPerCell(k, j, i);
      const Real nu_fluid0 = freq_info_.GetNuMin();
      for (int n = 0; n < nsamp; n++) {
        const int nswarm = swarm_d_.GetFullIndex(k, j, i, n);
        const Real dOmega = (mu_hi_(nswarm) - mu_lo_(nswarm)) *
                            (phi_hi_(nswarm) - phi_lo_(nswarm)) / (4. * M_PI);
        PARTHENON_DEBUG_REQUIRE(dOmega > 0., "Non-positive dOmega");

        Real nu_lab0 = freq_info_.GetNuMin();
        Real nu_fluid0 = 0.;
        SPACETIMELOOP(nu) { nu_fluid0 -= ncov_(nu, nswarm) * ucon[nu]; }
        nu_fluid0 *= nu_lab0;

        const Real shift =
            (std::log(nu_fluid0) - std::log(nu_lab0)) / freq_info_.GetDLogNu();
        interpolation::PiecewiseConstant interp(freq_info_.GetNumBins(),
                                                freq_info_.GetDLogNu(), shift);
        int nubin_shift[interp.maxStencilSize];
        Real nubin_wgt[interp.maxStencilSize];

        for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
          interp.GetIndicesAndWeights(nbin, nubin_shift, nubin_wgt);
          for (int isup = 0; isup < interp.StencilSize(); isup++) {
            PARTHENON_DEBUG_REQUIRE(
                !std::isnan(Inuinv_(nubin_shift[isup], ispec, nswarm)), "NAN intensity!");
            const Real nu = freq_info_.GetBinCenterNu(nubin_shift[isup]);
            v_(iblock, idx_Inu0_(ispec, nbin), k, j, i) +=
                nubin_wgt[isup] * Inuinv_(nubin_shift[isup], ispec, nswarm) * pow(nu, 3) *
                dOmega;
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void UpdateSampleIntensities(const Real &rho, const Real &T, const Real &Ye,
                               const Real ucon[4], const int &iblock, const int &k,
                               const int &j, const int &i, const Opacities &opacities,
                               const Real &dt) const {
    if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
      const int nsamp = swarm_d_.GetParticleCountPerCell(k, j, i);
      for (int n = 0; n < nsamp; n++) {
        const int nswarm = swarm_d_.GetFullIndex(k, j, i, n);

        Real nu_lab0 = freq_info_.GetNuMin();
        Real nu_fluid0 = 0.;
        SPACETIMELOOP(nu) { nu_fluid0 -= ncov_(nu, nswarm) * ucon[nu]; }
        nu_fluid0 *= nu_lab0;

        // TODO(BRR) check sign of shift
        const Real shift =
            (std::log(nu_fluid0) - std::log(nu_lab0)) / freq_info_.GetDLogNu();
        interpolation::PiecewiseConstant interp(freq_info_.GetNumBins(),
                                                freq_info_.GetDLogNu(), -shift);
        int nubin_shift[interp.maxStencilSize];
        Real nubin_wgt[interp.maxStencilSize];

        for (int ispec = 0; ispec < num_species_; ispec++) {
          for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
            const Real nu_lab = freq_info_.GetBinCenterNu(nbin);
            const Real nu_fluid =
                std::exp(std::log(nu_lab) - freq_info_.GetDLogNu() * shift);
            const Real ds = dt / ucon[0] / nu_fluid;
            // TODO(BRR) is nu_fluid correct here? We want the nu of the sample bin
            // (n.u) * nufluid(n) in the fluid frame
            const Real alphainv_a =
                nu_fluid * opacities.AngleAveragedAbsorptionCoefficient(
                               rho, T, Ye, species_[ispec], nu_fluid);
            const Real jinv_a =
                opacities.ThermalDistributionOfTNu(T, species_[ispec], nu_fluid) /
                (nu_fluid * nu_fluid * nu_fluid) * alphainv_a;

            // TODO(BRR) actually include at least elastic scattering
            const Real alphainv_s = 0.;
            const Real jinv_s = 0.;

            Inuinv_(nbin, ispec, nswarm) =
                (Inuinv_(nbin, ispec, nswarm) + ds * (jinv_a + jinv_s)) /
                (1. + ds * (alphainv_a + alphainv_s));

            PARTHENON_DEBUG_REQUIRE(!std::isnan(Inuinv_(nbin, ispec, nswarm)),
                                    "NAN intensity!");
          }
        }
      }
    }
  }

 private:
  const MBD *rc_;
  const VP &v_;
  const FlatIdx &idx_Inu0_;
  const FlatIdx &idx_Inu1_;
  const FrequencyInfo &freq_info_;
  const int num_species_;
  RadiationType species_[MaxNumRadiationSpecies];
  // TODO(BRR) replace with a pack
  ParArrayND<Real> ncov_;
  ParArrayND<Real> mu_lo_;
  ParArrayND<Real> mu_hi_;
  ParArrayND<Real> phi_lo_;
  ParArrayND<Real> phi_hi_;
  ParArrayND<Real> Inuinv_;
  SwarmDeviceContext swarm_d_;
};

} // namespace radiation

#endif // RADIATION_MOCMC_HPP_
