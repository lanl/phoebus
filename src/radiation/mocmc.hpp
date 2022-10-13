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

 public:
  /// Load MOCMC swarm and relevant swarm data if MOCMC is being used. Otherwise don't
  /// do anything. Also assume that a VariablePack with Inu0 and Inu1 already exists --
  /// just pass those in directly
  MOCMCInteractions(const MBD *rc, VP &v, const FlatIdx &idx_Inu0,
                    const FlatIdx &idx_Inu1, const FrequencyInfo &freq_info)
      : rc_(rc), v_(v), idx_Inu0_(idx_Inu0), idx_Inu1_(idx_Inu1), freq_info_(freq_info) {

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
//      printf("got ncov!\n");
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CalculateInu0(const int iblock, const int ispec, const int k, const int j,
                     const int i, const Real ucon[4]) const {
    if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
      //printf("num bins: %i\n", freq_info_.GetNumBins());
      for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
        v_(iblock, idx_Inu0_(ispec, nbin), k, j, i) = 0.;
      }

      const int nsamp = swarm_d_.GetParticleCountPerCell(k, j, i);
      const Real nu_fluid0 = freq_info_.GetNuMin();
      //printf("ucon: %e %e %e %e\n", ucon[0], ucon[1], ucon[2], ucon[3]);
      for (int n = 0; n < nsamp; n++) {
        const int nswarm = swarm_d_.GetFullIndex(k, j, i, n);
        const Real dOmega = (mu_hi_(nswarm) - mu_lo_(nswarm)) *
                            (phi_hi_(nswarm) - phi_lo_(nswarm)) / (4. * M_PI);

        Real nu_lab0 = 0.;
        SPACETIMELOOP(nu) {
          nu_lab0 -= ncov_(nu, nswarm) * ucon[nu];
        }
        nu_lab0 *= nu_fluid0;

        const Real shift = (std::log(nu_fluid0) - std::log(nu_lab0)) / freq_info_.GetDLogNu();
        interpolation::PiecewiseConstant interp(freq_info_.GetNumBins(), freq_info_.GetDLogNu(), shift);
        int nubin_shift[interp.maxStencilSize];
        Real nubin_wgt[interp.maxStencilSize];

//        printf("nu_lab0: %e nu_fluid0: %e\n", nu_lab0, nu_fluid0);
//        printf("[%i] ncov = %e %e %e %e shift = %e\n",
//          n, ncov_(0, nswarm), ncov_(1, nswarm), ncov_(2, nswarm), ncov_(3, nswarm),
//          shift);


        // TODO(BRR) interpolate onto correct frequency grid!
        for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
          v_(iblock, idx_Inu0_(ispec, nbin), k, j, i) +=
              Inuinv_(nbin, ispec, nswarm) * pow(freq_info_.GetBinCenterNu(nbin), 3) *
              dOmega;
          interp.GetIndicesAndWeights(nbin, nubin_shift, nubin_wgt);
          for (int isup = 0; isup < interp.StencilSize(); isup++) {
            v_(iblock, idx_Inu0_(ispec, nbin), k, j, i) +=
              nubin_wgt[isup] * Inuinv_(nubin_shift[isup], ispec, nswarm);
          }
        }
      }

    //  for (int nbin = 0; nbin < freq_info_.GetNumBins(); nbin++) {
        //printf("[%i] Inu = %e\n", nbin, v_(iblock, idx_Inu0_(ispec, nbin), k, j, i));
     // }
    }
  }

 private:
  const MBD *rc_;
  const VP &v_;
  const FlatIdx &idx_Inu0_;
  const FlatIdx &idx_Inu1_;
  const FrequencyInfo &freq_info_;
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
