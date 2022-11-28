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

#include "closure.hpp"
#include "closure_m1.hpp"
#include "closure_mocmc.hpp"

#include "radiation.hpp"

#ifndef RADIATION_OPACITY_AVERAGER_
#define RADIATION_OPACITY_AVERAGER_

namespace radiation {

template <class CLOSURE>
class OpacityAverager {
  using Opacities = Microphysics::Opacities;

 public:
  OpacityAverager(const CLOSURE &c, const FrequencyInfo &freq, const Opacities &opacities,
                  const parthenon::VariablePack<Real> &v,
                  const parthenon::vpack_types::FlatIdx &idx_Inu,
                  const singularity::RadiationType *species, const int iblock,
                  const int k, const int j, const int i)
      : c_(c), freq_(freq), opacities_(opacities), v_(v), idx_Inu_(idx_Inu), b_(iblock),
        k_(k), j_(j), i_(i) {
    for (int ispec = 0; ispec < MaxNumRadiationSpecies; ispec++) {
      species_[ispec] = species[ispec];
    }
  }

  Real GetAveragedAbsorptionOpacity(const Real rho, const Real T, const Real ye,
                                    const int ispec) const {
    printf("idx_Inu.IsValid(): %i\n", static_cast<int>(idx_Inu_.IsValid()));
    if constexpr (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {

      for (int n = 0; n < freq_.GetNumBins(); n++) {
        printf("[%i] nu: %e Inu: %e\n", n, freq_.GetBinCenterNu(n),
               v_(b_, idx_Inu_(ispec, n), k_, j_, i_));
      }

      Real kappaJ_num = 0.;
      Real kappaJ_denom = 0.;
      Real nu;
      for (int n = 0; n < freq_.GetNumBins(); n++) {
        // Trapezoidal rule
        const Real fac = (n == 0 || n == freq_.GetNumBins() - 1) ? 0.5 : 1.;
        nu = freq_.GetBinCenterNu(n);
        kappaJ_num += fac *
                      opacities_.AngleAveragedAbsorptionCoefficient(rho, T, ye,
                                                                    species_[ispec], nu) *
                      v_(b_, idx_Inu_(ispec, n), k_, j_, i_) * nu;
        kappaJ_denom += fac * v_(b_, idx_Inu_(ispec, n), k_, j_, i_) * nu;
      }

      printf("kappaJ: %e Planck: %e\n", robust::ratio(kappaJ_num, kappaJ_denom),
             opacities_.RosselandMeanAbsorptionCoefficient(rho, T, ye, species_[ispec]));

      // PARTHENON_FAIL("MOCMC!");
      // Handle case of zero particles in zone i.e. Inu = 0
      if (kappaJ_num < robust::SMALL()) {
        return mean_opac_.RosselandMeanAbsorptionCoefficient(rho, T, ye, species_[ispec]);
      } else {
        return robust::ratio(kappaJ_num, kappaJ_denom);
      }
    } else {
      return opacities_.RosselandMeanAbsorptionCoefficient(rho, T, ye, species_[ispec]);
    }
  }

 private:
  const CLOSURE &c_;
  const FrequencyInfo &freq_;

 public:
  const Opacities &opacities_;

 private:
  const parthenon::VariablePack<Real> &v_;
  const parthenon::vpack_types::FlatIdx &idx_Inu_;
  const int b_;
  const int k_;
  const int j_;
  const int i_;
  singularity::RadiationType species_[MaxNumRadiationSpecies];
};

} // namespace radiation

#endif // RADIATION_OPACITY_AVERAGER_
