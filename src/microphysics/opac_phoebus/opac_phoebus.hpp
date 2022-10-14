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

#ifndef MICROPHYSICS_OPAC_OPAC_HPP_
#define MICROPHYSICS_OPAC_OPAC_HPP_

#include <memory>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

// singularity includes
#include <singularity-opac/base/radiation_types.hpp>
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/mean_s_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>
#include <singularity-opac/neutrinos/s_opac_neutrinos.hpp>

namespace Microphysics {

using RadiationType = singularity::RadiationType;

/// One class to contain all opacity objects and wrap the subset of calls we use in
/// phoebus for convenience.
// TODO(BRR) template on opacity/scattering opacity being gray, and if they are constexpr if-away
// the table lookups for mean opacities
class Opacities {
  using Opacity = singularity::neutrinos::Opacity;
  using MeanOpacity = singularity::neutrinos::MeanOpacity;
  using SOpacity = singularity::neutrinos::SOpacity;
  using MeanSOpacity = singularity::neutrinos::MeanSOpacity;

 public:
  Opacities(const Opacity &opac, const MeanOpacity &m_opac, const SOpacity &s_opac,
            const MeanSOpacity &m_s_opac)
      : opac_(opac), m_opac_(m_opac), s_opac_(s_opac), m_s_opac_(m_s_opac) {}

  Real EnergyDensityFromTemperature(const Real &T, const RadiationType &type) const {
    return opac_.EnergyDensityFromTemperature(T, type);
  }

  Real RosselandMeanAbsorptionCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                          const RadiationType &type) const {
    return m_opac_.RosselandMeanAbsorptionCoefficient(rho, T, Ye, type);
  }

  Real RosselandMeanScatteringCoefficient(const Real &rho, const Real &T, const Real &Ye,
                                          const RadiationType &type) const {
    return m_s_opac_.RosselandMeanScatteringCoefficient(rho, T, Ye, type);
  }

 private:
  const Opacity &opac_;
  const MeanOpacity &m_opac_;
  const SOpacity &s_opac_;
  const MeanSOpacity &m_s_opac_;
};

namespace Opacity {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
} // namespace Opacity

} // namespace Microphysics

#endif // MICROPHYSICS_OPAC_OPAC_HPP_
