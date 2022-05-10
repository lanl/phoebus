// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
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

// system includes
#include <memory>
#include <string>
#include <vector>

// parthenon includes
#include <parthenon/package.hpp>

// singularity includes
#include <singularity-opac/neutrinos/mean_opacity_neutrinos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

// phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/unit_conversions.hpp"

namespace Microphysics {
namespace Opacity {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using namespace singularity::neutrinos;

  auto pkg = std::make_shared<StateDescriptor>("opacity");
  Params &params = pkg->AllParams();

  bool do_rad = pin->GetBoolean("physics", "rad");
  if (!do_rad) {
    return pkg;
  }

  const std::string radiation_method = pin->GetString("radiation", "method");
  std::set<std::string> gray_methods = {"moment_eddington", "moment_m1"};
  bool do_mean_opacity = false;
  if (gray_methods.count(radiation_method)) {
    do_mean_opacity = true;
  }

  const std::string block_name = "opacity";

  auto unit_conv = phoebus::UnitConversions(pin);
  double time_unit = unit_conv.GetTimeCodeToCGS();
  double mass_unit = unit_conv.GetMassCodeToCGS();
  double length_unit = unit_conv.GetLengthCodeToCGS();
  double temp_unit = unit_conv.GetTemperatureCodeToCGS();

  std::string opacity_type = pin->GetString(block_name, "type");
  std::set<std::string> known_opacity_types = {"scalefree", "tophat", "gray", "tabular"};
  if (!known_opacity_types.count(opacity_type)) {
    std::stringstream msg;
    msg << "Opacity model \"" << opacity_type << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  PARTHENON_REQUIRE(
      !(opacity_type == "scalefree" && !unit_conv.IsScaleFree()),
      "Scale free opacity only supported for scale-free phoebus simulations!");

  params.Add("type", opacity_type);

  bool scale_free = false;
  if (opacity_type == "scalefree") {
    scale_free = true;

    const Real kappa = pin->GetReal("opacity", "gray_kappa");
    params.Add("gray_kappa", kappa);

    singularity::neutrinos::Opacity opacity_host = ScaleFree(kappa);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);

  } else if (opacity_type == "tophat") {
    const Real C = pin->GetReal("opacity", "tophat_C");
    const Real numin = pin->GetReal("opacity", "tophat_numin");
    const Real numax = pin->GetReal("opacity", "tophat_numax");
    params.Add("C", C);
    params.Add("numin", numin);
    params.Add("numax", numax);

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<Tophat>(
        Tophat(C, numin, numax), time_unit, mass_unit, length_unit, temp_unit);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "gray") {
    const Real kappa = pin->GetReal("opacity", "gray_kappa");
    params.Add("gray_kappa", kappa);

    singularity::neutrinos::Opacity opacity_host =
        NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit, temp_unit);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "tabular") {
#ifdef SPINER_USE_HDF
    const std::string filename = pin->GetString("opacity", "filename");
    params.Add("filename", filename);

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<SpinerOpac>(
        SpinerOpac(filename), time_unit, mass_unit, length_unit, temp_unit);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
#else
    PARTHENON_FAIL("Tabular opacities requested but HDF5 is disabled!");
#endif
  }

  if (do_mean_opacity) {
    auto opacity_host = params.Get<singularity::neutrinos::Opacity>("h.opacity");
    const Real YeMin = pin->GetOrAddReal("mean_opacity", "yemin", 0.1);
    const Real YeMax = pin->GetOrAddReal("mean_opacity", "yemax", 0.5);
    const int NYe = pin->GetOrAddInteger("mean_opacity", "nye", 10);
    if (scale_free) {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(0.1));
      const Real lRhoMax = pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(10.));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(0.1));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(10.));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      MeanOpacity mean_opac_host = MeanOpacityScaleFree(
          opacity_host, lRhoMin, lRhoMax, NRho, lTMin, lTMax, NT, YeMin, YeMax, NYe);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_opacity", mean_opac_host);
      params.Add("d.mean_opacity", mean_opac_device);
    } else {
      const Real lRhoMin = pin->GetOrAddReal("mean_opacity", "lrhomin", std::log10(1.e5));
      const Real lRhoMax =
          pin->GetOrAddReal("mean_opacity", "lrhomax", std::log10(1.e14));
      const int NRho = pin->GetOrAddInteger("mean_opacity", "nrho", 10);
      const Real lTMin = pin->GetOrAddReal("mean_opacity", "ltmin", std::log10(1.e5));
      const Real lTMax = pin->GetOrAddReal("mean_opacity", "ltmax", std::log10(1.e12));
      const int NT = pin->GetOrAddInteger("mean_opacity", "nt", 10);
      MeanOpacity mean_opac_host = MeanOpacityCGS(opacity_host, lRhoMin, lRhoMax, NRho,
                                                  lTMin, lTMax, NT, YeMin, YeMax, NYe);
      auto mean_opac_device = mean_opac_host.GetOnDevice();
      params.Add("h.mean_opacity", mean_opac_host);
      params.Add("d.mean_opacity", mean_opac_device);
    }
  }

  return pkg;
}

} // namespace Opacity
} // namespace Microphysics
