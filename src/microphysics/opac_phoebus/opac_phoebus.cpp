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

  const std::string block_name = "opacity";
  
  auto unit_conv = phoebus::UnitConversions(pin);
  double time_unit = unit_conv.GetTimeCodeToCGS();
  double mass_unit = unit_conv.GetMassCodeToCGS();
  double length_unit = unit_conv.GetLengthCodeToCGS();
  double temp_unit = unit_conv.GetTemperatureCodeToCGS();

  std::string opacity_type = pin->GetString(block_name, "type");
  std::vector<std::string> known_opacity_types = {"tophat", "gray", "tabular"};
  if (std::find(known_opacity_types.begin(), known_opacity_types.end(),
                opacity_type) == known_opacity_types.end()) {
    std::stringstream msg;
    msg << "Opacity model \"" << opacity_type << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  if (opacity_type == "tophat") {
    const Real C = pin->GetReal("opacity", "tophat_C");
    const Real numin = pin->GetReal("opacity", "tophat_numin");
    const Real numax = pin->GetReal("opacity", "tophat_numax");

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<Tophat>(Tophat(C, numin, numax), time_unit, mass_unit, length_unit, temp_unit);
    singularity::neutrinos::Opacity opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "gray") {
    const Real kappa = pin->GetReal("opacity", "gray_kappa");

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<Gray>(Gray(kappa), time_unit, mass_unit, length_unit, temp_unit);
    singularity::neutrinos::Opacity opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "tabular") {
    const std::string filename = pin->GetString("opacity", "filename");

    singularity::neutrinos::Opacity opacity_host = NonCGSUnits<SpinerOpacity>(SpinerOpacity(filename), time_unit, mass_unit, length_unit, temp_unit);
    singularity::neutrinos::Opacity opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  }

  return pkg;
}

} // namespace Opacity
} // namespace Microphysics
