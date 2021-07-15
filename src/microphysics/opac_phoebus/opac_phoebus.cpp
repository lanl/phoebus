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
  auto pkg = std::make_shared<StateDescriptor>("eos");
  Params &params = pkg->AllParams();

  const std::string block_name = "opacity";

  std::vector<std::string> known_opacity_models = {"tophat", "gray"};
  if (std::find(known_opacity_models.begin(), known_opacity_models.end(),
                opacity_model) == known_opacity_models.end()) {
    std::stringstream msg;
    msg << "Opacity model \"" << opacity_model << "\" not recognized!";
    PARTHENON_FAIL(msg);

  std::string opacity_type = pin->GetString(block_name, "type");
  if (opacity_type == "tophat") {
    const Real C = pin->GetReal("opacity", "tophat_C");
    const Real numin = pin->GetReal("opacity", "tophat_numin");
    const Real numax = pin->GetReal("opacity", "tophat_numax");

    auto opacity_host = Tophat(C, numin, numax);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  } else if (opacity_type == "gray") {
    const Real kappa = pin->GetReal("opacity", "gray_kappa");

    auto opacity_host = Gray(kappa);
    auto opacity_device = opacity_host.GetOnDevice();
    params.Add("h.opacity", opacity_host);
    params.Add("d.opacity", opacity_device);
  }

  return pkg;
}

} // namespace Opacity
} // namespace Microphysics
