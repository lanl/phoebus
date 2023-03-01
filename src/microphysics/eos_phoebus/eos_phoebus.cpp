// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
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
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// parthenon includes
#include <parthenon/package.hpp>

// singularity includes
#include <singularity-eos/eos/eos.hpp>
#include <singularity-eos/eos/eos_builder.hpp>

// phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/variables.hpp"

namespace Microphysics {
namespace EOS {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

using names_t = std::vector<std::string>;
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using namespace singularity;
  auto pkg = std::make_shared<StateDescriptor>("eos");
  Params &params = pkg->AllParams();

  const std::string block_name = "eos";

  phoebus::UnitConversions unit_conv(pin);
  const Real time_unit = unit_conv.GetTimeCodeToCGS();
  const Real mass_unit = unit_conv.GetMassCodeToCGS();
  const Real length_unit = unit_conv.GetLengthCodeToCGS();
  const Real temp_unit = unit_conv.GetTemperatureCodeToCGS();
  const bool use_length_time = true;

  const std::vector<std::string> valid_eos_names = {IdealGas::EosType()
#ifdef SPINER_USE_HDF
                                                        ,
                                                    StellarCollapse::EosType()
#endif
  };

  std::string eos_type = pin->GetString(block_name, std::string("type"));
  params.Add("type", eos_type);
  if (eos_type.compare(IdealGas::EosType()) == 0) {
    const Real gm1 = pin->GetReal(block_name, "Gamma") - 1.0;
    const Real Cv = pin->GetReal(block_name, "Cv");

    EOS eos_host = UnitSystem<IdealGas>(IdealGas(gm1, Cv),
                                        eos_units_init::length_time_units_init_tag,
                                        time_unit, mass_unit, length_unit, temp_unit);
    EOS eos_device = eos_host.GetOnDevice();

    params.Add("d.EOS", eos_device);
    params.Add("h.EOS", eos_host);
#ifdef SPINER_USE_HDF
  } else if (eos_type == StellarCollapse::EosType()) {
    const std::string filename = pin->GetString(block_name, "filename");
    const bool use_sp5 = pin->GetOrAddBoolean(block_name, "use_sp5", true);
    const bool filter_bmod = pin->GetOrAddBoolean(block_name, "filter_bmod", true);
    const bool use_ye = pin->GetOrAddBoolean("fluid", "Ye", false);
    PARTHENON_REQUIRE_THROWS(use_ye,
                             "\"StellarCollapse\" EOS requires that Ye be enabled!");

    EOS eos_host =
        UnitSystem<StellarCollapse>(StellarCollapse(filename, use_sp5, filter_bmod),
                                    eos_units_init::length_time_units_init_tag, time_unit,
                                    mass_unit, length_unit, temp_unit);
    EOS eos_device = eos_host.GetOnDevice();

    params.Add("d.EOS", eos_device);
    params.Add("h.EOS", eos_host);
#endif
  } else {
    std::stringstream error_mesg;
    error_mesg << __func__ << ": " << eos_type << " is an invalid EOS selection."
               << " Valid EOS names are:\n";
    for (const auto &name : valid_eos_names) {
      error_mesg << "\t" << name << "\n";
    }
    error_mesg << std::endl;
    PARTHENON_THROW(error_mesg);
  }

  return pkg;
}
} // namespace EOS
} // namespace Microphysics
