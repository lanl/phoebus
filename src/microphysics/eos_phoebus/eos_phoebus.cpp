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
#include <singularity-eos/eos/eos.hpp>
#include <singularity-eos/eos/eos_builder.hpp>

// phoebus includes
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/unit_conversions.hpp"

namespace Microphysics {
namespace EOS {
using names_t = std::vector<std::string>;
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using namespace singularity;
  auto pkg = std::make_shared<StateDescriptor>("eos");
  Params &params = pkg->AllParams();

  const std::string block_name = "eos";
  auto FillRealParams = [&](ParameterInput *pin, EOSBuilder::params_t &params,
                            const names_t &param_names) {
    for (auto &name : param_names) {
      params[name].emplace<Real>(pin->GetReal(block_name, name));
    }
  };

  bool needs_ye = false; // TODO(JMM) descriptive enough?
  names_t names;
  EOSBuilder::EOSType type;
  EOSBuilder::modifiers_t modifiers;
  EOSBuilder::params_t base_params;
  EOSBuilder::params_t relativity_params;
  // EOSBuilder::params_t shifted_params, scaled_params;

  const std::vector<std::string> valid_eos_names = {
      IdealGas::EosType()
#ifdef SPINER_USE_HDF
      , SpinerEOSDependsRhoT::EosType(),
      SpinerEOSDependsRhoSie::EosType()
#endif
  };
  std::string eos_type = pin->GetString(block_name, std::string("type"));
  if (eos_type.compare(IdealGas::EosType()) == 0) {
    type = EOSBuilder::EOSType::IdealGas;
    names = {"Cv"};
    Real gm1 = pin->GetReal(block_name, std::string("Gamma")) - 1.0;
    params.Add("gm1", gm1);
    base_params["gm1"].emplace<Real>(gm1);
    /*
      // TODO(JMM): Disabling these for now
  } else if (eos_type.compare(Gruneisen::EosType()) == 0) {
    type = EOSBuilder::EOSType::Gruneisen;
    names = {"C0", "s1", "s2", "s3", "G0", "b", "rho0", "T0", "P0", "Cv"};
  } else if (eos_type.compare(JWL::EosType()) == 0) {
    type = EOSBuilder::EOSType::JWL;
    names = {"A", "B", "R1", "R2", "w", "rho0", "Cv"};
  } else if (eos_type.compare(DavisProducts::EosType()) == 0) {
    type = EOSBuilder::EOSType::DavisProducts;
    names = {"a", "b", "k", "n", "vc", "pc", "E0", "Cv"};
  } else if (eos_type.compare(DavisReactants::EosType()) == 0) {
    type = EOSBuilder::EOSType::DavisReactants;
    names = {"rho0", "e0", "P0", "T0",    "A",  "B",
             "C",    "G0", "Z",  "alpha", "Cv0"};
    */
#ifdef SPINER_USE_HDF
  } else if (eos_type.compare(SpinerEOSDependsRhoT::EosType()) == 0) {
    type = EOSBuilder::EOSType::SpinerEOSDependsRhoT;
    base_params["filename"].emplace<std::string>(
        pin->GetString(block_name, "filename"));
    base_params["reproducibility_mode"].emplace<bool>(
        pin->GetOrAddBoolean(block_name, "reproducibility_mode", false));
    if (pin->DoesParameterExist("block_name", "sesame_id")) {
      base_params["matid"].emplace<int>(
          pin->GetInteger(block_name, "sesame_id"));
    } else if (pin->DoesParameterExist("block_name", "material_name")) {
      base_params["materialName"].emplace<std::string>(
          pin->GetString(block_name, "sesame_name"));
    } else {
      std::stringstream msg;
      msg << "Neither sesame_id nor sesame_name exists for material "
          << block_name << std::endl;
      PARTHENON_THROW(msg);
    }
  } else if (eos_type == StellarCollapse::EosType()) {
    type = EOSBuilder::EOSType::StellarCollapse;
    base_params["filename"].emplace<std::string>(
        pin->GetString(block_name, "filename"));
    base_params["use_sp5"].emplace<bool>(
        pin->GetOrAddBoolean(block_name, "use_sp5", true));
    auto use_ye = pin->GetOrAddBoolean("fluid", "Ye", false);
    PARTHENON_REQUIRE_THROWS(use_ye, "\"StellarCollapse\" EOS requires that Ye be enabled!");
#endif
  } else {
    std::stringstream error_mesg;
    error_mesg << __func__ << ": " << eos_type
               << " is an invalid EOS selection."
               << " Valid EOS names are:\n";
    for (const auto &name : valid_eos_names) {
      error_mesg << "\t" << name << "\n";
    }
    error_mesg << std::endl;
    PARTHENON_THROW(error_mesg);
  }
  FillRealParams(pin, base_params, names);

  // modifiers
  /*
  // TODO(JMM): Disabling this for now
  Real shift = pin->GetOrAddReal(block_name, "shift", 0.0);
  Real scale = pin->GetOrAddReal(block_name, "scale", 1.0);
  if (shift != 0.0) {
    shifted_params["shift"].emplace<Real>(shift);
    modifiers[EOSBuilder::EOSModifier::Shifted] = shifted_params;
  }
  if (scale != 1.0) {
    scaled_params["scale"].emplace<Real>(scale);
    modifiers[EOSBuilder::EOSModifier::Scaled] = scaled_params;
  }
  */

  // Build the EOS
  singularity::EOS eos_host =
      EOSBuilder::buildEOS(type, base_params, modifiers);
  singularity::EOS eos_device = eos_host.GetOnDevice();
  params.Add("d.EOS", eos_device);
  params.Add("h.EOS", eos_host);
  params.Add("needs_ye", needs_ye);

  // Store eos params in case they're needed
  for (auto &name : names) {
    params.Add(name, (pin->GetReal(block_name, name)));
  }

  params.Add("unit_conv", phoebus::UnitConversions(pin));

  return pkg;
}
} // namespace EOS
} // namespace Microphysics
