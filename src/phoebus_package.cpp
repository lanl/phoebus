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

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::driver::prelude;

#include "phoebus_utils/unit_conversions.hpp"

namespace phoebus {

using parthenon::Params;
using parthenon::StateDescriptor;

// Simple package for storing simulation- and problem-wide objects and parameters for I/O
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("phoebus");
  Params &params = pkg->AllParams();

  // Add some parthenon parameters for output
  params.Add("tlim", pin->GetReal("parthenon/time", "tlim"));
  params.Add("nlim", pin->GetReal("parthenon/time", "nlim"));
  params.Add("integrator", pin->GetString("parthenon/time", "integrator"));
  params.Add("do_post_init_comms",
             pin->GetOrAddBoolean("phoebus", "do_post_init_comms", false));

  // Store unit conversions
  params.Add("unit_conv", phoebus::UnitConversions(pin));
  auto &unit_conv = params.Get<phoebus::UnitConversions>("unit_conv");
  const Real MassCodeToCGS = unit_conv.GetMassCodeToCGS();
  params.Add("MassCodeToCGS", MassCodeToCGS);
  const Real LengthCodeToCGS = unit_conv.GetLengthCodeToCGS();
  params.Add("LengthCodeToCGS", LengthCodeToCGS);
  const Real TimeCodeToCGS = unit_conv.GetTimeCodeToCGS();
  params.Add("TimeCodeToCGS", TimeCodeToCGS);
  const Real TemperatureCodeToCGS = unit_conv.GetTemperatureCodeToCGS();
  params.Add("TemperatureCodeToCGS", TemperatureCodeToCGS);

  auto code_constants = CodeConstants(unit_conv);
  params.Add("code_constants", phoebus::CodeConstants(unit_conv));

  return pkg;
}

} // namespace phoebus
