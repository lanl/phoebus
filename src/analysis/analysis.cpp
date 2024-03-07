#include "analysis.hpp"
#include "phoebus_utils/variables.hpp"

namespace analysis {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto analysis_pkg = std::make_shared<StateDescriptor>("analysis");
  Params &params = analysis_pkg->AllParams();
  Real sigma = pin->GetOrAddReal("analysis", "sigma", 2e-3);
  if (sigma < 0){
    PARTHENON_THROW("sigma must be greater than 0");
  }
  params.Add("sigma", sigma);

  return analysis_pkg;
} // initialize
} // namespace analysis
