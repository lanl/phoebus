#include "analysis.hpp"
#include "phoebus_utils/variables.hpp"

namespace analysis {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto analysis_pkg = std::make_shared<StateDescriptor>("analysis");
  Params &params = analysis_pkg->AllParams();
  Real sigma = pin->GetOrAddReal("analysis", "sigma", 2e-3);
  Real outside_pns_threshold =
      pin->GetOrAddReal("analysis", "outside_pns_threshold",
                        2.42e-5); // corresponds to entropy > 3 kb/baryon
  Real inside_pns_threshold = pin->GetOrAddReal("analysis", "inside_pns_threshold",
                                                0.008); // corresponds to r < 80 km
  Real radius_cutoff_mdot =
      pin->GetOrAddReal("analysis", "radius_cutoff_mdot", 0.04); // default 400km
  if (sigma < 0) {
    PARTHENON_THROW("sigma must be greater than 0");
  }
  params.Add("sigma", sigma);
  params.Add("outside_pns_threshold", outside_pns_threshold);
  params.Add("inside_pns_threshold", inside_pns_threshold);
  params.Add("radius_cutoff_mdot", radius_cutoff_mdot);

  return analysis_pkg;
} // initialize
} // namespace analysis
