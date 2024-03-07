#ifndef ANALYSIS_HPP_INCLUDED
#define ANALYSIS_HPP_INCLUDED
#include <parthenon/package.hpp>
#include <string>
#include <utils/error_checking.hpp>
#include <vector>

using namespace parthenon::package::prelude;

namespace analysis {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace analysis
#endif
