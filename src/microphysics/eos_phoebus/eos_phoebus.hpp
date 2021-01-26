#ifndef MICROPHYSICS_EOS_EOS_HPP_
#define MICROPHYSICS_EOS_EOS_HPP_

#include <memory>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace Microphysics {

namespace EOS {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
}

} // namespace Microphysics

#endif // MICROPHYSICS_EOS_EOS_HPP_
