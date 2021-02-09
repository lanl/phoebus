#include "unit_conversions.hpp"

namespace phoebus {

UnitConversions::UnitConversions(ParameterInput *pinput) : mass_(1.) {}

std::unique_ptr<UnitConversions> unit_conv;

} // namespace phoebus
