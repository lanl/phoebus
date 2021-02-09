#ifndef PHOEBUS_UTILS_UNIT_CONVERSIONS_HPP_
#define PHOEBUS_UTILS_UNIT_CONVERSIONS_HPP_

#include <basic_types.hpp>
#include <memory>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace phoebus {

// Object for converting between cgs and code units, based on relativistic mass
class UnitConversions {
  public:
    UnitConversions(ParameterInput *pin);// {}// : mass_(mass) {}

  public:
    Real GetMass() const { return mass_; }

  private:
    Real mass_;
};

extern Real solar_mass; // g

extern std::unique_ptr<UnitConversions> unit_conv;

} // namespace phoebus

#endif // PHOEBUS_UTILS_UNIT_CONVERSIONS_HPP_
