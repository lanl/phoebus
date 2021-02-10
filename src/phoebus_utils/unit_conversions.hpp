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
    UnitConversions(ParameterInput *pin);

    Real GetMassCodeToCGS() const { return mass_; }
    Real GetMassCGSToCode() const { return 1./mass_; }
    Real GetLengthCodeToCGS() const { return length_; }
    Real GetLengthCGSToCode() const { return 1./length_; }
    Real GetTimeCodeToCGS() const { return time_; }
    Real GetTimeCGSToCode() const { return 1./time_; }
    Real GetEnergyCodeToCGS() const { return energy_; }
    Real GetEnergyCGSToCode() const { return 1./energy_; }
    Real GetNumberDensityCodeToCGS() const { return number_density_; }
    Real GetNumberDensityCGSToCode() const { return 1./number_density_; }
    Real GetMassDensityCodeToCGS() const { return mass_density_; }
    Real GetMassDensityCGSToCode() const { return 1./mass_density_; }
    Real GetTemperatureCodeToCGS() const { return temperature_; }
    Real GetTemperatureCGSToCode() const { return 1./temperature_; }

  private:
    Real mass_;
    Real length_;
    Real time_;
    Real energy_;
    Real number_density_;
    Real mass_density_;
    Real temperature_;
};

extern Real solar_mass; // g

} // namespace phoebus

#endif // PHOEBUS_UTILS_UNIT_CONVERSIONS_HPP_
