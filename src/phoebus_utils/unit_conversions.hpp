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
  Real GetMassCGSToCode() const { return 1. / mass_; }

  Real GetLengthCodeToCGS() const { return length_; }
  Real GetLengthCGSToCode() const { return 1. / length_; }

  Real GetTimeCodeToCGS() const { return time_; }
  Real GetTimeCGSToCode() const { return 1. / time_; }

  Real GetEnergyCodeToCGS() const { return energy_; }
  Real GetEnergyCGSToCode() const { return 1. / energy_; }

  Real GetNumberDensityCodeToCGS() const { return number_density_; }
  Real GetNumberDensityCGSToCode() const { return 1. / number_density_; }

  Real GetMassDensityCodeToCGS() const { return mass_density_; }
  Real GetMassDensityCGSToCode() const { return 1. / mass_density_; }

  Real GetTemperatureCodeToCGS() const { return temperature_; }
  Real GetTemperatureCGSToCode() const { return 1. / temperature_; }

 private:
  Real mass_;
  Real length_;
  Real time_;
  Real energy_;
  Real number_density_;
  Real mass_density_;
  Real temperature_;
};

// Object for holding physical constants in code units for convenience
class CodeConstants {
  using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

 public:
  CodeConstants(UnitConversions unit_conv) {
    const Real TIME = unit_conv.GetTimeCGSToCode();
    const Real MASS = unit_conv.GetMassCGSToCode();
    const Real LENGTH = unit_conv.GetLengthCGSToCode();
    const Real ENERGY = MASS * LENGTH * LENGTH / (TIME * TIME);
    const Real TEMPERATURE = unit_conv.GetTemperatureCGSToCode();
    h_code_ = pc::h * MASS * LENGTH * LENGTH / TIME;
    c_code_ = pc::c * LENGTH / TIME;
    kb_code_ = pc::kb * ENERGY / TEMPERATURE;
    mp_code_ = pc::mp * MASS;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real h() const { return h_code_; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real c() const { return c_code_; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real kb() const { return kb_code_; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real mp() const { return mp_code_; }

 private:
  Real h_code_;
  Real c_code_;
  Real kb_code_;
  Real mp_code_;
};

constexpr Real solar_mass = 1.989e33; // g

} // namespace phoebus

#endif // PHOEBUS_UTILS_UNIT_CONVERSIONS_HPP_
