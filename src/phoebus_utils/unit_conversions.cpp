#include "unit_conversions.hpp"

namespace phoebus {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

UnitConversions::UnitConversions(ParameterInput *pin) {
  printf("A\n");
  int mass_g_exists = pin->DoesParameterExist("units", "mass_g");
  int mass_msun_exists = pin->DoesParameterExist("units", "mass_msun");

  PARTHENON_REQUIRE(mass_g_exists + mass_msun_exists == 1,
    "Must provide exactly one of mass_g, mass_msun!");

  if (mass_g_exists) {
    mass_ = pin->GetReal("units", "mass_g");
  }

  if (mass_msun_exists) {
    mass_ = pin->GetReal("units", "mass_msun")*solar_mass;
  }
}

Real solar_mass = 1.989e33; // g

std::unique_ptr<UnitConversions> unit_conv;

} // namespace phoebus
