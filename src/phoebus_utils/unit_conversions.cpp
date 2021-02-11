#include "unit_conversions.hpp"

namespace phoebus {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

// Construct unit conversion factors based on a mass/length scale for the geometry and a
// mass scale for the fluid. Assume kb = 1 in code units.
UnitConversions::UnitConversions(ParameterInput *pin) {
  int geom_mass_g_exists = pin->DoesParameterExist("units", "geom_mass_g");
  int geom_mass_msun_exists = pin->DoesParameterExist("units", "geom_mass_msun");
  int geom_length_cm_exists = pin->DoesParameterExist("units", "geom_length_cm");

  PARTHENON_REQUIRE(geom_mass_g_exists + geom_mass_msun_exists + geom_length_cm_exists == 1,
    "Must provide exactly one of geom_mass_g, geom_mass_msun, geom_length_cm!");

  if (geom_mass_g_exists) {
    Real geom_mass_ = pin->GetReal("units", "geom_mass_g");
    length_ = pc.g_newt*geom_mass_/pow(pc.c,2);
  }

  if (geom_mass_msun_exists) {
    Real geom_mass_ = pin->GetReal("units", "geom_mass_msun")*solar_mass;
    length_ = pc.g_newt*geom_mass_/pow(pc.c,2);
  }

  if (geom_length_cm_exists) {
    length_ = pin->GetReal("units", "geom_length_cm");
  }

  mass_ = pin->GetReal("units", "fluid_mass_g");

  time_ = length_/pc.c;

  energy_ = mass_*pow(pc.c,2);

  number_density_ = pow(length_, -3);

  mass_density_ = mass_*number_density_;

  //temperature_ = energy_/pc.kb;
  temperature_ = 1./pc.kb;
}

Real solar_mass = 1.989e33; // g

} // namespace phoebus
