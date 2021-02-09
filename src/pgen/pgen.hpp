#ifndef _PGEN_H_
#define _PGEN_H_

// Parthenon includes
#include <utils/error_checking.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singularity includes
#include <eos/eos.hpp>

// internal includes
#include "fluid/fluid.hpp"

// add the name of a namespace that contains your new ProblemGenerator
#define FOREACH_PROBLEM                                                                  \
  PROBLEM(phoebus)                                                                       \
  PROBLEM(shock_tube)                                                                    \
  PROBLEM(linear_modes)                                                                  \
  PROBLEM(thin_cooling)

/*
// DO NOT TOUCH THE MACROS BELOW
*/

// declare all the problem generators
#define PROBLEM(name) namespace name { void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin); }
  FOREACH_PROBLEM
#undef PROBLEM

namespace phoebus {

// make a map so we get all the function pointers available for lookup by name
static std::map<std::string, std::function<void(MeshBlock *pmb, ParameterInput *pin)>> pgen_dict ({
#define PROBLEM(name) {#name, name::ProblemGenerator} ,
  FOREACH_PROBLEM
#undef PROBLEM
});

/*
// END OF UNTOUCHABLE MACRO SECTION
*/

KOKKOS_FUNCTION
Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P);

} // namespace phoebus

#endif
