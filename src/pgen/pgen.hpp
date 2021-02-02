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

#define FOREACH_PROBLEM                                                                  \
  PROBLEM(phoebus)                                                                       \
  PROBLEM(ShockTube)                                                                     \
  PROBLEM(LinearModes)

// declare all the problem generators
#define PROBLEM(name) namespace name { void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin); }
  FOREACH_PROBLEM
#undef PROBLEM

namespace phoebus {

// make a map so we get all the function pointers available for lookup by name
static std::map<std::string, std::function<void(MeshBlock *pmb, ParameterInput *pin)>> prob ({
#define PROBLEM(name) {#name, name::ProblemGenerator} ,
  FOREACH_PROBLEM
#undef PROBLEM
});

KOKKOS_FUNCTION
Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P);

} // namespace phoebus

#endif
