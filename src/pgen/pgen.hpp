#ifndef _PGEN_H_
#define _PGEN_H_

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace phoebus {

// The problem generator for Riot.
extern void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);

void YuckyOldWornOutProblemGenerator(MeshBlock *pmb, ParameterInput *pin);

std::function<void(MeshBlock*, ParameterInput*)> GetProblemGenerator();

} // namespace Riot

#endif
