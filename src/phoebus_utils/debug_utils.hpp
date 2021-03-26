//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef PHOEBUS_UTILS_DEBUG_UTILS_HPP_
#define PHOEBUS_UTILS_DEBUG_UTILS_HPP_

#include <memory>
#include <string>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "phoebus_utils/variables.hpp"

namespace Debug {

template <typename T> TaskStatus PrintRHS(T *rc) {
  auto pm = rc->GetParentPointer();
  const std::vector<std::string> vars = {
      fluid_cons::density, fluid_cons::momentum, fluid_cons::energy};
  auto pack = rc->PackVariables(vars);
  printf("Right hand side:\n");
  printf("Dimensions = %d %d %d %d %d\n", pack.GetDim(5), pack.GetDim(4),
         pack.GetDim(3), pack.GetDim(2), pack.GetDim(1));
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "print RHS", DevExecSpace(), 0, pack.GetDim(5) - 1,
      0, pack.GetDim(4) - 1, 0, pack.GetDim(3) - 1, 0, pack.GetDim(2) - 1, 0,
      pack.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j,
                    const int i) { printf("%g, ", pack(b, v, k, j, i)); });
  printf("\n");
  return TaskStatus::complete;
}

} // namespace Debug

#endif // PHOEBUS_UTILS_DEBUG_UTILS_HPP_
