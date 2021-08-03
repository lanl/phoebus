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

#include <memory>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "gr1d.hpp"

using namespace parthenon::package::prelude;

namespace GR1D {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto gr1d = std::make_shared<StateDescriptor>("GR1D");
  bool enable_gr1d = pin->GetOrAddBoolean("GR1D", "enabled", false);
  if (!enable_gr1d) return gr1d;
}
} // namespace GR1D
