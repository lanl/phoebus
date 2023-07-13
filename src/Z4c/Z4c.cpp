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

// stdlib
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Parthenon
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Phoebus
#ifndef PHOEBUS_IN_UNIT_TESTS
#include "geometry/geometry.hpp"
#endif // PHOEBUS_UNIT_TESTS

#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/robust.hpp"

#include "Z4c.hpp"

using namespace parthenon::package::prelude;
using parthenon::AllReduce;
using parthenon::MetadataFlag;

namespace Z4c {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto Z4c = std::make_shared<StateDescriptor>("Z4c");
  Params &params = Z4c->AllParams();

  bool enable_Z4c = pin->GetOrAddBoolean("Z4c", "enabled", false);
  params.Add("enable_Z4c", enable_Z4c);
  
  return Z4c
}

#endif //PHOEBUS_UNIT_TEST

}//namespace Z4c
