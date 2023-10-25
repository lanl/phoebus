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

/*
Breif : Main function for Z4c formulation for the 
        Einstein's equations
Date : Jul.19.2023
Author : Hyun Lim

*/

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

// Z4c header to contain some utils including FD computations
#include "z4c.hpp"

using namespace parthenon::package::prelude;
using parthenon::AllReduce;
using parthenon::MetadataFlag;

namespace z4c {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto z4c = std::make_shared<StateDescriptor>("z4c");
  Params &params = z4c->AllParams();

  bool enable_z4c = pin->GetOrAddBoolean("z4c", "enabled", false);
  params.Add("enable_z4c", enable_Z4c);
  if (!enable_z4c) return z4c;
  
  // Add vars
  Metadata m_constraint_scalar({Metadata::Cell, Metadata::Metadata::Derived, Metadata::OneCopy});
  Metadata m_constraint_vector({Metadata::Cell, Metadata::Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3});
  z4c->AddField<constraint::H>(m_constraint_scalar);
  z4c->AddField<constraint::M>(m_constraint_vector);

  Metadata m_evolution_rank2({Metadata::Cell, Metadata::Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3,3});
  z4c->AddField<evolution::At>(m_evolution_rank2);
 
  

  return z4c
}

#endif //PHOEBUS_UNIT_TEST

}//namespace z4c
