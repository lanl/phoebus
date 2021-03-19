#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"

#include "geometry/mckinney_gammie_ryan.hpp"

namespace Geometry {

template <>
McKinneyGammieRyan GetTransformation<McKinneyGammieRyan>(StateDescriptor *pkg) {
  bool derefine_poles = pkg->Param<bool>("derefine_poles");
  Real h = pkg->Param<Real>("h");
  Real xt = pkg->Param<Real>("xt");
  Real alpha = pkg->Param<Real>("alpha");
  Real x0 = pkg->Param<Real>("x0");
  Real smooth = pkg->Param<Real>("smooth");
  return McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);
}

} // namespace Geometry
