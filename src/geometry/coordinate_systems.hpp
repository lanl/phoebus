#ifndef GEOMETRY_COORDINATE_SYSTEMS_HPP_
#define GEOMETRY_COORDINATE_SYSTEMS_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <utility>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "compile_constants.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/linear_algebra.hpp"

// coordinate system includes
#include "geometry/analytic_system.hpp"
#include "geometry/boyer_lindquist.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/fmks.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/minkowski.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "geometry/spherical_minkowski.hpp"

namespace Geometry {

// Coordinate system choices
using CoordSysMeshBlock = GEOMETRY_MESH_BLOCK;
using CoordSysMesh = GEOMETRY_MESH;

} // namespace Geometry

#endif // GEOMETRY_COORDINATE_SYSTEMS_HPP_
