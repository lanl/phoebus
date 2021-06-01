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
#include "geometry/inchworm.hpp"
#include "geometry/lumpy.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/minkowski.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/snake.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "geometry/spherical_minkowski.hpp"

namespace Geometry {

// Coordinate system choices
using CoordSysMeshBlock = GEOMETRY_MESH_BLOCK;
using CoordSysMesh = GEOMETRY_MESH;

} // namespace Geometry

#endif // GEOMETRY_COORDINATE_SYSTEMS_HPP_
