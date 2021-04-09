#ifndef GEOMETRY_GEOMETRY_DEFAULTS_HPP_
#define GEOMETRY_GEOMETRY_DEFAULTS_HPP_

#include <string>
#include <vector>

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// Phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace Geometry {

template <typename System>
void Initialize(ParameterInput *pin, StateDescriptor *geometry) { }

template <typename System> System GetCoordinateSystem(MeshBlockData<Real> *rc);

template <typename System> System GetCoordinateSystem(MeshData<Real> *rc);

template <typename System> void SetGeometry(MeshBlockData<Real> *rc);

template <typename Transformation>
Transformation GetTransformation(StateDescriptor *pkg);

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_DEFAULTS_HPP_
