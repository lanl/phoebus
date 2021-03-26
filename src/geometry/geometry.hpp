#ifndef GEOMETRY_GEOMETRY_HPP_
#define GEOMETRY_GEOMETRY_HPP_

#include <memory>

#include <parthenon/package.hpp>

#include "geometry/coordinate_systems.hpp"

#include "geometry/tetrads.hpp"

using namespace parthenon::package::prelude;

namespace Geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// Set geometry data on grid, if needed.
// Potentially a very expensive operation. You only want to do this
// once, but it is deally done per meshblock, right at the beginning of
// a problem generator.
void SetGeometry(MeshBlockData<Real> *rc);

CoordSysMeshBlock GetCoordinateSystem(MeshBlockData<Real> *rc);
CoordSysMesh GetCoordinateSystem(MeshData<Real> *rc);

} // namespace Geometry

#endif // GEOMETRY_GEOMETRY_HPP_
