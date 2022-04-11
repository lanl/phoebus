// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#ifndef PHOEBUS_UTILS_INTERPOLATION_HPP_
#define PHOEBUS_UTILS_INTERPOLATION_HPP_

// Spiner includes
#include <spiner/interpoaltion.hpp>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

// Phoebus includes
#include "phoebus_utils/grid_utils.hpp"
#include "phoebus_utils/robust.hpp"

namespace Interpolation {

using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using Spiner::weights_t;

namespace Nodes {
namespace Linear {

/*
 * Get interpolation weights for linear interpolation
 * PARAM[IN] - x - location to interpolate to
 * PARAM[IN] - nx - number of points along this direction. Used for sanity checks.
 * PARAM[IN] - coords - parthenon coords object
 * PARAM[OUT] - ix - index of points to interpolate
 * PARAM[OUT] - w - weights
 */
template <int DIR>
KOKKOS_INLINE_FUNCTION void GetWeights(const Real x, const int nx,
                                       const Coordinates_t &coords, int &ix,
                                       weights_t &w) {
  const Real min = Coordniates::GetXf<DIR>(0, coords);
  const Real dx = coords.Dx(DIR);
  ix = std::min(std::max(0, static_cast<int>(robust::ratio(x - min, dx))), nx - 2);
  const Real floor = min + ix * dx;
  w[1] = robust::ratio(x - floor, dx);
  w[0] = 1. - w[1];
}

/*
 * Trilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - X3, X2, X1: Coordinate locations
 * PARAM[IN] - nx3, nx2, nx1 - number of points along each direction
 * PARAM[IN] - coords - parthenon coords object
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - b, v, variable and meshblock index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(const Real X3, const Real X2, const Real X1,
                               const int nx3, const int nx2, const int nx1,
                               const Coordinates_t &coords, const Pack &p, int b, int v) {
  int ix[3];
  weights_t w[3];
  GetWeights(X1, nx1, coords, ix[0], w[0]);
  GetWeights(X2, nx2, coords, ix[1], w[1]);
  GetWeights(X3, nx3, coords, ix[2], w[2]);
  return (w[2][0] * (w[1][0] * (w[0][0] * p(b, v, ix[2], ix[1], ix[0]) +
                                w[0][1] * p(b, v, ix[2], ix[1], ix[0] + 1)) +
                     w[1][1] * (w[0][0] * p(b, v, ix[2], ix[1] + 1, ix[0]) +
                                w[0][1] * p(b, v, ix[2], ix[1] + 1, ix[0] + 1))) +
          w[2][1] * (w[1][0] * (w[0][0] * p(b, v, ix[2] + 1, ix[1], ix[0]) +
                                w[0][1] * p(b, v, ix[2] + 1, ix[1], ix[0] + 1)) +
                     w[1][1] * (w[0][0] * p(b, v, ix[2] + 1, ix[1] + 1, ix[0]) +
                                w[0][1] * p(b, v, ix[2] + 1, ix[1] + 1, ix[0] + 1))));
}

/*
 * Trilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - X3, X2, X1: Coordinate locations
 * PARAM[IN] - nx3, nx2, nx1 - number of points along each direction
 * PARAM[IN] - coords - parthenon coords object
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - b, v, variable and meshblock index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(const Real X3, const Real X2, const Real X1,
                               const int nx3, const int nx2, const int nx1,
                               const Coordinates_t &coords, const Pack &p, int v) {
  return Do(x3, x2, x1, nx3, nx2, nx1, coords, p, v);
}

} // namespace Linear
} // namespace Nodes
} // namespace Interpolation

#endif // PHOEBUS_UTILS_INTERPOLATION_HPP_
