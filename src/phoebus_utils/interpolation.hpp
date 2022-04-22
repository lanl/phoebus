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
#include <spiner/interpolation.hpp>

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

// TODO(JMM): Is this Interpolation::Do syntax reasonable? An
// alternative path would be a class called "LCInterp with all
// static functions. Then it could have an `operator()` which would
// be maybe nicer?
// TODO(JMM): Merge this w/ what Ben has done.
namespace Cent {
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
  const Real min = Coordinates::GetXv<DIR>(0, coords);
  const Real dx = coords.Dx(DIR);
  ix = std::min(std::max(0, static_cast<int>(robust::ratio(x - min, dx))), nx - 2);
  const Real floor = min + ix * dx;
  w[1] = robust::ratio(x - floor, dx);
  w[0] = 1. - w[1];
}

/*
 * Trilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2, X3 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(int b, const Real X1, const Real X2, const Real X3,
                               const Pack &p, int v) {
  const auto &coords = p.GetCoords(b);
  int ix[3];
  weights_t w[3];
  GetWeights<X1DIR>(X1, p.GetDim(1), coords, ix[0], w[0]);
  GetWeights<X2DIR>(X2, p.GetDim(2), coords, ix[1], w[1]);
  GetWeights<X3DIR>(X3, p.GetDim(3), coords, ix[2], w[2]);
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
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2, X3 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(int b, const Real X1, const Real X2, const Pack &p,
                               int v) {
  const auto &coords = p.GetCoords(b);
  int ix1, ix2;
  weights_t w1, w2;
  GetWeights<X1DIR>(X1, p.GetDim(1), coords, ix1, w1);
  GetWeights<X2DIR>(X2, p.GetDim(2), coords, ix2, w2);
  return (w2[0] * (w1[0] * p(b, v, ix2, ix1) + w1[1] * p(b, v, ix2, ix1 + 1)) +
          w2[1] * (w1[0] * p(b, v, ix2 + 1, ix1) + w1[1] * (b, v, ix2 + 1, ix1 + 1)));
}

/*
 * Trilinear or bilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - axisymmetric
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2, X3 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
// JMM: I know this won't vectorize because of the switch, but it
// probably won't anyway, since we're doing trilinear
// interpolation, which will kill memory locality.  Doing it this
// way means we can do trilinear vs bilinear which I think is a
// sufficient win at minimum code bloat.
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(bool axisymmetric, int b, const Real X1, const Real X2,
                               const Real X3, const Pack &p, int v) {
  if (axisymmetric) {
    return Do(b, X1, X2, X3, p, v);
  } else {
    return Do(b, X1, X2, p, v);
  }
}

} // namespace Linear
} // namespace Cent
} // namespace Interpolation

// Convenience Namespace Alias
namespace LCInterp = Interpolation::Cent::Linear;

#endif // PHOEBUS_UTILS_INTERPOLATION_HPP_
