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

#ifndef PHOEBUS_UTILS_RELATIVITY_UTILS_HPP_
#define PHOEBUS_UTILS_RELATIVITY_UTILS_HPP_

// System includes
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>

// Phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/robust.hpp"

namespace phoebus {

/*
 * Calculate the normal observer Lorentz factor from primitive three-velocity
 *
 * PARAM[IN] - vcon[3] - Gamma*normal observer three-velocity (phoebus primitive)
 * PARAM[IN] - gammacov[3][3] - Covariant three-metric
 *
 * RETURN - Lorentz factor in normal observer frame
 */
KOKKOS_INLINE_FUNCTION Real
GetLorentzFactor(const Real vcon[Geometry::NDSPACE], const Real gammacov[Geometry::NDSPACE][Geometry::NDSPACE]) {
  Real vsq = 0.;
  SPACELOOP2(ii, jj) {
    vsq += gammacov[ii][jj]*vcon[ii]*vcon[jj];
  }
  return std::sqrt(1. + vsq);
}

/*
 * Calculate the normal observer Lorentz factor from primitive three-velocity
 *
 * PARAM[IN] - vcon[3] - Gamma*normal pbserver three velocity (phoebus primitive)
 * PARAM[IN] - gcov[4][4] - Covariant four-metric
 *
 * RETURN - Lorentz factor in normal observer frame
 */
KOKKOS_INLINE_FUNCTION Real
GetLorentzFactor(const Real vcon[Geometry::NDSPACE], const Real gcov[Geometry::NDFULL][Geometry::NDFULL]) {
  Real vsq = 0.;
  SPACELOOP2(ii, jj) {
    vsq += gcov[ii+1][jj+1]*vcon[ii]*vcon[jj];
  }
  return std::sqrt(1. + vsq);
}

/*
 * Calculate the normal observer Lorentz factor from normal observer three-velocity
 *
 * PARAM[IN] - v[3] - Gamma*Normal observer three-velocity (phoebus primitive velocity)
 * PARAM[IN] - system - Coordinate system
 * PARAM[IN] - loc - Location on spatial cell where geometry is processed
 * PARAM[IN] - k - k index of meshblock cell
 * PARAM[IN] - j - j index of meshblock cell
 * PARAM[IN] - i - i index of meshblock cell
 *
 * RETURN - Lorentz factor in normal observer frame
 */
KOKKOS_INLINE_FUNCTION Real
GetLorentzFactor(const Real vcon[Geometry::NDSPACE], const Geometry::CoordSysMeshBlock &system,
                 CellLocation loc, const int k, const int j, const int i) {
  Real gamma[Geometry::NDSPACE][Geometry::NDSPACE];
  system.Metric(loc, k, j, i, gamma);
  return GetLorentzFactor(vcon, gamma);
}

/*
 * Calculate the coordinate frame four-velocity from the normal observer three-velocity
 *
 * PARAM[IN] - v[3] - Gamma*Normal observer three-velocity (phoebus primitive velocity)
 * PARAM[IN] - system - Coordinate system
 * PARAM[IN] - loc - Location on spatial cell where geometry is processed
 * PARAM[IN] - k - X3 index of meshblock cell
 * PARAM[IN] - j - X2 index of meshblock cell
 * PARAM[IN] - i - X1 index of meshblock cell
 * PARAM[OUT] - u - Coordinate frame contravariant four-velocity
 */
KOKKOS_INLINE_FUNCTION void
GetFourVelocity(const Real v[3], const Geometry::CoordSysMeshBlock &system,
                CellLocation loc, const int k, const int j, const int i,
                Real u[Geometry::NDFULL]) {
  Real beta[Geometry::NDSPACE];
  Real W = GetLorentzFactor(v, system, loc, k, j, i);
  Real alpha = system.Lapse(loc, k, j, i);
  system.ContravariantShift(loc, k, j, i, beta);
  u[0] = robust::ratio(W, std::abs(alpha));
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    u[l] = v[l - 1] - u[0] * beta[l - 1];
  }
}

} // namespace phoebus

#endif // PHOEBUS_UTILS_RELATIVITY_UTILS_HPP_
