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

#define SMALL (1.e-200)

namespace phoebus {

/*
 * Calculate the normal observer Lorentz factor from normal observer three-velocity
 *
 * PARAM[IN]    v[3]   Normal observer three-velocity (phoebus primitive velocity)
 * PARAM[IN]    system Coordinate system
 * PARAM[IN]    loc    Location on spatial cell where geometry is processed
 * PARAM[IN]    k      k index of meshblock cell
 * PARAM[IN]    j      j index of meshblock cell
 * PARAM[IN]    i      i index of meshblock cell
 *
 * RETURN Lorentz factor in normal observer frame
 */
KOKKOS_INLINE_FUNCTION Real
GetLorentzFactor(Real v[3], const Geometry::CoordSysMeshBlock &system,
                 CellLocation loc, const int k, const int j, const int i) {
  Real W = 1;
  Real gamma[Geometry::NDSPACE][Geometry::NDSPACE];
  system.Metric(loc, k, j, i, gamma);
  for (int l = 0; l < Geometry::NDSPACE; ++l) {
    for (int m = 0; m < Geometry::NDSPACE; ++m) {
      W -= v[l] * v[m] * gamma[l][m];
    }
  }
  W = 1. / std::sqrt(std::abs(W) + SMALL);
  return W;
}

/*
 * Calculate the coordinate frame four-velocity from the normal observer three-velocity
 *
 * PARAM[IN]    v[3]   Normal observer three-velocity (phoebus primitive velocity)
 * PARAM[IN]    system Coordinate system
 * PARAM[IN]    loc    Location on spatial cell where geometry is processed
 * PARAM[IN]    k      k index of meshblock cell
 * PARAM[IN]    j      j index of meshblock cell
 * PARAM[IN]    i      i index of meshblock cell
 * PARAM[OUT]   u      Coordinate frame four-velocity
 */
KOKKOS_INLINE_FUNCTION void
GetFourVelocity(Real v[3], const Geometry::CoordSysMeshBlock &system,
                CellLocation loc, const int k, const int j, const int i,
                Real u[Geometry::NDFULL]) {
  Real beta[Geometry::NDSPACE];
  Real W = GetLorentzFactor(v, system, loc, k, j, i);
  Real alpha = system.Lapse(loc, k, j, i);
  system.ContravariantShift(loc, k, j, i, beta);
  u[0] = W / (std::abs(alpha) + SMALL);
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    u[l] = W * v[l - 1] - u[0] * beta[l - 1];
  }
}

} // namespace phoebus

#undef SMALL

#endif // PHOEBUS_UTILS_RELATIVITY_UTILS_HPP_
