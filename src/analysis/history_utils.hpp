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

#ifndef ANALYSIS_HISTORY_UTILS_HPP_
#define ANALYSIS_HISTORY_UTILS_HPP_

#include <string>
#include <vector>

#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;
using namespace Geometry;

namespace History {

template <typename Pack, typename Geometry>
KOKKOS_INLINE_FUNCTION Real CalcMassFlux(Pack &pack, Geometry &geom, const int prho,
                                         const int pvel_lo, const int pvel_hi,
                                         const int b, const int k, const int j,
                                         const int i) {

  Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
  Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
  Real shift[3];
  geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

  const Real vel[] = {pack(b, pvel_lo, k, j, i), pack(b, pvel_lo + 1, k, j, i),
                      pack(b, pvel_hi, k, j, i)};

  Real gcov4[4][4];
  geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
  const Real W = phoebus::GetLorentzFactor(vel, gcov4);
  const Real ucon = vel[0] - shift[0] * W / lapse;

  return -lapse * gdet * pack(b, prho, k, j, i) * ucon;
}

template <typename Pack, typename Geometry>
KOKKOS_INLINE_FUNCTION Real CalcMagnetization(Pack &pack, Geometry &geom,
                                              const int pvel_lo, const int pvel_hi,
                                              const int pb_lo, const int pb_hi,
                                              const int prho, const int b, const int k,
                                              const int j, const int i) {
  Real gam[3][3];
  geom.Metric(CellLocation::Cent, k, j, i, gam);
  const Real Bp[] = {pack(b, pb_lo, k, j, i), pack(b, pb_lo + 1, k, j, i),
                     pack(b, pb_hi, k, j, i)};
  const Real vcon[] = {pack(b, pvel_lo, k, j, i), pack(b, pvel_lo + 1, k, j, i),
                       pack(b, pvel_hi, k, j, i)};
  const Real rho = pack(b, prho, k, j, i);
  const Real W = phoebus::GetLorentzFactor(vcon, gam);

  const Real bsq = phoebus::GetMagneticFieldSquared(gam, vcon, Bp, W);

  return bsq / rho;
}

template <typename Pack, typename Geometry>
KOKKOS_INLINE_FUNCTION Real CalcEMEnergyFlux(Pack &pack, Geometry &geom,
                                             const int pvel_lo, const int pvel_hi,
                                             const int pb_lo, const int pb_hi,
                                             const int b, const int k, const int j,
                                             const int i) {
  Real gam[3][3];
  geom.Metric(CellLocation::Cent, k, j, i, gam);
  Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
  Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
  Real shift[3];
  geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

  const Real vcon[] = {pack(b, pvel_lo, k, j, i), pack(b, pvel_lo + 1, k, j, i),
                       pack(b, pvel_hi, k, j, i)};
  const Real Bp[] = {pack(b, pb_lo, k, j, i), pack(b, pb_lo + 1, k, j, i),
                     pack(b, pb_hi, k, j, i)};
  const Real W = phoebus::GetLorentzFactor(vcon, gam);

  Real Bsq = 0.0;
  Real Bdotv = 0.0;
  Real vcov[3] = {0};
  Real Bcov[3] = {0};
  for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++) {
      Bsq += gam[m][n] * Bp[m] * Bp[n];
      vcov[m] += gam[m][n] * vcon[n];
      Bcov[m] += gam[m][n] * Bp[n];
      Bdotv += gam[m][n] * Bp[m] * robust::ratio(vcon[n], W);
    }

  const Real b0 = Bdotv * W / lapse;
  const Real ucov0 = -W * lapse;
  const Real ucon1 = vcon[0] - shift[0] * W / lapse;
  const Real bcon1 = robust::ratio((Bp[0] + lapse * b0 * ucon1), W);

  const Real bsq = phoebus::GetMagneticFieldSquared(gam, vcon, Bp, W);

  // energy flux of EM stress energy tensor T_{EM}^1_0
  const Real T_EM_10 = bsq * ucon1 * ucov0 - bcon1 * b0;

  return +T_EM_10 * lapse * gdet;
} // energy flux

template <typename Pack, typename Geometry>
KOKKOS_INLINE_FUNCTION Real CalcEMMomentumFlux(Pack &pack, Geometry &geom,
                                               const int pvel_lo, const int pvel_hi,
                                               const int pb_lo, const int pb_hi,
                                               const int b, const int k, const int j,
                                               const int i) {
  Real gam[3][3];
  geom.Metric(CellLocation::Cent, k, j, i, gam);
  Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
  Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
  Real shift[3];
  geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

  const Real vcon[] = {pack(b, pvel_lo, k, j, i), pack(b, pvel_lo + 1, k, j, i),
                       pack(b, pvel_hi, k, j, i)};
  const Real Bp[] = {pack(b, pb_lo, k, j, i), pack(b, pb_lo + 1, k, j, i),
                     pack(b, pb_hi, k, j, i)};
  const Real W = phoebus::GetLorentzFactor(vcon, gam);

  Real Bsq = 0.0;
  Real Bdotv = 0.0;
  Real vcov[3] = {0};
  Real Bcov[3] = {0};
  for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++) {
      Bsq += gam[m][n] * Bp[m] * Bp[n];
      vcov[m] += gam[m][n] * vcon[n];
      Bcov[m] += gam[m][n] * Bp[n];
      Bdotv += gam[m][n] * Bp[m] * robust::ratio(vcon[n], W);
    }

  const Real b0 = Bdotv * W / lapse;
  const Real ucon1 = vcon[0] - shift[0] * W / lapse;
  const Real ucov1 = vcov[0];
  const Real bcon1 = robust::ratio((Bp[0] + lapse * b0 * ucon1), W);
  const Real bcov1 = robust::ratio((Bcov[0] + lapse * b0 * ucov1), W);

  const Real bsq = phoebus::GetMagneticFieldSquared(gam, vcon, Bp, W);

  // momentum flux part of EM stress energy tensor
  const Real T_EM_11 = bsq * ucon1 * ucov1 - bcon1 * bcov1 + 0.5 * bsq;

  return -T_EM_11 * lapse * gdet;
} // momentum flux

} // namespace History

#endif // ANALYSIS_HISTORY_UTILS_HPP_
