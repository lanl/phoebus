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
#include "phoebus_utils/variables.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

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

  return lapse * gdet * pack(b, prho, k, j, i) * ucon;
}

} // namespace History

#endif // ANALYSIS_HISTORY_UTILS_HPP_
