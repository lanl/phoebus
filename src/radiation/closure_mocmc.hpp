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

#ifndef CLOSURE_MOCMC_HPP_
#define CLOSURE_MOCMC_HPP_

#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"
#include "radiation/closure.hpp"
#include "radiation/local_three_geometry.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace radiation {

using namespace LinearAlgebra;
using namespace robust;

/// Holds methods for closing the radiation moment equations as well as calculating
/// radiation moment source terms.
template <class Vec, class Tens2, class SET = ClosureSettings<>>
class ClosureMOCMC : public ClosureEdd<Vec, Tens2, SET> {

 public:
  using typename ClosureEdd<Vec, Tens2, SET>::LocalGeometryType;

  //-------------------------------------------------------------------------------------
  /// Constructor just calculates the inverse 3-metric, covariant three-velocity, and the
  /// Lorentz factor for the given background state.
  KOKKOS_FUNCTION
  ClosureMOCMC(const Vec con_v_in, LocalGeometryType *g)
      : ClosureEdd<Vec, Tens2, SET>(con_v_in, g) {}

  KOKKOS_FUNCTION
  ClosureStatus GetCovTilPiFromPrim(const Real J, const Vec cov_H, Tens2 *con_tilPi) {
    PARTHENON_FAIL("MOCMC has already stored TilPi elsewhere!");

    // Do nothing
    return ClosureStatus::fail;
  }

  KOKKOS_FUNCTION
  ClosureStatus GetCovTilPiFromCon(Real E, const Vec cov_F, Real &xi, Real &phi,
                                   Tens2 *con_tilPi) {
    PARTHENON_FAIL("MOCMC has already stored TilPi elsewhere!");

    // Do nothing
    return ClosureStatus::fail;
  }
};
} // namespace radiation

#endif // CLOSURE_MOCMC_HPP_
