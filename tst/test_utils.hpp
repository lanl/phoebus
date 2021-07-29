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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

KOKKOS_INLINE_FUNCTION
bool SoftEquiv(const Real &val, const Real &ref, const Real tol = 1.e-8,
               const bool ignore_small = true) {
  if (ignore_small) {
    if (fabs(val) < tol && fabs(ref) < tol) {
      return true;
    }
  }

  if (fabs(val - ref) < tol * fabs(ref) / 2) {
    return true;
  } else {
    return false;
  }
}

#endif // TEST_UTILS_HPP_
