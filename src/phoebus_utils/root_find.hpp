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

#ifndef PHOEBUS_UTILS_ROOT_FIND_HPP_
#define PHOEBUS_UTILS_ROOT_FIND_HPP_

namespace root_find {

enum class RootFindStatus { success, failure };

template <typename F>
KOKKOS_INLINE_FUNCTION void broyden2(F &func, Real v[2], const int max_iter = 50,
                                     const Real tol = 1.e-10,
                                     RootFindStatus *status = nullptr) {
  double resid[2];
  func(v[0], v[1], resid);

  if (status != nullptr) {
    *status = RootFindStatus::success;
  }
}

template <typename F>
KOKKOS_INLINE_FUNCTION Real itp(F &func, Real a, Real b, const Real tol,
                                RootFindStatus *status = nullptr) {
  constexpr Real kappa1 = 0.1;
  constexpr Real kappa2 = 2.0;
  constexpr int n0 = 0;

  const int nmax = std::ceil(std::log2(0.5 * (b - a) / tol)) + n0;
  int j = 0;

  Real ya = func(a);
  Real yb = func(b);
  if (ya * yb > 0.0) {
    if (status != nullptr) {
      *status = RootFindStatus::failure;
    }
    printf("Root not bracketed in find_root. a, b, ya, yb = %g %g %g %g\n", a, b, ya, yb);
    return 0.5 * (a + b);
  }
  Real sign = (ya < 0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;

  while (b - a > 2.0 * tol) {
    const Real xh = 0.5 * (a + b);
    const Real xf = (yb * a - ya * b) / (yb - ya);
    const Real xhxf = xh - xf;
    const Real delta = std::min(kappa1 * std::pow(b - a, kappa2), std::abs(xhxf));
    const Real sigma = (xhxf > 0 ? 1.0 : -1.0);
    const Real xt = (delta <= sigma * xhxf ? xf + sigma * delta : xh);
    const Real r = tol * std::pow(2.0, nmax - j) - 0.5 * (b - a);
    const Real xitp = (std::fabs(xt - xh) > r ? xh - sigma * r : xt);
    const Real yitp = sign * func(xitp);
    if (yitp > 0.0) {
      b = xitp;
      yb = yitp;
    } else if (yitp < 0.0) {
      a = xitp;
      ya = yitp;
    } else {
      a = xitp;
      b = xitp;
    }
    j++;
  }
  if (status != nullptr) {
    *status = RootFindStatus::success;
  }
  return 0.5 * (a + b);
}

template <typename F>
KOKKOS_INLINE_FUNCTION Real bisect(F func, Real a, Real b, const Real tol) {

  Real ya = func(a);
  Real yb = func(b);
  if (ya * yb > 0.0) {
    printf("root failure: %g %g   %g %g\n", a, b, ya, yb);
  }
  PARTHENON_REQUIRE(ya * yb <= 0.0, "Root not bracketed in find_root from\n" +
                                        std::to_string(a) + "\n" + std::to_string(b) +
                                        "\n" + std::to_string(ya) + "\n" +
                                        std::to_string(yb) + "\n");
  Real sign = (ya < 0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;

  while (b - a > tol) {
    const Real xh = 0.5 * (a + b);
    const Real yh = sign * func(xh);
    if (yh > 0.0) {
      b = xh;
      yb = yh;
    } else if (yh < 0.0) {
      a = xh;
      ya = yh;
    } else {
      a = xh;
      b = xh;
    }
  }
  return 0.5 * (a + b);
}

} // namespace root_find

#endif // PHOEBUS_UTILS_ROOT_FIND_HPP_
