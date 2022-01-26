#ifndef PHOEBUS_UTILS_ROBUST_HPP_
#define PHOEBUS_UTILS_ROBUST_HPP_

#include <kokkos_abstraction.hpp>
#include <limits>

namespace robust {

constexpr Real SMALL = 10 * std::numeric_limits<Real>::epsilon();

KOKKOS_FORCEINLINE_FUNCTION
Real make_positive(const Real val) {
  return std::max(val,SMALL);
}

KOKKOS_FORCEINLINE_FUNCTION
Real make_bounded(const Real val, const Real vmin, const Real vmax) {
  return std::min(std::max(val,vmin+SMALL), vmax*(1.0-SMALL));
}

template <typename T> KOKKOS_INLINE_FUNCTION int sgn(const T &val) {
  return (T(0) <= val) - (val < T(0));
}
template <typename T> KOKKOS_INLINE_FUNCTION T ratio(const T &a, const T &b) {
  const T sgn = denom >= 0 ? 1 : -1;
  return a / (b + sgn * std::numeric_limits<T>::min());
}
template <> KOKKOS_INLINE_FUNCTION Real ratio(const Real &a, const Real &b) {
  return a / (b + sgn(b) * SMALL);
}

} // namespace robust

#endif // PHOEBUS_UTILS_ROBUST_HPP_
