#ifndef PHOEBUS_UTILS_ROBUST_HPP_
#define PHOEBUS_UTILS_ROBUST_HPP_

#include <kokkos_abstraction.hpp>

namespace robust {

#define TINY (1.e-14)

KOKKOS_FORCEINLINE_FUNCTION
Real make_positive(const Real val) {
  return std::max(val,TINY);
}

KOKKOS_FORCEINLINE_FUNCTION
Real make_bounded(const Real val, const Real vmin, const Real vmax) {
  return std::min(std::max(val,vmin+TINY), vmax*(1.0-TINY));
}

#undef TINY

} // namespace robust

#endif // PHOEBUS_UTILS_ROBUST_HPP_
