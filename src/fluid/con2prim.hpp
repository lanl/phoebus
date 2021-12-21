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

#ifndef CON2PRIM_HPP_
#define CON2PRIM_HPP_

namespace con2prim {

enum class ConToPrimStatus {success, failure};
struct FailFlags {
  static constexpr Real success = 1.0;
  static constexpr Real fail = 0.0;
};

template <typename T>
class VarAccessor {
 public:
  KOKKOS_FUNCTION
  VarAccessor(const T &var, const int k, const int j, const int i)
              : var_(var), b_(0), k_(k), j_(j), i_(i) {}
  VarAccessor(const T &var, const int b, const int k, const int j, const int i)
              : var_(var), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int n) const {
    return var_(b_, n, k_, j_, i_);
  }
  const int b_, i_, j_, k_;
 private:
  const T &var_;
};

} // namespace con2prim

#endif // CON2PRIM_HPP_
