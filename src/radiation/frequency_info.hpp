// © 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

#include <cmath>

#ifndef RADIATION_FREQUENCY_INFO_
#define RADIATION_FREQUENCY_INFO_

namespace radiation {

class FrequencyInfo {

 public:
  KOKKOS_FUNCTION
  FrequencyInfo() : lnumin_(0.), dlnu_(0.), num_bins_(0) {}

  KOKKOS_FUNCTION
  FrequencyInfo(const Real numin, const Real dlnu, const int num_bins)
      : lnumin_(std::log(numin)), dlnu_(dlnu), num_bins_(num_bins) {}

  KOKKOS_INLINE_FUNCTION Real GetLeftBinEdgeNu(const int n) const {
    PARTHENON_DEBUG_REQUIRE(n >= 0 && n < num_bins_, "n is out of bounds!");
    return std::exp(lnumin_ + n * dlnu_);
  }

  KOKKOS_INLINE_FUNCTION Real GetRightBinEdgeNu(const int n) const {
    PARTHENON_DEBUG_REQUIRE(n >= 0 && n < num_bins_, "n is out of bounds!");
    return std::exp(lnumin_ + (n + 1) * dlnu_);
  }

  KOKKOS_INLINE_FUNCTION Real GetBinCenterNu(const int n) const {
    PARTHENON_DEBUG_REQUIRE(n >= 0 && n < num_bins_, "n is out of bounds!");
    return std::exp(lnumin_ + (n + 0.5) * dlnu_);
  }

  KOKKOS_INLINE_FUNCTION Real GetNuMin() const { return std::exp(lnumin_); }

  KOKKOS_INLINE_FUNCTION Real GetNuMax() const {
    return std::exp(lnumin_ + num_bins_ * dlnu_);
  }

  KOKKOS_INLINE_FUNCTION Real GetNumBins() const { return num_bins_; }

  KOKKOS_INLINE_FUNCTION Real GetDLogNu() const { return dlnu_; }

 private:
  const Real lnumin_;
  const Real dlnu_;
  const int num_bins_;
};

} // namespace radiation

#endif // RADIATION_FREQUENCY_INFO_
