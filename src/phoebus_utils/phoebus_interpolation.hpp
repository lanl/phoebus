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

#ifndef PHOEBUS_UTILS_INTERPOLATION_
#define PHOEBUS_UTILS_INTERPOLATION_

namespace interpolation {

/// Base class for providing interpolation methods on uniformly spaced data.
/// Constructor is provided with spacing, number of support points, and desired
/// shift. GetIndicesAndWeights then updates arrays of indices and weights for
/// calculating the interpolated data. These arrays are of size StencilSize().
/// Data is forced to zero outside the boundaries.
class Interpolation {
 public:
  KOKKOS_FUNCTION
  Interpolation(const int nSupport, const Real dx, const Real shift)
      : nSupport_(nSupport), dx_(dx), shift_(shift), ishift_(std::round(shift)) {}

  virtual void GetIndicesAndWeights(const int i, int *idx, Real *wgt) = 0;
  virtual int constexpr StencilSize() = 0;

 protected:
  const int nSupport_;
  const Real dx_;
  Real shift_;
  int ishift_;
};

class PiecewiseConstant : public Interpolation {
 public:
  KOKKOS_FUNCTION
  PiecewiseConstant(const int nSupport, const Real dx, const Real shift)
      : Interpolation(nSupport, dx, shift) {}

  KOKKOS_INLINE_FUNCTION
  void GetIndicesAndWeights(const int i, int *idx, Real *wgt) override {
    idx[0] = i + ishift_;
    wgt[0] = 1.;
    if (idx[0] < 0 || idx[0] >= nSupport_) {
      idx[0] = 0;
      wgt[0] = 0.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  int constexpr StencilSize() override { return 1; }
};

class Linear : public Interpolation {
 public:
  KOKKOS_FUNCTION
  Linear(const int nSupport, const Real dx, const Real shift)
      : Interpolation(nSupport, dx, shift) {
    PARTHENON_FAIL("Not written yet!");
  }

  KOKKOS_INLINE_FUNCTION
  void GetIndicesAndWeights(const int i, int *idx, Real *wgt) override {
    idx[0] = std::floor(i + shift_);
    idx[1] = idx[0] + 1;

    wgt[0] = wgt[1] = 1. - wgt[0];

    for (int nsup = 0; nsup < 2; nsup++) {
      if (idx[nsup] < 0 || idx[nsup] >= nSupport_) {
        idx[nsup] = 0;
        wgt[nsup] = 0.;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  int constexpr StencilSize() override { return 2; }
};

} // namespace interpolation

#endif // PHOEBUS_UTILS_INTERPOLATION_
