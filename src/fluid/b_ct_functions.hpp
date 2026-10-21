// © 2026. Triad National Security, LLC. All rights reserved.
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

// Much of this is taken near-directly from KHARMA
// https://github.com/parthenon-hpc-lab/kharma/

// Device-callable CT helpers.

#ifndef FLUID_B_CT_FUNCTIONS_HPP_
#define FLUID_B_CT_FUNCTIONS_HPP_

#include <parthenon/parthenon.hpp>
using namespace parthenon::package::prelude;

namespace b_ct {

using parthenon::TopologicalElement;
using TE = TopologicalElement;

// Face-field divergence at cell (k, j, i).
template <typename Pack, typename Coords>
KOKKOS_INLINE_FUNCTION Real face_div(const Coords &coords, const Pack &v, const int ndim,
                                     const int k, const int j, const int i) {
  Real du =
      (v(TE::F1, 0, k, j, i + 1) - v(TE::F1, 0, k, j, i)) / coords.template Dxc<1>();
  if (ndim > 1) {
    du += (v(TE::F2, 0, k, j + 1, i) - v(TE::F2, 0, k, j, i)) / coords.template Dxc<2>();
  }
  if (ndim > 2) {
    du += (v(TE::F3, 0, k + 1, j, i) - v(TE::F3, 0, k, j, i)) / coords.template Dxc<3>();
  }
  return du;
}

template <int diff_face, int diff_side, int offset, int DIM>
KOKKOS_FORCEINLINE_FUNCTION Real FaceDiff(
    const ParArrayND<Real, parthenon::VariableState> &fine,
    const parthenon::Coordinates_t &coords, int l, int m, int n, int fk, int fj, int fi) {
  if constexpr (diff_face + 1 > DIM) {
    return 0.;
  } else {
    constexpr int df_is_k = 2 * (diff_face == 2 && DIM > 2);
    constexpr int df_is_j = 2 * (diff_face == 1 && DIM > 1);
    constexpr int df_is_i = 2 * (diff_face == 0 && DIM > 0);
    constexpr int ds_is_k = (diff_side == 2 && DIM > 2);
    constexpr int ds_is_j = (diff_side == 1 && DIM > 1);
    constexpr int ds_is_i = (diff_side == 0 && DIM > 0);
    constexpr int of_is_k = (offset == 2 && DIM > 2);
    constexpr int of_is_j = (offset == 1 && DIM > 1);
    constexpr int of_is_i = (offset == 0 && DIM > 0);
    return fine(diff_face, l, m, n, fk + df_is_k + ds_is_k + of_is_k,
                fj + df_is_j + ds_is_j + of_is_j, fi + df_is_i + ds_is_i + of_is_i) *
               coords.template FaceArea<diff_face + 1>(fk + df_is_k + ds_is_k + of_is_k,
                                                       fj + df_is_j + ds_is_j + of_is_j,
                                                       fi + df_is_i + ds_is_i + of_is_i) -
           fine(diff_face, l, m, n, fk + ds_is_k + of_is_k, fj + ds_is_j + of_is_j,
                fi + ds_is_i + of_is_i) *
               coords.template FaceArea<diff_face + 1>(fk + ds_is_k + of_is_k,
                                                       fj + ds_is_j + of_is_j,
                                                       fi + ds_is_i + of_is_i) -
           fine(diff_face, l, m, n, fk + df_is_k + of_is_k, fj + df_is_j + of_is_j,
                fi + df_is_i + of_is_i) *
               coords.template FaceArea<diff_face + 1>(fk + df_is_k + of_is_k,
                                                       fj + df_is_j + of_is_j,
                                                       fi + df_is_i + of_is_i) +
           fine(diff_face, l, m, n, fk + of_is_k, fj + of_is_j, fi + of_is_i) *
               coords.template FaceArea<diff_face + 1>(fk + of_is_k, fj + of_is_j,
                                                       fi + of_is_i);
  }
}

// Divergence-preserving prolongation from Olivares et al. (2019).
struct ProlongateInternalOlivares {
  static constexpr bool OperationRequired(TE fel, TE cel) {
    return IsSubmanifold(fel, cel);
  }

  template <int DIM, TE fel = TE::CC, TE cel = TE::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t &coarse_coords,
     const ParArrayND<Real, parthenon::VariableState> *,
     const ParArrayND<Real, parthenon::VariableState> *pfine) {
    if constexpr (fel != TE::F1 && fel != TE::F2 && fel != TE::F3) {
      return;
    } else {
      constexpr int me = static_cast<int>(fel) % 3;
      constexpr int next = (me + 1) % 3;
      constexpr int third = (me + 2) % 3;

      auto &fine = *pfine;
      const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

      const Real a[3] = {(coords.template Dxc<2>(fj) * coords.template Dxc<2>(fj) -
                          coords.template Dxc<3>(fk) * coords.template Dxc<3>(fk)) /
                             (coords.template Dxc<2>(fj) * coords.template Dxc<2>(fj) +
                              coords.template Dxc<3>(fk) * coords.template Dxc<3>(fk)),
                         (coords.template Dxc<3>(fk) * coords.template Dxc<3>(fk) -
                          coords.template Dxc<1>(fi) * coords.template Dxc<1>(fi)) /
                             (coords.template Dxc<3>(fk) * coords.template Dxc<3>(fk) +
                              coords.template Dxc<1>(fi) * coords.template Dxc<1>(fi)),
                         (coords.template Dxc<1>(fi) * coords.template Dxc<1>(fi) -
                          coords.template Dxc<2>(fj) * coords.template Dxc<2>(fj)) /
                             (coords.template Dxc<1>(fi) * coords.template Dxc<1>(fi) +
                              coords.template Dxc<2>(fj) * coords.template Dxc<2>(fj))};

      const Real coeff[4][4] = {{3 + a[next], 1 - a[next], 3 - a[third], 1 + a[third]},
                                {3 + a[next], 1 - a[next], 1 + a[third], 3 - a[third]},
                                {1 - a[next], 3 + a[next], 3 - a[third], 1 + a[third]},
                                {1 - a[next], 3 + a[next], 1 + a[third], 3 - a[third]}};

      constexpr int diff_k = (me == 2 && DIM > 2);
      constexpr int diff_j = (me == 1 && DIM > 1);
      constexpr int diff_i = (me == 0 && DIM > 0);

      for (int elem = 0; elem < 4; elem++) {
        const int off_i =
            (DIM > 0) ? (elem % 2) * (me == 1) + (elem / 2) * (me == 2) + (me == 0) : 0;
        const int off_j =
            (DIM > 1) ? (elem % 2) * (me == 2) + (elem / 2) * (me == 0) + (me == 1) : 0;
        const int off_k =
            (DIM > 2) ? (elem % 2) * (me == 0) + (elem / 2) * (me == 1) + (me == 2) : 0;

        fine(me, l, m, n, fk + off_k, fj + off_j, fi + off_i) =
            (0.5 * (fine(me, l, m, n, fk + off_k - diff_k, fj + off_j - diff_j,
                         fi + off_i - diff_i) *
                        coords.template Volume<fel>(fk + off_k - diff_k,
                                                    fj + off_j - diff_j,
                                                    fi + off_i - diff_i) +
                    fine(me, l, m, n, fk + off_k + diff_k, fj + off_j + diff_j,
                         fi + off_i + diff_i) *
                        coords.template Volume<fel>(fk + off_k + diff_k,
                                                    fj + off_j + diff_j,
                                                    fi + off_i + diff_i)) +
             1. / 16 *
                 (coeff[elem][0] *
                      FaceDiff<next, me, -1, DIM>(fine, coords, l, m, n, fk, fj, fi) +
                  coeff[elem][1] *
                      FaceDiff<next, me, third, DIM>(fine, coords, l, m, n, fk, fj, fi) +
                  coeff[elem][2] *
                      FaceDiff<third, me, -1, DIM>(fine, coords, l, m, n, fk, fj, fi) +
                  coeff[elem][3] * FaceDiff<third, me, next, DIM>(fine, coords, l, m, n,
                                                                  fk, fj, fi))) /
            coords.template Volume<fel>(fk + off_k, fj + off_j, fi + off_i);
      }
    }
  }
};

} // namespace b_ct

#endif // FLUID_B_CT_FUNCTIONS_HPP_
