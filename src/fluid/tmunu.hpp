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

#ifndef FLUID_TMUNU_HPP_
#define FLUID_TMUNU_HPP_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"

namespace fluid {

const std::vector<std::string> TMUNU_VARS = {
    fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
    fluid_prim::pressure, fluid_prim::bfield};
// Indices are upstairs
template <typename CoordinateSystem, typename Pack>
class StressEnergyTensorCon {
public:
  // TODO(JMM): Should these be moved out of Geometry?
  static constexpr int ND = Geometry::NDFULL;
  static constexpr Real SMALL = Geometry::SMALL;
  static constexpr CellLocation loc = CellLocation::Cent;

  StressEnergyTensorCon() = default;

  template <typename Data> StressEnergyTensorCon(Data *rc) {
    PackIndexMap imap;
    pack_ = rc->PackVariables(TMUNU_VARS, imap);
    system_ = Geometry::GetCoordinateSystem(rc);

    ir_ = imap[fluid_prim::density].first;
    iv_ = imap[fluid_prim::velocity].first;
    iu_ = imap[fluid_prim::energy].first;
    ip_ = imap[fluid_prim::pressure].first;
    ib_ = imap[fluid_prim::bfield].first;
  }

  // TODO(JMM): Assumes cell centers. If that needs to change, this
  // signature needs to change.
  // TODO(JMM): Should I use enable_if or static asserts or anything?
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void operator()(Real T[ND][ND],
                                         Args &&... args) const {
    static_assert(sizeof...(Args) >= 3, "Must at least have k, j, i");
    static_assert(sizeof...(Args) <= 4, "Must have no more than b, k, j, i");
    Real u[ND], b[ND], g[ND][ND], bsq;
    GetTmunuTerms_(u, b, bsq, g, std::forward<Args>(args)...);
    system_.SpacetimeMetricInverse(loc, std::forward<Args>(args)..., g);
    Real rho = GetVar_(ir_, std::forward<Args>(args)...);
    Real uu = GetVar_(iu_, std::forward<Args>(args)...);
    Real P = GetVar_(ip_, std::forward<Args>(args)...);
    int k = std::get<0>(std::forward_as_tuple(std::forward<Args>(args)...));
    int j = std::get<1>(std::forward_as_tuple(std::forward<Args>(args)...));
    int i = std::get<2>(std::forward_as_tuple(std::forward<Args>(args)...));
    if (i == 120 && j == 120) {
      printf("tmunu: %e %e %e vel: %e %e %e\n", rho, uu, P,
        v_(1, std::forward<Args>(args)...),
        v_(2, std::forward<Args>(args)...),
        v_(3, std::forward<Args>(args)...));
    }
    Real A = rho + uu + P + bsq;
    Real B = P + 0.5 * bsq;
    for (int mu = 0; mu < ND; ++mu) {
      for (int nu = 0; nu < ND; ++nu) {
        T[mu][nu] = A * u[mu] * u[nu] + B * g[mu][nu] - b[mu] * b[nu];
      }
    }
  }

private:
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int b, const int k, const int j,
               const int i) const {
    return pack_(b, v, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int k, const int j, const int i) const {
    return pack_(v, k, j, i);
  }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real v_(int l, Args &&... args) const {
    return GetVar_(iv_ + l - 1, std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real b_(int l, Args &&... args) const {
    return (ib_ > 0 ? GetVar_(ib_ + l - 1, std::forward<Args>(args)...) : 0.0);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void GetTmunuTerms_(Real u[ND], Real b[ND], Real &bsq,
                                             Real gscratch[4][4],
                                             Args &&... args) const {
    Real beta[ND - 1];
    Real W = 1.0;
    Real Bdotv = 0.0;
    Real Bsq = 0.0;
    auto gcov = reinterpret_cast<Real(*)[3]>(&gscratch[0][0]);
    system_.Metric(loc, std::forward<Args>(args)..., gcov);
    for (int l = 1; l < ND; ++l) {
      for (int m = 1; m < ND; ++m) {
        const Real gamma = gcov[l - 1][m - 1];
        const Real &vl = v_(l, std::forward<Args>(args)...);
        const Real &vm = v_(m, std::forward<Args>(args)...);
        W -= vl * vm * gamma;
        const Real &bl = b_(l, std::forward<Args>(args)...);
        const Real &bm = b_(m, std::forward<Args>(args)...);
        Bdotv += bl * vm * gamma;
        Bsq += bl * bm * gamma;
      }
    }
    const Real iW = std::sqrt(std::abs(W));
    W = 1. / (iW + SMALL);

    Real alpha = system_.Lapse(loc, std::forward<Args>(args)...);
    system_.ContravariantShift(loc, std::forward<Args>(args)..., beta);
    u[0] = W / (std::abs(alpha) + SMALL);
    b[0] = u[0] * Bdotv;
    for (int l = 1; l < ND; ++l) {
      u[l] = W * v_(l, std::forward<Args>(args)...) - u[0] * beta[l - 1];
      b[l] = iW * (b_(l, std::forward<Args>(args)...) + alpha * b[0] * u[l]);
    }

    b[0] *= alpha;
    bsq = (Bsq + b[0] * b[0]) * iW * iW;
  }

  Pack pack_;
  CoordinateSystem system_;
  int ir_, iv_, iu_, ip_, ib_;
};

// using TmunuMesh = StressEnergyTensorCon<MeshBlockPack<VariablePack<Real>>>;
using TmunuMeshBlock =
    StressEnergyTensorCon<Geometry::CoordSysMeshBlock, VariablePack<Real>>;

// TmunuMesh BuildStressEnergyTensor(MeshData<Real> *rc) { return TmunuMesh(rc);
// }
TmunuMeshBlock BuildStressEnergyTensor(MeshBlockData<Real> *rc) {
  return TmunuMeshBlock(rc);
}

} // namespace fluid

#endif // FLUID_TMUNU_HPP_
