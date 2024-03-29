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

#include <interface/sparse_pack.hpp>
#include <parthenon/package.hpp>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

namespace fluid {
using namespace parthenon::package::prelude;

// Indices are upstairs
template <typename CoordinateSystem>
class StressEnergyTensorCon {
 public:
  // TODO(JMM): Should these be moved out of Geometry?
  static constexpr int ND = Geometry::NDFULL;
  static constexpr CellLocation loc = CellLocation::Cent;

  StressEnergyTensorCon() = default;

  template <typename Data>
  StressEnergyTensorCon(Data *rc) {
    namespace p = fluid_prim;
    using phoebus::MakePackDescriptor;
    static auto desc =
        MakePackDescriptor<p::density, p::velocity, p::energy, p::pressure, p::bfield>(
            rc);
    pack_ = desc.GetPack(rc);
    system_ = Geometry::GetCoordinateSystem(rc);
    ir_ = pack_.GetLowerBoundHost(0, p::density());
    iv_ = pack_.GetLowerBoundHost(0, p::velocity());
    iu_ = pack_.GetLowerBoundHost(0, p::energy());
    ip_ = pack_.GetLowerBoundHost(0, p::pressure());
    ib_ = pack_.GetLowerBoundHost(0, p::bfield());
  }

  // TODO(JMM): Assumes cell centers. If that needs to change, this
  // signature needs to change.
  // TODO(JMM): Should I use enable_if or static asserts or anything?
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void operator()(Real T[ND][ND], Args &&...args) const {
    static_assert(sizeof...(Args) >= 3, "Must at least have k, j, i");
    static_assert(sizeof...(Args) <= 4, "Must have no more than b, k, j, i");
    Real u[ND], b[ND], g[ND][ND], bsq;
    GetTmunuTerms_(u, b, bsq, g, std::forward<Args>(args)...);
    system_.SpacetimeMetricInverse(loc, std::forward<Args>(args)..., g);
    Real rho = GetVar_(ir_, std::forward<Args>(args)...);
    Real uu = GetVar_(iu_, std::forward<Args>(args)...);
    Real P = GetVar_(ip_, std::forward<Args>(args)...);
    Real A = rho + uu + P + bsq;
    Real B = P + 0.5 * bsq;
    SPACETIMELOOP2(mu, nu) {
      T[mu][nu] = A * u[mu] * u[nu] + B * g[mu][nu] - b[mu] * b[nu];
    }
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int b, const int k, const int j, const int i) const {
    return pack_(b, v, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int k, const int j, const int i) const {
    return pack_(0, v, k, j, i);
  }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real v_(int l, Args &&...args) const {
    return GetVar_(iv_ + l - 1, std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real b_(int l, Args &&...args) const {
    return (ib_ >= 0 ? GetVar_(ib_ + l - 1, std::forward<Args>(args)...) : 0.0);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void GetTmunuTerms_(Real u[ND], Real b[ND], Real &bsq,
                                             Real gscratch[4][4], Args &&...args) const {
    Real beta[ND - 1];
    Real Bdotv = 0.0;
    Real Bsq = 0.0;
    auto gammacov = reinterpret_cast<Real(*)[3]>(&gscratch[0][0]);
    system_.Metric(loc, std::forward<Args>(args)..., gammacov);
    Real vp[3] = {v_(1, std::forward<Args>(args)...), v_(2, std::forward<Args>(args)...),
                  v_(3, std::forward<Args>(args)...)};
    const Real W = phoebus::GetLorentzFactor(vp, gammacov);

    SPACELOOP2(ii, jj) {
      const Real &bi = b_(ii + 1, std::forward<Args>(args)...);
      const Real &bj = b_(jj + 1, std::forward<Args>(args)...);
      Bdotv += bi * (vp[jj] / W) * gammacov[ii][jj];
      Bsq += bi * bj * gammacov[ii][jj];
    }
    const Real iW = robust::ratio(1., W);

    Real alpha = system_.Lapse(loc, std::forward<Args>(args)...);
    system_.ContravariantShift(loc, std::forward<Args>(args)..., beta);
    u[0] = robust::ratio(W, std::abs(alpha));
    b[0] = u[0] * Bdotv;
    for (int l = 1; l < ND; ++l) {
      u[l] = vp[l - 1] - u[0] * beta[l - 1];
      b[l] = iW * (b_(l, std::forward<Args>(args)...) + alpha * b[0] * u[l]);
    }

    bsq = Bsq * iW * iW + Bdotv * Bdotv;
  }

  parthenon::SparsePack<fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
                        fluid_prim::pressure, fluid_prim::bfield>
      pack_;
  CoordinateSystem system_;
  int ir_, iv_, iu_, ip_, ib_;
};

using TmunuMesh = StressEnergyTensorCon<Geometry::CoordSysMesh>;
using TmunuMeshBlock = StressEnergyTensorCon<Geometry::CoordSysMeshBlock>;

TmunuMesh BuildStressEnergyTensor(MeshData<Real> *rc);
TmunuMeshBlock BuildStressEnergyTensor(MeshBlockData<Real> *rc);

} // namespace fluid

#endif // FLUID_TMUNU_HPP_
