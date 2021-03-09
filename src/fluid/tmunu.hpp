#ifndef FLUID_TMUNU_HPP_
#define FLUID_TMUNU_HPP_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace fluid {

// TODO(JMM): Add B-fields
const std::vector<std::string> TMUNU_VARS = {"p.density", "p.velocity",
                                             "p.energy", "pressure",
                                             "p.bfield"};
// Indices are upstairs
template <typename Pack> class StressEnergyTensorCon {
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

    ir_ = imap["p.density"].first;
    iv_ = imap["p.velocity"].first;
    iu_ = imap["p.energy"].first;
    ip_ = imap["pressure"].first;
    ib_ = imap["p.field"].first;
  }

  // TODO(JMM): Assumes cell centers. If that needs to change, this
  // signature needs to change.
  // TODO(JMM): Should I use enable_if or static asserts or anything?
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void operator()(Real T[ND][ND], Args &&... args) const {
    static_assert(sizeof...(Args) >= 3, "Must at least have k, j, i");
    static_assert(sizeof...(Args) <= 4, "Must have no more than b, k, j, i");
    Real u[ND], b[ND], g[ND][ND], bsq;
    GetTmunuTerms_(u, b, bsq, g, std::forward<Args>(args)...);
    system_.SpacetimeMetricInverse(loc, std::forward<Args>(args)..., g);
    Real rho = GetVar_(ir_, std::forward<Args>(args)...);
    Real uu = GetVar_(iu_, std::forward<Args>(args)...);
    Real P = GetVar_(ip_, std::forward<Args>(args)...);
    // TODO(JMM): Add B-fields in here
    Real A = rho + uu + P + bsq;
    Real B = P + 0.5*bsq;
    for (int mu = 0; mu < ND; ++mu) {
      for (int nu = 0; nu < ND; ++nu) {
        T[mu][nu] = A * u[mu] * u[nu] + B * g[mu][nu] - b[mu]*b[nu];
      }
    }
  }

private:
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int b, const int k, const int j, const int i) const {
    return pack_(b, v, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real GetVar_(int v, const int k, const int j, const int i) const { return pack_(v, k, j, i); }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real v_(int l, Args &&... args) const {
    return GetVar_(iv_ + l - 1, std::forward<Args>(args)...);
  }

  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real b_(int l, Args &&... args) const {
    return (ib_ > 0 ? GetVar_(ib_ + l - 1, std::forward<Args>(args)...) : 0.0);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void GetTmunuTerms_(Real u[ND], Real b[ND], Real &bsq, Real gscratch[4][4], Args &&... args) const {
    Real beta[ND - 1];
    Real W = 1.0;
    Real Bdotv = 0.0;
    Real Bsq = 0.0;
    auto gcov = reinterpret_cast<Real (*)[3]>(gscratch);
    system_.Metric(loc,std::forward<Args>(args)..., gcov);
    for (int l = 1; l < ND; ++l) {
      for (int m = 1; m < ND; ++m) {
        const Real gamma = gcov[l-1][m-1];
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
    b[0] = u[0]*Bdotv;
    for (int l = 1; l < ND; ++l) {
      u[l] = W * v_(l, std::forward<Args>(args)...) - u[0] * beta[l-1];
      b[l] = iW * (b_(l, std::forward<Args>(args)...) + alpha*b[0]*u[l]);
    }

    b[0] *= alpha;
    bsq = (Bsq + b[0]*b[0])*iW*iW;
  }

  Pack pack_;
  Geometry::CoordinateSystem system_;
  int ir_, iv_, iu_, ip_, ib_;
};

//using TmunuMesh = StressEnergyTensorCon<MeshBlockPack<VariablePack<Real>>>;
using TmunuMeshBlock = StressEnergyTensorCon<VariablePack<Real>>;

//TmunuMesh BuildStressEnergyTensor(MeshData<Real> *rc) { return TmunuMesh(rc); }
TmunuMeshBlock BuildStressEnergyTensor(MeshBlockData<Real> *rc) {
  return TmunuMeshBlock(rc);
}

} // namespace fluid

#endif // FLUID_TMUNU_HPP_
