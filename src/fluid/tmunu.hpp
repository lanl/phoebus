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
                                             "p.energy", "pressure"};
// Indices are upstairs
template <typename Pack> class StressEnergyTensorCon {
public:
  // TODO(JMM): Should these be moved out of Geometry?
  static constexpr int ND = Geometry::CoordinateSystem::NDFULL;
  static constexpr Real SMALL = Geometry::SMALL;
  static constexpr CellLocation loc = CellLocation::Cent;

  KOKKOS_INLINE_FUNCTION
  StressEnergyTensorCon() = default;

  template <typename Data> StressEnergyTensorCon(Data *rc) {
    PackIndexMap imap;
    pack_ = rc->PackVariables(TMUNU_VARS, imap);
    system_ = Geometry::GetCoordinateSystem(rc);

    // TODO(JMM): Perhaps this should be a macro
    for (auto &name : TMUNU_VARS) {
      PARTHENON_REQUIRE_THROWS(imap[name].second >= 0, name + " found");
    }
    ir_ = imap["p.density"].first;
    iv_ = imap["p.velocity"].first;
    iu_ = imap["p.energy"].first;
    ip_ = imap["pressure"].first;
  }

  // TODO(JMM): Assumes cell centers. If that needs to change, this
  // signature needs to change.
  // TODO(JMM): Should I use enable_if or static asserts or anything?
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void operator()(Real T[ND][ND], Args &&... args) const {
    static_assert(sizeof...(Args) >= 3, "Must at least have k, j, i");
    static_assert(sizeof...(Args) <= 4, "Must have no more than b, k, j, i");
    Real u[ND], g[ND][ND];
    system_.SpacetimeMetricInverse(loc, std::forward<Args>(args)..., g);
    GetFourVelocity_(u, std::forward<Args>(args)...);
    Real rho = GetVar_(ir_, std::forward<Args>(args)...);
    Real uu = GetVar_(iu_, std::forward<Args>(args)...);
    Real P = GetVar_(ip_, std::forward<Args>(args)...);
    // TODO(JMM): Add B-fields in here
    Real A = rho + uu + P; // + b^2
    Real B = P;            // + b^2/2
    for (int mu = 0; mu < ND; ++mu) {
      for (int nu = 0; nu < ND; ++nu) {
        T[mu][nu] = A * u[mu] * u[nu] + B * g[mu][nu];
      }
    }
  }

private:
  KOKKOS_INLINE_FUNCTION
  Real GetVar_(int v, const int b, const int k, const int j, const int i) const {
    return pack_(b, v, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real GetVar_(int v, const int k, const int j, const int i) const { return pack_(v, k, j, i); }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION Real v_(int l, Args &&... args) const {
    return GetVar_(iv_ + l, std::forward<Args>(args)...);
  }

  // TODO(JMM): Should these be available elsewhere/more publicly?
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION Real GetLorentzFactor_(Args &&... args) const {
    Real W = 1;
    for (int l = 1; l < ND; ++l) {
      for (int m = 1; m < ND; ++m) {
        Real gamma = system_.Metric(l, m, loc, std::forward<Args>(args)...);
        Real vl = v_(l, std::forward<Args>(args)...);
        Real vm = v_(m, std::forward<Args>(args)...);
        W -= vl * vm * gamma;
      }
    }
    W = 1. / std::sqrt(std::abs(W) + SMALL);
    return W;
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void GetFourVelocity_(Real u[ND], Args &&... args) const {
    Real beta[ND - 1];
    Real W = GetLorentzFactor_(std::forward<Args>(args)...);
    Real alpha = system_.Lapse(loc, std::forward<Args>(args)...);
    system_.ContravariantShift(loc, std::forward<Args>(args)..., beta);
    u[0] = W / (std::abs(alpha) + SMALL);
    for (int l = 1; l < ND; ++l) {
      u[l] = W * v_(l - 1, std::forward<Args>(args)...) - u[0] * beta[l-1];
    }
  }

  Pack pack_;
  Geometry::CoordinateSystem system_;
  int ir_, iv_, iu_, ip_;
};

//using TmunuMesh = StressEnergyTensorCon<MeshBlockPack<VariablePack<Real>>>;
using TmunuMeshBlock = StressEnergyTensorCon<VariablePack<Real>>;

//TmunuMesh BuildStressEnergyTensor(MeshData<Real> *rc) { return TmunuMesh(rc); }
TmunuMeshBlock BuildStressEnergyTensor(MeshBlockData<Real> *rc) {
  return TmunuMeshBlock(rc);
}

} // namespace fluid

#endif // FLUID_TMUNU_HPP_
