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

#ifndef CON2PRIM_MM_HPP_
#define CON2PRIM_MM_HPP_

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <singularity-eos/eos/eos.hpp>

#include "con2prim.hpp"
#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"

namespace con2prim_mm {

  // TODO(BRR) Use singularity-eos
  constexpr Real gam = 5./3.;

using namespace con2prim;
using namespace root_find;

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real Ssq, const Real tau, const Real Bsq, const Real SdB)
      : D_(D), Ssq_(Ssq), tau_(tau), Bsq_(Bsq), SdB_(SdB) {}

  KOKKOS_INLINE_FUNCTION
  Real GetPressure() { return P_; }

  KOKKOS_INLINE_FUNCTION
  Real GetLorentzFactor() { return gamma_; }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real Wp) {
    SetLorentzFactorEnthalpy(Wp);
    SetPressure();
    Real resid = Wp - tau_ - P_ + Bsq_ / 2. +
                 (Bsq_ * Ssq_ - SdB_ * SdB_) / (2. * pow(Bsq_ + Wp + D_, 2.));
    return resid;
  }

 private:
  const Real D_, Ssq_, tau_, Bsq_, SdB_;

  Real gamma_;
  Real wp_; // u + P
  Real P_;

  KOKKOS_INLINE_FUNCTION
  void SetLorentzFactorEnthalpy(const Real Wp) {
    const Real W = Wp + D_;
    const Real WpB = pow(W + Bsq_, 2);
    const Real A = Ssq_ / WpB + SdB_ * SdB_ / (W * W * WpB) * (2. * W + Bsq_);
    const Real xsq = A / (1. - A);
    gamma_ = sqrt(1. + xsq);
    wp_ = 1. / (gamma_ * gamma_) * (Wp - D_ * xsq / (1. + gamma_));
  }

  KOKKOS_INLINE_FUNCTION
  void SetPressure() {
    // TODO(BRR) Use singularity calls for this, right now 5/3 gamma law only
    P_ = wp_ * (gam - 1.) / gam;
  }
};

struct CellGeom {
  template <typename CoordinateSystem>
  KOKKOS_FUNCTION CellGeom(const CoordinateSystem &geom, const int k, const int j,
                           const int i)
      : gammadet(geom.DetGamma(CellLocation::Cent, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, k, j, i)) {
    geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, k, j, i, gammacon);
    geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  }
  template <typename CoordinateSystem>
  CellGeom(const CoordinateSystem &geom, const int b, const int k, const int j,
           const int i)
      : gammadet(geom.DetGamma(CellLocation::Cent, b, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, b, k, j, i)) {
    geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, b, k, j, i, gammacon);
    geom.ContravariantShift(CellLocation::Cent, b, k, j, i, beta);
  }
  Real gcov[4][4];
  Real gammacon[3][3];
  Real beta[3];
  const Real gammadet;
  const Real lapse;
};

template <typename Data_t, typename T>
class ConToPrim {
 public:
  ConToPrim(Data_t *rc, const Real tol, const int max_iterations)
      : var(rc->PackVariables(Vars(), imap)), prho(imap[fluid_prim::density].first),
        crho(imap[fluid_cons::density].first), pvel_lo(imap[fluid_prim::velocity].first),
        pvel_hi(imap[fluid_prim::velocity].second),
        cmom_lo(imap[fluid_cons::momentum].first),
        cmom_hi(imap[fluid_cons::momentum].second), peng(imap[fluid_prim::energy].first),
        ceng(imap[fluid_cons::energy].first), pb_lo(imap[fluid_prim::bfield].first),
        pb_hi(imap[fluid_prim::bfield].second), cb_lo(imap[fluid_cons::bfield].first),
        cb_hi(imap[fluid_cons::bfield].second), pye(imap[fluid_prim::ye].second),
        cye(imap[fluid_cons::ye].second), prs(imap[fluid_prim::pressure].first),
        tmp(imap[fluid_prim::temperature].first),
        sig_lo(imap[internal_variables::cell_signal_speed].first),
        sig_hi(imap[internal_variables::cell_signal_speed].second),
        gm1(imap[fluid_prim::gamma1].first), rel_tolerance(tol), max_iter(max_iterations),
        h0sq_(1.0) {}

  std::vector<std::string> Vars() {
    return std::vector<std::string>(
        {fluid_prim::density, fluid_cons::density, fluid_prim::velocity,
         fluid_cons::momentum, fluid_prim::energy, fluid_cons::energy, fluid_prim::bfield,
         fluid_cons::bfield, fluid_prim::ye, fluid_cons::ye, fluid_prim::pressure,
         fluid_prim::temperature, internal_variables::cell_signal_speed,
         fluid_prim::gamma1});
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION ConToPrimStatus operator()(const CoordinateSystem &geom,
                                                    const singularity::EOS &eos,
                                                    const Coordinates_t &coords,
                                                    Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    Real x1 = coords.x1v(std::forward<Args>(args)...);
    Real x2 = coords.x2v(std::forward<Args>(args)...);
    Real x3 = coords.x3v(std::forward<Args>(args)...);
    return solve(v, g, eos, x1, x2, x3);
  }

  int NumBlocks() { return var.GetDim(5); }

 private:
  PackIndexMap imap;
  const T var;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int pb_lo, pb_hi;
  const int cb_lo, cb_hi;
  const int pye, cye;
  const int prs, tmp, sig_lo, sig_hi, gm1;
  const Real rel_tolerance;
  const int max_iter;
  const Real h0sq_;

  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus solve(const VarAccessor<T> &v, const CellGeom &g,
                        const singularity::EOS &eos, const Real x1, const Real x2,
                        const Real x3) const {
    // Update primitive B fields first if available
    if (pb_hi > 0) {
      SPACELOOP(i) { v(pb_lo + i) = v(cb_lo + i) / g.gammadet; }
    }

    // Catch negative density
    if (v(crho) < 0.) {
      return ConToPrimStatus::failure;
    }

    // Update Ye if available
    if (pye > 0) {
      v(pye) = v(cye) / v(crho);
    }

    // Constants
    const Real D = v(crho) / g.gammadet;
    const Real Scov[3] = {v(cmom_lo) / g.gammadet, v(cmom_lo + 1) / g.gammadet,
                          v(cmom_lo + 2) / g.gammadet};
    Real Bcon[3] = {0.};
    for (int ii = pb_lo; ii <= pb_hi; ii++) {
      Bcon[ii - pb_lo] = v(ii);
    }
    Real Ssq = 0.;
    Real Bsq = 0.;
    Real SdB = 0.;
    Real Bcov[3] = {0.};
    SPACELOOP(ii) {
      SdB += Scov[ii] * Bcon[ii];
      SPACELOOP(jj) {
        Ssq += g.gammacon[ii][jj] * Scov[ii] * Scov[jj];
        Bsq += g.gcov[ii + 1][jj + 1] * Bcon[ii] * Bcon[jj];
        Bcov[ii] += g.gcov[ii + 1][jj + 1] * Bcon[jj];
      }
    }

    const Real ncov[4] = {-g.lapse, 0., 0., 0.};
    const Real ncon[4] = {1. / g.lapse, -g.beta[0] / g.lapse, -g.beta[1] / g.lapse,
                          -g.beta[2] / g.lapse};

// Calculate Eulerian frame non-rest-mass energy density tau
#if USE_VALENCIA
    const Real tau = v(ceng) / g.gammadet;
#else
    Real Qcov[4] = {(v(ceng) - v(crho)) / g.gammadet, v(cmom_lo) / g.gammadet,
                    v(cmom_lo + 1) / g.gammadet, v(cmom_lo + 2) / g.gammadet};
    Real tau = 0.;
    SPACETIMELOOP(mu) { tau -= Qcov[mu] * ncon[mu]; }
    tau -= D;
#endif // USE_VALENCIA

    Residual res(D, Ssq, tau, Bsq, SdB);

    RootfindStatus status;
    Real Wp = root_find::itp(res, 0.0, 10.*D, rel_tolerance, max_iter, &status);
    if (v.i_ == 128 && v.j_ == 128) {
      printf("Wp: %e status: %i\n", Wp, static_cast<int>(status));
      Real Wp_real = 1.13024;
      Wp = Wp_real;
      const Real W = Wp + D;
      const Real WpB = pow(W + Bsq, 2);
      const Real A = Ssq / WpB + SdB * SdB / (W * W * WpB) * (2. * W + Bsq);
      const Real xsq = A / (1. - A);
      printf("W: %e WpB: %e Ssq: %e SdB: %e\n",
        W, WpB, Ssq, SdB);
      const Real gamma = sqrt(1. + xsq);
      const Real wp = 1. / (gamma * gamma) * (Wp - D * xsq / (1. + gamma));
      printf("xsq: %e gamma: %e wp: %e\n", xsq, gamma, wp);
    }
    if (status == RootfindStatus::failure) {
      return ConToPrimStatus::failure;
    }
    const Real gamma = res.GetLorentzFactor();
    const Real P = res.GetPressure();
    if (v.i_ == 128 && v.j_ == 128) printf("gamma: %e P: %e\n", gamma, P);

    v(prho) = D / gamma;
    v(prs) = P;
    // TODO(BRR) use singularity for this
    v(peng) = P / (gam - 1.);
    // v(peng) = v(prho) * eos.InternalEnergyFromDensityPressure(v(prho), P);
    v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng));
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp)) / v(prs);

    Real vcov[3];
    SPACELOOP(ii) {
      vcov[ii] = 1. / (Wp + D + Bsq) * (Scov[ii] + SdB / (Wp + D) * Bcov[ii]);
    }
    Real vcon[3] = {0.};
    SPACELOOP2(ii, jj) { vcon[ii] += g.gammacon[ii][jj] * vcov[jj]; }
    SPACELOOP(ii) { v(pvel_lo + ii) = gamma * vcon[ii]; }

    // Set signal speed
    Real Bdv = 0.;
    SPACELOOP(ii) { Bdv += Bcov[ii] * vcon[ii]; }
    Real bcon[] = {gamma * Bdv / g.lapse, 0.0, 0.0, 0.0};
    const Real bsq = (Bsq + g.lapse * g.lapse * bcon[0] * bcon[0]) / gamma / gamma;
    Real sig[3] = {0.};
    prim2con::CalculateSignalSpeed(v(prho), v(peng), v(prs), vcon, bsq, gamma, gm1,
                                   g.gcov, g.gammacon, g.beta, g.lapse, sig);
    for (int i = 0; i < sig_hi - sig_lo + 1; i++) {
      v(sig_lo + i) = sig[i];
    }

    return ConToPrimStatus::success;
  }
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>, VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>, MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol,
                                  const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}

} // namespace con2prim_mm

#endif // CON2PRIM_MM_HPP_
