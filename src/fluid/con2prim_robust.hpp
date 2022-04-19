//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef CON2PRIM_ROBUST_HPP_
#define CON2PRIM_ROBUST_HPP_

#include <cmath>
#include <fenv.h>

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <singularity-eos/eos/eos.hpp>

#include "con2prim_statistics.hpp"
#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"

#define CON2PRIM_ROBUST_PRINT_FAILURES 0

namespace con2prim_robust {

using namespace robust;

enum class ConToPrimStatus { success, failure };
struct FailFlags {
  static constexpr Real success = 1.0;
  static constexpr Real fail = 0.0;
};

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real q, const Real bsq, const Real bsq_rpsq,
           const Real rsq, const Real rbsq, const Real v0sq, const Real Ye,
           const singularity::EOS &eos, const fixup::Bounds &bnds, const Real x1,
           const Real x2, const Real x3)
      : D_(D), q_(q), bsq_(bsq), bsq_rpsq_(bsq_rpsq), rsq_(rsq), rbsq_(rbsq), v0sq_(v0sq),
        eos_(eos), bounds_(bnds), x1_(x1), x2_(x2), x3_(x3) {
    lambda_[0] = Ye;
    Real garbage = 0.0;
    bounds_.GetFloors(x1_, x2_, x3_, rho_floor_, garbage);
    bounds_.GetCeilings(x1_, x2_, x3_, gam_max_, e_max_);

#if !USE_C2P_ROBUST_FLOORS
    rho_floor_ = 1.e-20;
#endif
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real x_mu(const Real mu) { return 1.0 / (1.0 + mu * bsq_); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real rbarsq_mu(const Real mu, const Real x) {
    return x * (x * rsq_ + mu * (1.0 + x) * rbsq_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real qbar_mu(const Real mu, const Real x) {
    const Real mux = mu * x;
    return q_ - 0.5 * (bsq_ + mux * mux * bsq_rpsq_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real vhatsq_mu(const Real mu, const Real rbarsq) {
    const Real vsq_trial = mu * mu * rbarsq;
    if (vsq_trial > v0sq_) {
      used_gamma_max_ = true;
      return v0sq_;
    } else {
      used_gamma_max_ = false;
      return vsq_trial;
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real iWhat_mu(const Real vhatsq) { return std::sqrt(1.0 - vhatsq); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real rhohat_mu(const Real iWhat) {
    const Real rho_trial = D_ * iWhat;
    if (rho_trial <= rho_floor_) {
      used_density_floor_ = true;
      return rho_floor_;
    } else {
      used_density_floor_ = false;
      return rho_trial;
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real ehat_mu(const Real mu, const Real qbar, const Real rbarsq, const Real vhatsq,
               const Real What) {
    Real rho = rhohat_mu(1.0 / What);
    bounds_.GetFloors(x1_, x2_, x3_, rho, e_floor_);
    const Real ehat_trial =
        What * (qbar - mu * rbarsq) + vhatsq * What * What / (1.0 + What);
    used_energy_floor_ = false;
    used_energy_max_ = false;
    if (ehat_trial <= e_floor_) {
      used_energy_floor_ = true;
      return e_floor_;
    } else if (ehat_trial > e_max_) {
      used_energy_max_ = true;
      return e_max_;
    } else {
      return ehat_trial;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real mu) {
    const Real x = x_mu(mu);
    const Real rbarsq = rbarsq_mu(mu, x);
    const Real qbar = qbar_mu(mu, x);
    const Real vhatsq = vhatsq_mu(mu, rbarsq);
    const Real iWhat = iWhat_mu(vhatsq);
    const Real What = 1.0 / iWhat;
    Real rhohat = rhohat_mu(iWhat);
    Real ehat = ehat_mu(mu, qbar, rbarsq, vhatsq, What);
    const Real Phat = eos_.PressureFromDensityInternalEnergy(rhohat, ehat, lambda_);
    Real hhat = rhohat * (1.0 + ehat) + Phat;
    const Real ahat = Phat / make_positive(hhat - Phat);
    hhat /= rhohat;

    const Real nua = (1.0 + ahat) * (1.0 + ehat) * iWhat;
    const Real nub = (1.0 + ahat) * (1.0 + qbar - mu * rbarsq);
    const Real nuhat = std::max(nua, nub);

    const Real muhat = 1.0 / (nuhat + mu * rbarsq);
    return mu - muhat;
  }

  KOKKOS_INLINE_FUNCTION
  Real compute_upper_bound(const Real h0sq) {
    auto func = [=](const Real x) { return aux_func(x, h0sq); };
    root_find::RootFind root;
    return root.itp(func, 1.e-16, 1.0 / sqrt(h0sq), 1.e-3, -1.0);
  }

  KOKKOS_INLINE_FUNCTION
  bool used_density_floor() const { return used_density_floor_; }
  KOKKOS_INLINE_FUNCTION
  bool used_energy_floor() const { return used_energy_floor_; }
  KOKKOS_INLINE_FUNCTION
  bool used_energy_max() const { return used_energy_max_; }
  KOKKOS_INLINE_FUNCTION
  bool used_gamma_max() const { return used_gamma_max_; }
  KOKKOS_INLINE_FUNCTION
  bool used_bounds() const {
    return (used_density_floor_ || used_energy_floor_ || used_energy_max_ ||
            used_gamma_max_);
  }

 private:
  const Real D_, q_, bsq_, bsq_rpsq_, rsq_, rbsq_, v0sq_;
  const singularity::EOS &eos_;
  const fixup::Bounds &bounds_;
  const Real x1_, x2_, x3_;
  Real lambda_[2];
  Real rho_floor_, e_floor_, gam_max_, e_max_;
  bool used_density_floor_, used_energy_floor_, used_energy_max_, used_gamma_max_;

  KOKKOS_FORCEINLINE_FUNCTION
  Real aux_func(const Real mu, const Real h0sq) {
    const Real x = 1.0 / (1.0 + mu * bsq_);
    const Real rbarsq = x * (rsq_ * x + mu * (1.0 + x) * rbsq_);
    return mu * std::sqrt(h0sq + rbarsq) - 1.0;
  }
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
  Real &operator()(const int n) const { return var_(b_, n, k_, j_, i_); }
  const int b_, i_, j_, k_;

 private:
  const T &var_;
};

struct CellGeom {
  template <typename CoordinateSystem>
  KOKKOS_FUNCTION CellGeom(const CoordinateSystem &geom, const int k, const int j,
                           const int i)
      : gdet(geom.DetGamma(CellLocation::Cent, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, k, j, i)) {
    geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
    SPACELOOP2(m, n) { gcov[m][n] = gcov4[m + 1][n + 1]; }
    geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  }
  template <typename CoordinateSystem>
  CellGeom(const CoordinateSystem &geom, const int b, const int k, const int j,
           const int i)
      : gdet(geom.DetGamma(CellLocation::Cent, b, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, b, k, j, i)) {
    geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, gcov4);
    SPACELOOP2(m, n) { gcov[m][n] = gcov4[m + 1][n + 1]; }
    geom.MetricInverse(CellLocation::Cent, b, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, b, k, j, i, beta);
  }
  Real gcov[3][3];
  Real gcov4[4][4];
  Real gcon[3][3];
  Real beta[3];
  const Real gdet;
  const Real lapse;
};

template <typename Data_t, typename T>
class ConToPrim {
 public:
  ConToPrim(Data_t *rc, fixup::Bounds bnds, const Real tol, const int max_iterations)
      : bounds(bnds), var(rc->PackVariables(Vars(), imap)),
        prho(imap[fluid_prim::density].first), crho(imap[fluid_cons::density].first),
        pvel_lo(imap[fluid_prim::velocity].first),
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
        gm1(imap[fluid_prim::gamma1].first),
        c2p_mu(imap[internal_variables::c2p_mu].first), rel_tolerance(tol),
        max_iter(max_iterations), h0sq_(1.0) {}

  std::vector<std::string> Vars() {
    return std::vector<std::string>(
        {fluid_prim::density, fluid_cons::density, fluid_prim::velocity,
         fluid_cons::momentum, fluid_prim::energy, fluid_cons::energy, fluid_prim::bfield,
         fluid_cons::bfield, fluid_prim::ye, fluid_cons::ye, fluid_prim::pressure,
         fluid_prim::temperature, internal_variables::cell_signal_speed,
         fluid_prim::gamma1, internal_variables::c2p_mu});
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION ConToPrimStatus operator()(const CoordinateSystem &geom,
                                                    const singularity::EOS &eos,
                                                    const Coordinates_t &coords,
                                                    Args &&...args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    Real x1 = coords.x1v(std::forward<Args>(args)...);
    Real x2 = coords.x2v(std::forward<Args>(args)...);
    Real x3 = coords.x3v(std::forward<Args>(args)...);
    return solve(v, g, eos, x1, x2, x3);
  }

  int NumBlocks() { return var.GetDim(5); }

 private:
  fixup::Bounds bounds;
  PackIndexMap imap;
  const T var;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int pb_lo, pb_hi;
  const int cb_lo, cb_hi;
  const int pye, cye;
  const int prs, tmp, sig_lo, sig_hi, gm1, c2p_mu;
  const Real rel_tolerance;
  const int max_iter;
  const Real h0sq_;

  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus solve(const VarAccessor<T> &v, const CellGeom &g,
                        const singularity::EOS &eos, const Real x1, const Real x2,
                        const Real x3) const {
    int num_nans = std::isnan(v(crho)) + std::isnan(v(cmom_lo)) + std::isnan(v(ceng));
    if (num_nans > 0) return ConToPrimStatus::failure;
    const Real igdet = 1.0 / g.gdet;

    Real rhoflr = 0.0;
    Real epsflr;
    bounds.GetFloors(x1, x2, x3, rhoflr, epsflr);
    Real gam_max, eps_max;
    bounds.GetCeilings(x1, x2, x3, gam_max, eps_max);

    const Real D = v(crho) * igdet;
#if USE_VALENCIA
    const Real tau = v(ceng) * igdet;
#else
    Real Qcov[4] = {(v(ceng) - v(crho)) * igdet, v(cmom_lo) * igdet,
                    v(cmom_lo + 1) * igdet, v(cmom_lo + 2) * igdet};
    Real ncon[4] = {1. / g.lapse, -g.beta[0] / g.lapse, -g.beta[1] / g.lapse,
                    -g.beta[2] / g.lapse};
    Real tau = 0.;
    SPACETIMELOOP(mu) { tau -= Qcov[mu] * ncon[mu]; }
    tau -= D;
#endif // USE_VALENCIA
    const Real q = tau / D;

    if (pye > 0) v(pye) = v(cye) / v(crho);
    Real ye_local = (pye > 0) ? v(pye) : 0.5;

    Real bsq = 0.0;
    Real bsq_rpsq = 0.0;
    Real rbsq = 0.0;
    Real rsq = 0.0;
    Real bdotr = 0.0;
    Real rcon[3];
    // r_i
    Real rcov[] = {robust::ratio(v(cmom_lo), v(crho)),
                   robust::ratio(v(cmom_lo + 1), v(crho)),
                   robust::ratio(v(cmom_lo + 2), v(crho))};
    SPACELOOP(i) {
      rcon[i] = 0.0;
      SPACELOOP(j) { rcon[i] += g.gcon[i][j] * rcov[j]; }
      rsq += rcon[i] * rcov[i];
    }
    Real bu[] = {0.0, 0.0, 0.0};
    if (pb_hi > 0) {
      // first set the primitive b-fields
      SPACELOOP(i) { v(pb_lo + i) = v(cb_lo + i) * igdet; }
      const Real sD = 1.0 / std::sqrt(std::max(D, rhoflr));
      // b^i
      SPACELOOP(i) {
        bu[i] = v(pb_lo + i) * sD;
        bdotr += bu[i] * rcov[i];
      }
      SPACELOOP2(i, j) bsq += g.gcov[i][j] * bu[i] * bu[j];
      bsq = std::max(0.0, bsq);

      rbsq = bdotr * bdotr;
      bsq_rpsq = bsq * rsq - rbsq;
    }
    const Real zsq = rsq / h0sq_;
    Real v0sq = std::min(zsq / (1.0 + zsq), 1.0 - 1.0 / (gam_max * gam_max));

    Residual res(D, q, bsq, bsq_rpsq, rsq, rbsq, v0sq, ye_local, eos, bounds, x1, x2, x3);

    // find the upper bound
    // TODO(JCD): revisit this.  is it worth it to find the upper bound?
    //            Doesn't seem to be at a quick glance.
    // const Real mu_r = res.compute_upper_bound(h0sq_);
    // solve
    root_find::RootFind root(max_iter);
    const Real mu = root.regula_falsi(res, 0.0, 1.0, rel_tolerance, v(c2p_mu));
    v(c2p_mu) = mu;
#if CON2PRIM_STATISTICS
    con2prim_statistics::Stats::add(root.iteration_count);
#endif

    // now unwrap everything into primitive and conserved vars
    const Real x = res.x_mu(mu);
    const Real rbarsq = res.rbarsq_mu(mu, x);
    Real vsq = res.vhatsq_mu(mu, rbarsq);
    Real iW = res.iWhat_mu(vsq);
    v(prho) = res.rhohat_mu(iW);
    const Real qbar = res.qbar_mu(mu, x);
    Real W = 1.0 / iW;
    Real eos_lambda[2];
    if (pye > 0) eos_lambda[0] = v(pye);
    eos_lambda[1] = std::log10(v(tmp)); // initial guess
    v(peng) = res.ehat_mu(mu, qbar, rbarsq, vsq, W);
    v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng), eos_lambda);
    v(peng) *= v(prho);
    v(prs) = eos.PressureFromDensityTemperature(v(prho), v(tmp), eos_lambda);
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp), eos_lambda) / v(prs);
    PARTHENON_DEBUG_REQUIRE(v(prs) > robust::EPS(), "Pressure must be positive");

    Real vel[3];
    SPACELOOP(i) {
      vel[i] = W * mu * x * (rcon[i] + mu * bdotr * bu[i]);
      v(pvel_lo + i) = vel[i];
    }
    if (pb_hi > 0) {
      SPACELOOP(i) { bu[i] = v(pb_lo + i); }
    }

    Real sig[3];
    if (res.used_bounds()) {
      Real ye_prim = 0.0;
      if (pye > 0) ye_prim = v(pye);
      Real ye_cons;
      Real S[3];
      Real bcons[3];
      prim2con::p2c(v(prho), vel, bu, v(peng), ye_prim, v(prs), v(gm1), g.gcov4, g.gcon,
                    g.beta, g.lapse, g.gdet, v(crho), S, bcons, v(ceng), ye_cons, sig);
      SPACELOOP(i) { v(cmom_lo + i) = S[i]; }

      if (pye > 0) v(cye) = ye_cons;
    } else {
      // just compute signal speeds
      SPACELOOP(m) vel[m] *= iW;
      bsq = 0.0;
      if (pb_hi > 0) {
        Real bdotv = 0.0;
        SPACELOOP2(m, n) {
          bsq += bu[m] * bu[n] * g.gcov4[m + 1][n + 1];
          bdotv += bu[m] * vel[n] * g.gcov4[m + 1][n + 1];
        }
        bsq = iW * iW * bsq + bdotv * bdotv;
      }

      prim2con::signal_speeds(v(prho), v(peng), v(prs), bsq, vel, vsq, v(gm1), g.lapse,
                              g.beta, g.gcon, sig);
    }

    for (int i = 0; i < sig_hi - sig_lo + 1; i++) {
      v(sig_lo + i) = sig[i];
    }

    num_nans = std::isnan(v(crho)) + std::isnan(v(cmom_lo)) + std::isnan(v(ceng));

    if (num_nans > 0 || res.used_gamma_max()) {
      return ConToPrimStatus::failure;
    }
    return ConToPrimStatus::success;
  }
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>, VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>, MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, fixup::Bounds bounds,
                                  const Real tol, const int max_iter) {
  return C2P_Block_t(rc, bounds, tol, max_iter);
}
/*inline C2P_Mesh_t ConToPrimSetup(MeshData<Real> *rc) {
  return C2P_Mesh_t(rc);
}*/

} // namespace con2prim_robust

#endif // CON2PRIM_ROBUST_HPP_
