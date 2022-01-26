//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef CON2PRIM_HPP_
#define CON2PRIM_HPP_

#include <limits>

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <singularity-eos/eos/eos.hpp>

#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

namespace con2prim {

enum class ConToPrimStatus { success, failure };
struct FailFlags {
  static constexpr Real success = 0.0;
  static constexpr Real fail = 1.0;
};

class Residual {
public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real tau, const Real Bsq, const Real Ssq,
           const Real BdotS, const singularity::EOS &eos)
      : D_(D), tau_(tau), Bsq_(Bsq), Ssq_(Ssq), BdotSsq_(BdotS * BdotS),
        Ye_(std::numeric_limits<Real>::signaling_NaN()), eos_(eos) {}
  KOKKOS_FUNCTION
  Residual(const Real D, const Real tau, const Real Bsq, const Real Ssq,
           const Real BdotS, const Real Ye, const singularity::EOS &eos)
      : D_(D), tau_(tau), Bsq_(Bsq), Ssq_(Ssq), BdotSsq_(BdotS * BdotS),
        Ye_(Ye), eos_(eos) {
    lambda_[0] = Ye_;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real sfunc(const Real z, const Real Wp) const {
    Real zBsq = (z + Bsq_);
    zBsq *= zBsq;
    return (zBsq - Ssq_ - (2 * z + Bsq_) * BdotSsq_ / (z * z)) * Wp * Wp - zBsq;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real taufunc(const Real z, const Real Wp, const Real p) {
    return (tau_ + D_ - z - Bsq_ + BdotSsq_ / (2.0 * z * z) + p) * Wp * Wp +
           0.5 * Bsq_;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const Real rho, const Real Temp, Real res[2]) {
    const Real p = eos_.PressureFromDensityTemperature(rho, Temp, lambda_);
    const Real sie = eos_.InternalEnergyFromDensityTemperature(rho, Temp, lambda_);
    const Real Wp = D_ / rho;
    const Real z = (rho * (1.0 + sie) + p) * Wp * Wp;
    res[0] = sfunc(z, Wp);
    res[1] = taufunc(z, Wp, p);
  }
#ifndef NDEBUG
  KOKKOS_INLINE_FUNCTION
  void print() {
    printf("Residual Report: %16.14g %16.14g %16.14g %16.14g %16.14g\n", D_,
           tau_, Bsq_, Ssq_, BdotSsq_);
  }
#endif
private:
  const singularity::EOS &eos_;
  const Real D_, tau_, Bsq_, Ssq_, BdotSsq_, Ye_;
  Real lambda_[2] = {0., 0.};
};

template <typename T> class VarAccessor {
public:
  KOKKOS_FUNCTION
  VarAccessor(const T &var, const int k, const int j, const int i)
      : var_(var), b_(0), k_(k), j_(j), i_(i) {}
  VarAccessor(const T &var, const int b, const int k, const int j, const int i)
      : var_(var), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int n) const { return var_(b_, n, k_, j_, i_); }

private:
  const T &var_;
  const int b_, i_, j_, k_;
};

struct CellGeom {
  template <typename CoordinateSystem>
  KOKKOS_FUNCTION CellGeom(const CoordinateSystem &geom, const int k,
                           const int j, const int i)
      : gdet(geom.DetGamma(CellLocation::Cent, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, k, j, i)) {
    geom.Metric(CellLocation::Cent, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  }
  template <typename CoordinateSystem>
  CellGeom(const CoordinateSystem &geom, const int b, const int k, const int j,
           const int i)
      : gdet(geom.DetGamma(CellLocation::Cent, b, k, j, i)),
        lapse(geom.Lapse(CellLocation::Cent, b, k, j, i)) {
    geom.Metric(CellLocation::Cent, b, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, b, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, b, k, j, i, beta);
  }
  Real gcov[3][3];
  Real gcon[3][3];
  Real beta[3];
  const Real gdet;
  const Real lapse;
};

template <typename Data_t, typename T> class ConToPrim {
public:
  ConToPrim(Data_t *rc, const Real tol, const int max_iterations)
      : ConToPrim(rc, PackIndexMap(), tol, max_iterations) {}
  ConToPrim(Data_t *rc, PackIndexMap imap, const Real tol,
            const int max_iterations)
      : var(rc->PackVariables(Vars(), imap)),
        prho(imap[fluid_prim::density].first),
        crho(imap[fluid_cons::density].first),
        pvel_lo(imap[fluid_prim::velocity].first),
        pvel_hi(imap[fluid_prim::velocity].second),
        cmom_lo(imap[fluid_cons::momentum].first),
        cmom_hi(imap[fluid_cons::momentum].second),
        peng(imap[fluid_prim::energy].first),
        ceng(imap[fluid_cons::energy].first),
        pb_lo(imap[fluid_prim::bfield].first),
        pb_hi(imap[fluid_prim::bfield].second),
        cb_lo(imap[fluid_cons::bfield].first),
        cb_hi(imap[fluid_cons::bfield].second),
        pye(imap[fluid_prim::ye].second), cye(imap[fluid_cons::ye].second),
        prs(imap[fluid_prim::pressure].first),
        tmp(imap[fluid_prim::temperature].first),
        sig_lo(imap[internal_variables::cell_signal_speed].first),
        sig_hi(imap[internal_variables::cell_signal_speed].second),
        gm1(imap[fluid_prim::gamma1].first),
        scr_lo(imap[internal_variables::c2p_scratch].first), rel_tolerance(tol),
        max_iter(max_iterations) {}

  std::vector<std::string> Vars() {
    return std::vector<std::string>(
        {fluid_prim::density, fluid_cons::density, fluid_prim::velocity,
         fluid_cons::momentum, fluid_prim::energy, fluid_cons::energy,
         fluid_prim::bfield, fluid_cons::bfield, fluid_prim::ye, fluid_cons::ye,
         fluid_prim::pressure, fluid_prim::temperature,
         internal_variables::cell_signal_speed, fluid_prim::gamma1,
         internal_variables::c2p_scratch});
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION void Setup(const CoordinateSystem &geom,
                                    Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    setup(v, g);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION ConToPrimStatus operator()(const singularity::EOS &eos,
                                                    Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    return solve(v, eos);
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION void Finalize(const singularity::EOS &eos,
                                       const CoordinateSystem &geom,
                                       Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    finalize(v, g, eos);
  }

  int NumBlocks() { return var.GetDim(5); }

private:
  const T var;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int pb_lo, pb_hi;
  const int cb_lo, cb_hi;
  const int pye, cye;
  const int prs, tmp, sig_lo, sig_hi, gm1, scr_lo;
  const Real rel_tolerance;
  const int max_iter;
  constexpr static int iD = 0;
  constexpr static int itau = 1;
  constexpr static int iBsq = 2;
  constexpr static int iSsq = 3;
  constexpr static int iBdotS = 4;

  KOKKOS_INLINE_FUNCTION
  void finalize(const VarAccessor<T> &v, const CellGeom &g,
                const singularity::EOS &eos) const {
    const Real igdet = 1.0 / g.gdet;
    const Real &D = v(scr_lo + iD);
    const Real &Bsq = v(scr_lo + iBsq);
    const Real &BdotS = v(scr_lo + iBdotS);
    Real &rho_guess = v(prho);
    Real &T_guess = v(tmp);
    double lambda[2] = {0., 0.};
    if (pye >= 0) {
      lambda[0] = v(pye);
    }
    v(prs) = eos.PressureFromDensityTemperature(rho_guess, T_guess, lambda);
    v(peng) = rho_guess *
              eos.InternalEnergyFromDensityTemperature(rho_guess, T_guess, lambda);
    const Real H = rho_guess + v(peng) + v(prs);
    v(gm1) = eos.BulkModulusFromDensityTemperature(rho_guess, T_guess, lambda) / v(prs);

    const Real W = D / rho_guess;
    const Real W2 = W * W;
    const Real z = (rho_guess + v(peng) + v(prs)) * W2;
    const Real izbsq = 1. / (z + Bsq);
    SPACELOOP(i) {
      Real sconi = 0.0;
      SPACELOOP(j) {
        sconi += g.gcon[i][j] * v(cmom_lo + j);
      }
      sconi *= igdet;
      if (pb_lo > 0) {
        v(pvel_lo + i) = izbsq * (sconi + BdotS * v(pb_lo + i) / z);
      } else {
        v(pvel_lo + i) = izbsq * sconi;
      }
    }

    // cell-centered signal speeds
    Real vasq = BdotS / (H * W); // this is just bcon[0]*lapse
    vasq = (Bsq + vasq * vasq);  // this is now b^2 * W^2
    vasq /= (H + vasq);          // now this is the alven speed squared
    Real cssq = v(gm1) * v(prs) / H;
    cssq += vasq - cssq * vasq; // estimate of fast magneto-sonic speed
    const Real vsq = 1.0 - 1.0 / (W2);
    const Real vcoff = g.lapse / (1.0 - vsq * cssq);
    for (int i = 0; i < sig_hi - sig_lo + 1; i++) {
      const Real vpm = sqrt(cssq * (1.0 - vsq) *
                            (g.gcon[i][i] * (1.0 - vsq * cssq) -
                             v(pvel_lo + i) * v(pvel_lo + i) * (1.0 - cssq)));
      Real vp = vcoff * (v(pvel_lo + i) * (1.0 - cssq) + vpm) - g.beta[i];
      Real vm = vcoff * (v(pvel_lo + i) * (1.0 - cssq) - vpm) - g.beta[i];
      v(sig_lo + i) = std::max(std::fabs(vp), std::fabs(vm));
    }

    // Convert from three-velocity to phoebus primitive veloity
    SPACELOOP(ii) {
      v(pvel_lo + ii) = W*v(pvel_lo + ii);
    }
  }

  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus solve(const VarAccessor<T> &v, const singularity::EOS &eos,
                        bool print = false) const {
    using robust::sgn;
    using robust::ratio;
    constexpr Real SMALL = 10 * std::numeric_limits<Real>::epsilon();
    Real &D = v(scr_lo + iD);
    Real &tau = v(scr_lo + itau);
    Real &Bsq = v(scr_lo + iBsq);
    Real &Ssq = v(scr_lo + iSsq);
    Real &BdotS = v(scr_lo + iBdotS);
    Residual Rfunc(D, tau, Bsq, Ssq, BdotS, eos);
#ifndef NDEBUG
    if (print) {
      Rfunc.print();
    }
#endif
    Real &rho_guess = v(prho);
    Real &T_guess = v(tmp);

    int iter = 0;
    bool converged = false;
    Real res[2], resp[2];
    Real jac[2][2];
    constexpr Real delta_fact_min = 1.e-6;
    Real delta_fact = delta_fact_min;
    constexpr Real delta_adj = 1.2;
    constexpr Real idelta_adj = 1. / delta_adj;
    const Real delta_min = std::max(rel_tolerance*delta_fact_min, SMALL);
    Rfunc(rho_guess, T_guess, res);
    do {
      Real drho = delta_fact * rho_guess;
      if (std::abs(drho) < delta_min) { // deltas cannot be 0
        drho = sgn(drho)*delta_min;
      }
      Real idrho = ratio(1., drho);
      Rfunc(rho_guess + drho, T_guess, resp);
      jac[0][0] = (resp[0] - res[0]) * idrho;
      jac[1][0] = (resp[1] - res[1]) * idrho;
      Real dT = delta_fact * T_guess;
      if (std::abs(dT) < delta_min) { // deltas cannot be 0
        dT = sgn(dT)*delta_min;
      }
      Real idT = ratio(1., dT);
      Rfunc(rho_guess, T_guess + dT, resp);
      jac[0][1] = (resp[0] - res[0]) * idT;
      jac[1][1] = (resp[1] - res[1]) * idT;

      const Real det = (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]);
      const Real idet = 1. / det;
      if (std::abs(det) < 1.e-16) {
        delta_fact *= delta_adj;
        iter++;
        continue;
      }
      Real delta_rho = -(res[0] * jac[1][1] - jac[0][1] * res[1]) * idet;
      Real delta_T = -(jac[0][0] * res[1] - jac[1][0] * res[0]) * idet;

      if (std::abs(delta_rho) / rho_guess < rel_tolerance &&
          std::abs(delta_T) / T_guess < rel_tolerance) {
        converged = true;
      }
#ifndef NDEBUG
      if (print) {
        printf("%d %g %g %g %g %g %g\n", iter, rho_guess, T_guess, delta_rho,
               delta_T, res[0], res[1]);
      }
#endif

      Real alpha = 1.0;
      if (rho_guess + delta_rho < 0.0) {
        alpha = -0.1 * rho_guess / delta_rho;
      } else if (rho_guess + delta_rho > D) {
        alpha = (D - rho_guess) / delta_rho;
      }
      delta_rho = alpha * delta_rho;
      delta_T = (T_guess + delta_T < 0.0 ? -0.1 * T_guess : delta_T);

      const Real res0 = res[0] * res[0] + res[1] * res[1];
      Rfunc(rho_guess + delta_rho, T_guess + delta_T, res);
      Real res1 = res[0] * res[0] + res[1] * res[1];
      alpha = 1.0;
      int cnt = 0;
      while (res1 >= res0 && cnt < 5) {
        alpha *= 0.5;
        Rfunc(rho_guess + alpha * delta_rho, T_guess + alpha * delta_T, res);
        res1 = res[0] * res[0] + res[1] * res[1];
        cnt++;
      }

      rho_guess += alpha * delta_rho;
      T_guess += alpha * delta_T;
      iter++;

      delta_fact *= (delta_fact > delta_fact_min ? idelta_adj : 1.0);

    } while (converged != true && iter < max_iter);

    if (!converged) {
#ifndef NDEBUG
      printf("ConToPrim failed state: %g %g %g %g %g %g %g\n", rho_guess,
             T_guess, v(crho), v(cmom_lo), v(cmom_lo + 1), v(cmom_lo + 2),
             v(ceng));
      if (!print)
        solve(v, eos, true);
#endif
      return ConToPrimStatus::failure;
    }
    return ConToPrimStatus::success;
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION void setup(const VarAccessor<T> &v,
                                    const CellGeom &g) const {
    Real &D = v(scr_lo + iD);
    Real &tau = v(scr_lo + itau);
    Real &Bsq = v(scr_lo + iBsq);
    Real &Ssq = v(scr_lo + iSsq);
    Real &BdotS = v(scr_lo + iBdotS);
    const Real igdet = 1. / g.gdet;
    D = v(crho) * igdet;
    tau = v(ceng) * igdet;

    // electron fraction
    if (pye > 0)
      v(pye) = v(cye) / v(crho);

    BdotS = 0.0;
    Bsq = 0.0;
    // bfield
    if (pb_hi > 0) {
      // set primitive fields
      for (int i = 0; i < 3; i++) {
        v(pb_lo + i) = v(cb_lo + i) * igdet;
      }
      // take some dot products
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          Bsq += g.gcov[i][j] * v(pb_lo + i) * v(pb_lo + j);
        }
        BdotS += v(pb_lo + i) * v(cmom_lo + i);
      }
      // don't forget S_j has a \sqrt{gamma} in it...get rid of it here
      BdotS *= igdet;
    }

    Ssq = 0.0;
    Real W = 0.0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Ssq += g.gcon[i][j] * v(cmom_lo + i) * v(cmom_lo + j);
        W += g.gcov[i][j] * v(pvel_lo + i) * v(pvel_lo + j);
      }
    }
    Ssq *= igdet * igdet;
    W = sqrt(1.0 / (1.0 - W));
    v(prho) = D / W; // initial guess for density
  }
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>, VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>, MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol,
                                  const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}
/*inline C2P_Mesh_t ConToPrimSetup(MeshData<Real> *rc) {
  return C2P_Mesh_t(rc);
}*/

} // namespace con2prim

#endif // CON2PRIM_HPP_
