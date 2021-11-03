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

#include <fenv.h>

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <singularity-eos/eos/eos.hpp>

#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"

namespace con2prim_robust {

using namespace robust;

enum class ConToPrimStatus {success, failure};
struct FailFlags {
  static constexpr Real success = 1.0;
  static constexpr Real fail = 0.0;
};

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root(F &func, Real a, Real b, const Real tol, int line) {
  constexpr Real kappa1 = 0.1;
  constexpr Real kappa2 = 2.0;
  constexpr int n0 = 0;

  const int nmax = std::ceil(std::log2(0.5*(b-a)/tol)) + n0;
  int j = 0;

  Real ya = func(a);
  Real yb = func(b);
  if (ya * yb > 0.0) {
    printf("Root not bracketed in find_root\n");
    return 0.5*(a+b);
  }
  Real sign = (ya < 0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;

  while (b-a > 2.0*tol) {
    const Real xh = 0.5*(a + b);
    const Real xf = (yb*a - ya*b)/(yb - ya);
    const Real xhxf = xh - xf;
    const Real delta = std::min(kappa1*std::pow(b-a,kappa2),std::abs(xhxf));
    const Real sigma = (xhxf > 0 ? 1.0 : -1.0);
    const Real xt = (delta <= sigma*xhxf ? xf + sigma*delta : xh);
    const Real r = tol*std::pow(2.0,nmax-j) - 0.5*(b - a);
    const Real xitp = (std::fabs(xt - xh) > r ? xh - sigma*r : xt);
    const Real yitp = sign*func(xitp);
    if (yitp > 0.0) {
      b = xitp;
      yb = yitp;
    } else if (yitp < 0.0) {
      a = xitp;
      ya = yitp;
    } else {
      a = xitp;
      b = xitp;
    }
    j++;
  }
  return 0.5*(a+b);
}

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root_bisect(F func, Real a, Real b, const Real tol, int line) {

  Real ya = func(a);
  Real yb = func(b);
  if (ya*yb > 0.0) {
    printf("root failure: %g %g   %g %g\n", a, b, ya, yb);
  }
  PARTHENON_REQUIRE(ya * yb <= 0.0, "Root not bracketed in find_root from\n"
    + std::to_string(a) + "\n"
    + std::to_string(b) + "\n"
    + std::to_string(ya) + "\n"
    + std::to_string(yb) + "\n");
  Real sign = (ya < 0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;

  while (b-a > tol) {
    const Real xh = 0.5*(a + b);
    const Real yh = sign*func(xh);
    if (yh > 0.0) {
      b = xh;
      yb = yh;
    } else if (yh < 0.0) {
      a = xh;
      ya = yh;
    } else {
      a = xh;
      b = xh;
    }
  }
  return 0.5*(a+b);
}

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real q, const Real bsq,
	   const Real bsq_rpsq, const Real rsq,
	   const Real rbsq, const Real v0sq, const singularity::EOS &eos,
     const fixup::Bounds &bnds, const Real x1, const Real x2, const Real x3)
	   //const Real rho_floor, const Real e_floor,
     //const Real gam_max, const Real e_max)
    : D_(D), q_(q), bsq_(bsq), bsq_rpsq_(bsq_rpsq), rsq_(rsq),
      rbsq_(rbsq), v0sq_(v0sq),
      eos_(eos), bounds_(bnds),
      x1_(x1), x2_(x2), x3_(x3)
      //rho_floor_(rho_floor), e_floor_(e_floor),
      //gam_max_(gam_max), e_max_(e_max)
      {
    Real garbage = 0.0;
    bounds_.GetFloors(x1_, x2_, x3_, rho_floor_, garbage);
    bounds_.GetCeilings(x1_, x2_, x3_, gam_max_, e_max_);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real x_mu(const Real mu) {
    return 1.0/(1.0 + mu*bsq_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real rbarsq_mu(const Real mu, const Real x) {
    return x*(x * rsq_ + mu * (1.0 + x) * rbsq_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real qbar_mu(const Real mu, const Real x) {
    const Real mux = mu*x;
    return q_ - 0.5*(bsq_ + mux*mux*bsq_rpsq_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real vhatsq_mu(const Real mu, const Real rbarsq) {
    const Real vsq_trial = mu*mu*rbarsq;
    if (vsq_trial > v0sq_) {
      used_gamma_max_ = true;
      return v0sq_;
    } else {
      used_gamma_max_ = false;
      return vsq_trial;
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real iWhat_mu(const Real vhatsq) {
    return std::sqrt(1.0 - vhatsq);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real rhohat_mu(const Real iWhat) {
    const Real rho_trial = D_*iWhat;
    if (rho_trial <= rho_floor_) {
      used_density_floor_ = true;
      //printf("set true\n");
      return rho_floor_;
    } else {
      used_density_floor_ = false;
      //printf("set false\n");
      return rho_trial;
    }
    //return std::max(D_*iWhat, rho_floor_);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real ehat_mu(const Real mu, const Real qbar, const Real rbarsq, const Real vhatsq, const Real What) {
    Real rho = rhohat_mu(1.0/What);
    bounds_.GetFloors(x1_, x2_, x3_, rho, e_floor_);
    const Real ehat_trial = What*(qbar - mu*rbarsq) + vhatsq*What*What/(1.0 + What);
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
    //feenableexcept(FE_DIVBYZERO|FE_INVALID);
    const Real x = x_mu(mu);
    const Real rbarsq = rbarsq_mu(mu,x);
    const Real qbar = qbar_mu(mu,x);
    const Real vhatsq = vhatsq_mu(mu,rbarsq);
    const Real iWhat = iWhat_mu(vhatsq);
    const Real What = 1.0/iWhat;
    Real rhohat = rhohat_mu(iWhat);
    Real ehat = ehat_mu(mu,qbar,rbarsq,vhatsq,What);
    // insert bounds for rhohat and ehat
    // or how about bounding That?
    //const Real That = eos_.TemperatureFromDensityInternalEnergy(rhohat,ehat);
    const Real Phat = eos_.PressureFromDensityInternalEnergy(rhohat,ehat);
    Real hhat = rhohat*(1.0 + ehat) + Phat;
    const Real ahat = Phat/make_positive(hhat-Phat);
    hhat /= rhohat;

    const Real nua = (1.0 + ahat) * (1.0 + ehat)*iWhat; //hhat*iWhat;
    const Real nub = (1.0 + ahat)*(1.0 + qbar - mu*rbarsq);
    const Real nuhat = std::max(nua, nub);

    const Real muhat = 1.0/(nuhat + mu*rbarsq);
    //fedisableexcept(FE_DIVBYZERO|FE_INVALID);
    //PARTHENON_REQUIRE(muhat > 0.0, "muhat < 0");
    return mu-muhat;
  }

  KOKKOS_INLINE_FUNCTION
  Real compute_upper_bound(const Real h0sq) {
    auto func = [=](const Real x) {
      return aux_func(x,h0sq);
    };
    return find_root(func,1.e-16,1.0/sqrt(h0sq),rho_floor_, __LINE__);
  }

  KOKKOS_INLINE_FUNCTION
  bool used_density_floor() const { return used_density_floor_; }
  KOKKOS_INLINE_FUNCTION
  bool used_energy_floor() const { return used_energy_floor_; }
  KOKKOS_INLINE_FUNCTION
  bool used_energy_max() const { return used_energy_max_; }
  KOKKOS_INLINE_FUNCTION
  bool used_gamma_max() const { return used_gamma_max_; }

 private:
  const Real D_, q_, bsq_, bsq_rpsq_, rsq_, rbsq_, v0sq_;
  const singularity::EOS &eos_;
  const fixup::Bounds &bounds_;
  const Real x1_, x2_, x3_;
  Real rho_floor_, e_floor_, gam_max_, e_max_;
  bool used_density_floor_, used_energy_floor_, used_energy_max_, used_gamma_max_;

  KOKKOS_FORCEINLINE_FUNCTION
  Real aux_func(const Real mu, const Real h0sq) {
    const Real x = 1.0/(1.0 + mu*bsq_);
    const Real rbarsq = x*(rsq_ * x + mu*(1.0 + x)*rbsq_);
    return mu*std::sqrt(h0sq + rbarsq) - 1.0;
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
  Real &operator()(const int n) const {
    return var_(b_, n, k_, j_, i_);
  }
  const int b_, i_, j_, k_;
 private:
  const T &var_;
};

struct CellGeom {
  template<typename CoordinateSystem>
  KOKKOS_FUNCTION
  CellGeom(const CoordinateSystem &geom,
           const int k, const int j, const int i)
    : gdet(geom.DetGamma(CellLocation::Cent,k,j,i)),
      lapse(geom.Lapse(CellLocation::Cent,k,j,i)) {
    geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
    SPACELOOP2(m,n) {
      gcov[m][n] = gcov4[m+1][n+1];
    }
    geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  }
  template<typename CoordinateSystem>
  CellGeom(const CoordinateSystem &geom,
           const int b, const int k, const int j, const int i)
    : gdet(geom.DetGamma(CellLocation::Cent,b,k,j,i)),
      lapse(geom.Lapse(CellLocation::Cent,b,k,j,i)) {
    geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, gcov4);
    SPACELOOP2(m,n) {
      gcov[m][n] = gcov4[m+1][n+1];
    }
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
    : bounds(bnds),
      var(rc->PackVariables(Vars(), imap)),
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
      pye(imap[fluid_prim::ye].second),
      cye(imap[fluid_cons::ye].second),
      prs(imap[fluid_prim::pressure].first),
      tmp(imap[fluid_prim::temperature].first),
      sig_lo(imap[internal_variables::cell_signal_speed].first),
      sig_hi(imap[internal_variables::cell_signal_speed].second),
      gm1(imap[fluid_prim::gamma1].first),
      rel_tolerance(tol),
      max_iter(max_iterations),
      h0sq_(1.0) { }

  std::vector<std::string> Vars() {
    return std::vector<std::string>({fluid_prim::density, fluid_cons::density,
                                    fluid_prim::velocity, fluid_cons::momentum,
                                    fluid_prim::energy, fluid_cons::energy,
                                    fluid_prim::bfield, fluid_cons::bfield,
                                    fluid_prim::ye, fluid_cons::ye,
                                    fluid_prim::pressure, fluid_prim::temperature,
                                    internal_variables::cell_signal_speed, fluid_prim::gamma1});
  }

  KOKKOS_INLINE_FUNCTION
  bool my_cell(const Real x1, const Real x2) const {
    //if (fabs(x1 - 3.24365) < 1.e-3 && fabs(x2 - 0.493164) < 1.e-3) return true;
    return false;
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus operator()(const CoordinateSystem &geom, const singularity::EOS &eos, const Coordinates_t &coords, Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    //Real x[4];
    //geom.Coords(CellLocation::Cent, std::forward<Args>(args)..., x);
    Real x1 = coords.x1v(std::forward<Args>(args)...);
    Real x2 = coords.x2v(std::forward<Args>(args)...);
    Real x3 = coords.x3v(std::forward<Args>(args)...);
    return solve(v,g,eos,x1,x2,x3);
  }

  int NumBlocks() {
    return var.GetDim(5);
  }

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
  const int prs, tmp, sig_lo, sig_hi, gm1;
  const Real rel_tolerance;
  const int max_iter;
  const Real h0sq_;

  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus solve(const VarAccessor<T> &v, const CellGeom &g, const singularity::EOS &eos,
                        const Real x1, const Real x2, const Real x3) const {
    if (v.i_ == 140 && v.j_ == 120) {
      printf("c2p C: %e %e %e %e %e\n", v(crho), v(ceng), v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2));
    }
    /*PARTHENON_REQUIRE(!std::isnan(v(crho)), "v(crho) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo)), "v(cmom_lo) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+1)), "v(cmom_lo+1) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+2)), "v(cmom_lo+2) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(ceng)), "v(ceng) = NaN");
    if (cb_hi > 0) {
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo)), "v(cb_lo) = NaN");
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo+1)), "v(cb_lo+1) = NaN");
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo+2)), "v(cb_lo+2) = NaN");
    }*/
    int num_nans = std::isnan(v(crho)) + std::isnan(v(cmom_lo)) + std::isnan(ceng);
    if (num_nans > 0) return ConToPrimStatus::failure;
    const Real igdet = 1.0/g.gdet;

    Real rhoflr = 0.0;
    Real epsflr;
    bounds.GetFloors(x1,x2,x3,rhoflr,epsflr);
    Real gam_max, eps_max;
    bounds.GetCeilings(x1,x2,x3,gam_max,eps_max);
    bool negative_crho = false;
    /*if (v(crho) <= 0.0) {
      printf("v(crho) < 0: %g    %g %g\n", v(crho)*igdet, x1, x2);
      v(crho) = 1.e-50;
      negative_crho = true;
    }*/
    const Real D = v(crho)*igdet;
    #if USE_VALENCIA
    const Real tau = v(ceng)*igdet;
    #else
    //if (v.i_ == 140 && v.j_ == 120) {
    //printf("c2p C: %e %e %e %e %e\n", v(crho), v(ceng), v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2));
    //}
    Real Qcov[4] = {(v(ceng) - v(crho))*igdet,
                      v(cmom_lo)*igdet,
                      v(cmom_lo+1)*igdet,
                      v(cmom_lo+2)*igdet};
    Real ncon[4] = {1./g.lapse, -g.beta[0]/g.lapse, -g.beta[1]/g.lapse, -g.beta[2]/g.lapse};
    Real tau = 0.;
    SPACETIMELOOP(mu) {
      tau -= Qcov[mu]*ncon[mu];
    }
    //if (v.i_ == 140 && v.j_ == 120) {
    //printf("pretau: %e\n", tau);
    //printf("ncon: %e %e %e %e\n", ncon[0], ncon[1], ncon[2], ncon[3]);
    //printf("Qcov: %e %e %e %e\n", Qcov[0], Qcov[1], Qcov[2], Qcov[3]);
    //}
    tau -= D;
    #endif // USE_VALENCIA
    if (v.i_ == 140 && v.j_ == 120) {
    printf("tau: %e (tau 3+1: %e)\n", tau, 4.855544e+01*igdet);
    //exit(-1);
    }
    const Real q = tau/D;
    //PARTHENON_REQUIRE(D > 0, "D < 0");

    if (pye > 0) v(pye) = v(cye)/v(crho);

    Real bsq = 0.0;
    Real bsq_rpsq = 0.0;
    Real rbsq = 0.0;
    Real rsq = 0.0;
    Real bdotr = 0.0;
    Real rcon[3];
    // r_i
    Real rcov[] = {v(cmom_lo)/v(crho), v(cmom_lo+1)/v(crho), v(cmom_lo+2)/v(crho)};
    SPACELOOP(i) {
      rcon[i] = 0.0;
      SPACELOOP(j) {
        rcon[i] += g.gcon[i][j] * rcov[j];
      }
      rsq += rcon[i] * rcov[i];
    }
    /*Real Dtau = D + tau;
    Dtau *= Dtau;
    if (rsq * D*D > Dtau) {
      const Real scale = std::sqrt(Dtau/(rsq*D*D));
      SPACELOOP(i) {
        rcon[i] *= scale;
        rcov[i] *= scale;
        v(cmom_lo+i) *= scale;

      }
      rsq *= scale*scale;
      printf("scaled back momuntum %g\n", scale);
    }*/
    /*
    if (rsq > 1.e10) {
      const Real rscale = sqrt(1.e10/rsq);
      SPACELOOP(i) {
        rcon[i] *= rscale;
        rcov[i] *= rscale;
      }
      rsq = 1.e10;
    }
    */
    PARTHENON_REQUIRE(rsq >= 0.0, "rsq < 0");
    Real bu[] = {0.0, 0.0, 0.0};
    if (pb_hi > 0) {
      // first set the primitive b-fields
      SPACELOOP(i) {
        v(pb_lo+i) = v(cb_lo+i)*igdet;
      }
      const Real sD = 1.0/std::sqrt(std::max(D,rhoflr));
      // b^i
      SPACELOOP(i) {
        bu[i] = v(pb_lo+i)*sD;
        bdotr += bu[i]*rcov[i];
      }
      SPACELOOP2(i,j) bsq += g.gcov[i][j] * bu[i] * bu[j];
      bsq = std::max(0.0, bsq);

      rbsq = bdotr*bdotr;
      bsq_rpsq = bsq * rsq - rbsq;
    }
    const Real zsq = rsq/h0sq_;
    Real v0sq = std::min(zsq/(1.0 + zsq), 1.0 - 1.0/(gam_max*gam_max));
    if (v0sq >= 1) {
      printf("whoa: %g %g %g %g\n", rsq, h0sq_, zsq, v0sq);
    }
    //v0sq = make_bounded(v0sq, 0.0, 1.0);
    if (!(bsq >= 0)) {
      printf("bsq < 0: %e\n", bsq);
      PARTHENON_FAIL("bsq < 0");
    }
    if (!(rbsq >= 0)) {
      printf("rbsq < 0: %e\n", rbsq);
      PARTHENON_FAIL("rbsq < 0");
    }
    /*PARTHENON_REQUIRE(bsq_rpsq >= 0, "bsq_rpsq < 0: " + std::to_string(bsq_rpsq)
      + " " + std::to_string(bsq)
      + " " + std::to_string(rsq)
      + " " + std::to_string(rbsq));*/
    //PARTHENON_REQUIRE(v0sq < 1, "v0sq >= 1: " + std::to_string(v0sq));
    //PARTHENON_REQUIRE(v0sq >= 0, "v0sq < 0: " + std::to_string(v0sq));

    if (v.i_ == 140 && v.j_ == 120) {
      printf("D: %e q: %e bsq: %e bsq_rpsq: %e rsq: %e rbsq: %e v0sq: %e\n",
        D, q, bsq, bsq_rpsq, rsq, rbsq, v0sq);
    }
    Residual res(D,q,bsq,bsq_rpsq,rsq,rbsq,v0sq,eos,bounds,x1,x2,x3);//rhoflr,epsflr,gam_max,eps_max);

    // find the upper bound
    //const Real mu_r = res.compute_upper_bound(h0sq_);
    // solve
    const Real mu = find_root(res, 0.0, 1.0, rel_tolerance, __LINE__);
    if (v.i_ == 140 && v.j_ == 120) {
      printf("mu: %e\n", mu);
    }
    //if(atm) printf("used atm\n");
    if(my_cell(x1,x2)) {
      printf("res = %e\n", res(mu));
    }

    // now unwrap everything into primitive and conserved vars
    const Real x = res.x_mu(mu);
    const Real rbarsq = res.rbarsq_mu(mu,x);
    Real vsq = res.vhatsq_mu(mu,rbarsq);
    Real iW = res.iWhat_mu(vsq);
    v(prho) = res.rhohat_mu(iW);
    const Real qbar = res.qbar_mu(mu,x);
    Real W = 1.0/iW;
    v(peng) = res.ehat_mu(mu, qbar, rbarsq, vsq, W);
    v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng));
    v(peng) *= v(prho);
    v(prs) = eos.PressureFromDensityTemperature(v(prho), v(tmp));
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp))/v(prs);

    /*if (v(prho) < 0.99999*rhoflr ||
        v(peng)/v(prho) < 0.99999*epsflr ||
        v(peng)/v(prho) > 1.00001*eps_max ||
        W > 1.00001*gam_max) {
      printf("bounds violated %e %e %e %e\n", v(prho)/rhoflr, v(peng)/v(prho)/epsflr,
                                              v(peng)/v(prho)/eps_max, W/gam_max);
    }*/

    //const Real vscale = (W > 10 ? 10.0/W : 1.0);
    Real vel[3];
    SPACELOOP(i) {
      //v(pvel_lo+i) = atm ? 0 : mu*x*(rcon[i] + mu*bdotr*bu[i]);
      vel[i] = mu*x*(rcon[i] + mu*bdotr*bu[i]);
      v(pvel_lo+i) = W*vel[i];
    }
    if (v.i_ == 140 && v.j_ == 120)
    {
      printf("c2p: W: %e vp: %e %e %e v: %e %e %e rho: %e u: %e\n", W, v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2), vel[0],vel[1],vel[2], v(prho), v(peng));
    }
    if (pb_hi > 0) {
      SPACELOOP(i) {
        bu[i] = v(pb_lo+i);
      }
    }

    /*if (res.used_density_floor() ||
        res.used_energy_floor() ||
        res.used_gamma_max()) {
      v(prho) = rhoflr;
      SPACELOOP(i) {
        v(pvel_lo+i) = 0.0;
        vel[i] = v(pvel_lo+i);
      }
      v(peng) = epsflr;
      v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng));
      v(peng) *= v(prho);
      v(prs) = eos.PressureFromDensityTemperature(v(prho), v(tmp));
      v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp))/v(prs);
    }*/

    Real ye_prim = 0.0;
    if (pye > 0) ye_prim = v(pye);
    Real ye_cons;
    Real S[3];
    Real bcons[3];
    Real sig[3];
    if (v.i_ == 140 && v.j_ == 120) {
    printf("rho: %e vel: %e %e %e bu: %e %e %e peng: %e ye: %e prs: %e gm1: %e\n",
      v(prho), vel[0], vel[1], vel[2], bu[0], bu[1], bu[2], v(peng), ye_prim, v(prs), v(gm1));
    }
    prim2con::p2c(v(prho), vel, bu, v(peng), ye_prim, v(prs), v(gm1),
                  g.gcov4, g.gcon, g.beta, g.lapse, g.gdet,
                  v(crho), S, bcons, v(ceng), ye_cons, sig);

    SPACELOOP(i) {
      v(cmom_lo+i) = S[i];
    }

    if (pye > 0) v(cye) = ye_cons;

    for (int i = 0; i < sig_hi-sig_lo+1; i++) {
      v(sig_lo+i) = sig[i];
    }

    if (v.i_ == 140 && v.j_ == 120) {
    printf("c2p aftermath: C: %e %e %e %e %e\n",
      v(crho), v(ceng), v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2));
    }

    num_nans = std::isnan(v(crho)) + std::isnan(v(cmom_lo)) + std::isnan(v(ceng));
    //PARTHENON_REQUIRE(!std::isnan(v(crho)), "v(crho) = NaN");
    //PARTHENON_REQUIRE(!std::isnan(v(cmom_lo)), "v(cmom_lo) = NaN");
    //PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+1)), "v(cmom_lo+1) = NaN");
    //PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+2)), "v(cmom_lo+2) = NaN");
    //PARTHENON_REQUIRE(!std::isnan(v(ceng)), "v(ceng) = NaN");

    if (//res.used_density_floor() ||
        //res.used_energy_max() ||
        //res.used_energy_floor() ||
        //res.used_gamma_max() ||
        num_nans > 0)
      return ConToPrimStatus::failure;
    return ConToPrimStatus::success;
  }
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>,VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>,MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, fixup::Bounds bounds, const Real tol, const int max_iter) {
  return C2P_Block_t(rc, bounds, tol, max_iter);
}
/*inline C2P_Mesh_t ConToPrimSetup(MeshData<Real> *rc) {
  return C2P_Mesh_t(rc);
}*/

} // namespace con2prim

#endif // CON2PRIM_ROBUST_HPP_
