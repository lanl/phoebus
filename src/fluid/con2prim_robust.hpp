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
#include <eos/eos.hpp>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

namespace con2prim_robust {

using namespace robust;

enum class ConToPrimStatus {success, failure};
struct FailFlags {
  static constexpr Real success = 0.0;
  static constexpr Real fail = 1.0;
};

KOKKOS_INLINE_FUNCTION
void GetFloors(const Real &r, const Real &theta, Real &rho, Real &eps) {
#if PHOEBUS_GEOMETRY == FMKS
  rho = 1e-5*pow(r,-2);
  eps = 1e-1*pow(r,-1);
#else
  rho = 1e-8;
  eps = 1e-3;
#endif // PHOEBUS_GEOMETRY
}


template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root(F &func, Real a, Real b, const Real tol, int line) {
  constexpr Real kappa1 = 0.1;
  constexpr Real kappa2 = 2.0;
  constexpr int n0 = 0;

  const int nmax = std::ceil(std::log2((b-a)/tol)) + n0;
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

  while (b-a > tol) {
    const Real xh = 0.5*(a + b);
    const Real xf = (yb*a - ya*b)/(yb - ya);
    const Real xhxf = xh - xf;
    const Real delta = std::min(kappa1*std::pow(b-a,kappa2),std::abs(xhxf));
    const Real sigma = (xhxf > 0 ? 1.0 : -1.0);
    const Real xt = (delta <= sigma*xhxf ? xf + sigma*delta : xh);
    const Real r = std::min(0.5*tol*(1<<(nmax-j)) - 0.5*(b - a), std::abs(xt-xh));
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
	   const Real rho_floor = 1e-11, const Real e_floor = 1e-9)
    : D_(D), q_(q), bsq_(bsq), bsq_rpsq_(bsq_rpsq), rsq_(rsq),
      rbsq_(rbsq), v0sq_(v0sq),
      eos_(eos), rho_floor_(rho_floor), e_floor_(e_floor) {}

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
    return std::min(mu*mu*rbarsq, v0sq_);
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
    return std::max(What*(qbar - mu*rbarsq) + vhatsq*What*What/(1.0 + What), e_floor_);
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
  Real get_density_floor() const {return rho_floor_;}

 private:
  const Real D_, q_, bsq_, bsq_rpsq_, rsq_, rbsq_, v0sq_;
  const singularity::EOS &eos_;
  const Real rho_floor_, e_floor_;
  bool used_density_floor_;

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
 private:
  const T &var_;
  const int b_, i_, j_, k_;
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
  ConToPrim(Data_t *rc, const Real tol, const int max_iterations)
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
      pye(imap[fluid_prim::ye].second),
      cye(imap[fluid_cons::ye].second),
      prs(imap[fluid_prim::pressure].first),
      tmp(imap[fluid_prim::temperature].first),
      sig_lo(imap[internal_variables::cell_signal_speed].first),
      sig_hi(imap[internal_variables::cell_signal_speed].second),
      gm1(imap[fluid_prim::gamma1].first),
      rel_tolerance(tol),
      max_iter(max_iterations),
      h0sq_(1.0) {
    auto *pm = rc->GetParentPointer().get();
    StateDescriptor *pkg = pm->packages.Get("geometry").get();
    transform = Geometry::GetTransformation<Geometry::McKinneyGammieRyan>(pkg);
  }

  std::vector<std::string> Vars() {
    return std::vector<std::string>({fluid_prim::density, fluid_cons::density,
                                    fluid_prim::velocity, fluid_cons::momentum,
                                    fluid_prim::energy, fluid_cons::energy,
                                    fluid_prim::bfield, fluid_cons::bfield,
                                    fluid_prim::ye, fluid_cons::ye,
                                    fluid_prim::pressure, fluid_prim::temperature,
                                    internal_variables::cell_signal_speed, fluid_prim::gamma1});
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
    return solve(v,g,eos,x1,x2);
  }

  int NumBlocks() {
    return var.GetDim(5);
  }

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
  Geometry::McKinneyGammieRyan transform;
  

  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus solve(const VarAccessor<T> &v, const CellGeom &g, const singularity::EOS &eos, Real x1, Real x2) const {
    PARTHENON_REQUIRE(!std::isnan(v(crho)), "v(crho) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo)), "v(cmom_lo) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+1)), "v(cmom_lo+1) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+2)), "v(cmom_lo+2) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(ceng)), "v(ceng) = NaN");
    if (cb_hi > 0) {
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo)), "v(cb_lo) = NaN");
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo+1)), "v(cb_lo+1) = NaN");
      PARTHENON_REQUIRE(!std::isnan(v(cb_lo+2)), "v(cb_lo+2) = NaN");
    }
    const Real igdet = 1.0/g.gdet;
    
    if (v(crho) <= 0.0) {
      printf("v(crho) < 0: %g    %g %g\n", v(crho)*igdet, x1, x2);
    }
    const Real D = v(crho)*igdet;
    const Real tau = v(ceng)*igdet;
    const Real q = tau/D;
    PARTHENON_REQUIRE(D > 0, "D < 0");

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
      const Real sD = 1.0/std::sqrt(D);
      // b^i
      SPACELOOP(i) {
        bu[i] = v(pb_lo+i)*sD;
        bdotr += bu[i]*rcov[i];
      }
      SPACELOOP2(i,j) bsq += g.gcov[i][j] * bu[i] * bu[j];

      rbsq = bdotr*bdotr;
      bsq_rpsq = bsq * rsq - rbsq + 1.0e-26;
    }
    const Real zsq = rsq/h0sq_;
    Real v0sq = zsq/(1.0 + zsq);//, 1.0 - 1.0/1.0e4);
    if (v0sq >= 1) {
      printf("whoa: %g %g %g %g\n", rsq, h0sq_, zsq, v0sq);
    }
    //v0sq = make_bounded(v0sq, 0.0, 1.0);
    PARTHENON_REQUIRE(bsq >= 0, "bsq < 0: " + std::to_string(bsq));
    PARTHENON_REQUIRE(rbsq >= 0, "rbsq < 0: " + std::to_string(rbsq));
    PARTHENON_REQUIRE(bsq_rpsq >= 0, "bsq_rpsq < 0: " + std::to_string(bsq_rpsq)
      + " " + std::to_string(bsq)
      + " " + std::to_string(rsq)
      + " " + std::to_string(rbsq));
    PARTHENON_REQUIRE(v0sq < 1, "v0sq >= 1: " + std::to_string(v0sq));
    PARTHENON_REQUIRE(v0sq >= 0, "v0sq < 0: " + std::to_string(v0sq));

    // set up the residual functor
    Real r_bl = transform.bl_radius(x1);
    Real th_bl = transform.bl_theta(x1,x2);
    Real rhoflr, epsflr;
    GetFloors(r_bl,th_bl,rhoflr,epsflr);
    Residual res(D,q,bsq,bsq_rpsq,rsq,rbsq,v0sq,eos,rhoflr,epsflr);

    // find the upper bound
    //const Real mu_r = res.compute_upper_bound(h0sq_);
    // solve
    const Real mu = find_root(res, 0.0, 1.0, rel_tolerance, __LINE__);
    //if(atm) printf("used atm\n");

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

    // bool atm = v(prho) <= rhoflr;
    /*if (!atm) {
      printf("disk: %g %g %g\n", v(prho), v(peng)/v(prho), v(prs));
    }*/

    SPACELOOP(i) {
      //v(pvel_lo+i) = atm ? 0 : mu*x*(rcon[i] + mu*bdotr*bu[i]);
      v(pvel_lo+i) = mu*x*(rcon[i] + mu*bdotr*bu[i]);
    }
    //SPACELOOP(i) v(pvel_lo+i) = mu*(x*rperp[i] - rpar[i]);
    // and now make sure v is appropriately bounded
    //if (atm) {
    //if (std::isnan(W) || W > 100.0) printf("W large: %g %g\n", vsq, W);
    //}
    /*
    if (vsq > 1.0 - iW*iW) {
      const Real scale = std::sqrt( (1.0 - iW*iW)/vsq );
      //printf("scaling velocity %g %g %g\n", scale, vsq, iW);
      SPACELOOP(i) v(pvel_lo+i) *= scale;
    }*/
    // primitives are recovered

    // now recompute conserved vars to ensure consistency

    // first density
    v(crho) = v(prho) * W * g.gdet;
    if (std::isnan(v(crho)) || v(crho) < 0.0) {
      printf("caught NaN rho: %g %g %g %g %g\n", v(crho), v(prho), W, 1.0-vsq, g.gdet);
    }
  
    Real Bdotv = 0.0;
    Real bcon[] = {0.0, 0.0, 0.0, 0.0};
    Real bcov[] = {0.0, 0.0, 0.0};
    bsq = 0.0;
    if (pb_hi > 0) {
      Real Bsq = 0.0;
      SPACELOOP2(i,j) {
        Bsq += g.gcov[i][j] * v(pb_lo+i) * v(pb_lo+j);
        Bdotv += g.gcov[i][j] * v(pb_lo+i) * v(pvel_lo+j);
      }
      bcon[0] = W * Bdotv / g.lapse;
      bsq = (Bsq + g.lapse*g.lapse*bcon[0]*bcon[0])*iW*iW;
      SPACELOOP(i) bcon[i+1] = v(pb_lo+i)/W + g.lapse*bcon[0] * (v(pvel_lo+i) - g.beta[i]/g.lapse);
      SPACELOOP(i) {
        bcov[i] = 0.0;
        SPACETIMELOOP(j) {
          bcov[i] += g.gcov4[i+1][j] * bcon[j];
        }
      }
    }

    // set momentum
    const Real rhohWsq = (v(prho) + v(peng) + v(prs) + bsq) * W * W;
    SPACELOOP(i) {
      Real vcovi = 0.0;
      SPACELOOP(j) {
        vcovi += g.gcov[i][j] * v(pvel_lo+j);
      }
      v(cmom_lo + i) = g.gdet * rhohWsq * vcovi - g.lapse*bcon[0] * bcov[i];
    }

    // now energy
    v(ceng) = g.gdet*(rhohWsq - (v(prs) + 0.5*bsq) - g.lapse*g.lapse*bcon[0]*bcon[0]) - v(crho);
    
    if (cye > 0) v(cye) = v(pye)*v(crho);

    // and finally get signal speeds
    Real vasq = bsq*W*W/rhohWsq;
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp))/v(prs);
    Real cssq = v(gm1)*v(prs)/(v(prho) + v(peng) + v(prs));
    cssq += vasq - cssq*vasq; // estimate of fast magneto-sonic speed
    const Real vcoff = g.lapse/(1.0 - vsq*cssq);

    // get a null vector
    /*Real k[] = {0.0, 0.0, 0.0, 0.0};
    k[1] = 1.0;
    Real AA = gcov4[0][0];
    Real BB = 2.0*gcov[0][1]*k[1];
    Real CC = gcov[1][1]*k[1]*k[1];
    k[0] = (-BB - std::sqrt(BB*BB - 4.0*AA*CC))/(2.0*AA);
    Real null_speed0 = k[1]/k[0];*/

    for (int i = 0; i < sig_hi-sig_lo+1; i++) {
      const Real vpm = std::sqrt(cssq*(1.0 - vsq)
                        *(g.gcon[i][i]*(1.0 - vsq*cssq) 
                          - v(pvel_lo+i)*v(pvel_lo+i)*(1.0 - cssq)));
      Real vp = vcoff*(v(pvel_lo+i)*(1.0 - cssq) + vpm) - g.beta[i];
      Real vm = vcoff*(v(pvel_lo+i)*(1.0 - cssq) - vpm) - g.beta[i];
      v(sig_lo+i) = std::max(std::fabs(vp), std::fabs(vm));
    }

    /*if (atm) {
      printf("\natm state:\n");
      printf("rho: %g %g\n", v(crho), v(prho));
      printf("v1: %g %g\n", v(cmom_lo), v(pvel_lo));
      printf("v2: %g %g\n", v(cmom_lo+1), v(pvel_lo+1));
      printf("v3: %g %g\n", v(cmom_lo+2), v(pvel_lo+2));
      printf("eng: %g %g\n\n", v(ceng), v(peng));
    }*/

    PARTHENON_REQUIRE(!std::isnan(v(crho)), "v(crho) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo)), "v(cmom_lo) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+1)), "v(cmom_lo+1) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(cmom_lo+2)), "v(cmom_lo+2) = NaN");
    PARTHENON_REQUIRE(!std::isnan(v(ceng)), "v(ceng) = NaN");


    return ConToPrimStatus::success;
  }
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>,VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>,MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol, const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}
/*inline C2P_Mesh_t ConToPrimSetup(MeshData<Real> *rc) {
  return C2P_Mesh_t(rc);
}*/

} // namespace con2prim

#endif // CON2PRIM_ROBUST_HPP_
