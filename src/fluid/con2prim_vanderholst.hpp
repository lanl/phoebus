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

#ifndef CON2PRIM_VANDERHOLST_HPP_
#define CON2PRIM_VANDERHOLST_HPP_

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

namespace con2prim_vanderholst {

enum class ConToPrimStatus {success, failure};
struct FailFlags {
  static constexpr Real success = 1.0;
  static constexpr Real fail = 0.0;
};

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root(F &func, Real a, Real b, const Real tol, int maxiter, int line, bool &success) {
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

  int niter = 0;

  while (b-a > 2.0*tol && niter < maxiter) {
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
    niter++;
  }

  Real ans = 0.5*(a + b);

  if (niter == maxiter || isnan(ans)) {
    success = false;
  } else {
    success = true;
  }

  return 0.5*(a+b);
}

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root_secant(F func, const Real x_guess, const Real tol, const int maxiter, bool &success) {
  int niter = 0;

  Real x0 = x_guess;
  Real x1 = (1. + 1.e-5)*x_guess;

  while (fabs(x0 - x1)/fabs(x0) > tol) {
  //while (fabs(func(x1)) > tol) {
    Real x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0));
    x0 = x1;
    x1 = x2;
    niter++;
    if (niter == maxiter) {
      success = false;
      return x1;
    }
  }

  success = true;
  if (isnan(x1)) {
    success = false;
  }
  return x1;
}

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root_bisect(F func, Real a, Real b, const Real tol, const int maxiter, int line, bool &success) {

  Real ya = func(a);
  Real yb = func(b);
  if (ya*yb > 0.0) {
    //printf("root failure: %g %g   %g %g\n", a, b, ya, yb);
    success = false;
    return (a+b)/2.;
  }
  //PARTHENON_REQUIRE(ya * yb <= 0.0, "Root not bracketed in find_root from");
  Real sign = (ya < 0 ? 1.0 : -1.0);
  ya *= sign;
  yb *= sign;

  int niter = 0;
  while (fabs(b-a) > tol) {
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
    niter++;

    if (niter == maxiter) {
      break;
    }
  }

  if (niter == maxiter) {
    success = false;
  } else {
    success = true;
  }

  return 0.5*(a+b);
}

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real Ssq, const Real tau, const Real gamma) :
    D_(D), Ssq_(Ssq), tau_(tau), gamma_(gamma) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real xi) {
    const Real Gamma = sqrt(1./(1. - Ssq_/(xi*xi)));
    return xi - (gamma_ - 1.)/gamma_*(xi/(Gamma*Gamma) - D_/Gamma) - tau_ - D_;
  }

  private:
    const Real D_;
    const Real Ssq_;
    const Real tau_;
    const Real gamma_;
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
  //const int b_, i_, j_, k_;
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

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus Solve(const CoordinateSystem &geom, const singularity::EOS &eos,
                        Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);

    bool is_high_dens = v(prho) > 0.01 ? true : false;
    is_high_dens = true;

    // Calculate initial guess
    const Real h = 1. + v(peng)/v(prho) +  v(prs)/v(prho);
    Real w = v(prho)*h;
    Real vsq = 0.;
    Real vel[3] = {v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2)};
    SPACELOOP2(ii, jj) {
      vsq += g.gcov[ii][jj]*vel[ii]*vel[jj];
    }
    vsq = (vsq > 0.9996) ? 0.9996 : vsq;
    Real Gamma = 1./sqrt(1. - vsq);
    Real xi_guess = Gamma*Gamma*w;

    if (isnan(xi_guess)) {
      printf("Nan guess! [%i %i] xi = %e Gamma = %e w = %e vsq = %e vel = %e %e %e\n",
        v.i_, v.j_, xi_guess, gamma, w, vsq, vel[0], vel[1], vel[2]);
      //PARTHENON_FAIL("no");
    }

    //if (v.i_ == 138 && v.j_ == 186) {
    /*if (v.i_ == 120 && v.j_ == 149) {
      printf("c2p prim before: %e %e %e %e %e\n", v(prho), v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2),
        v(peng));
      printf("c2p cons before: %e %e %e %e %e\n", v(crho), v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2),
        v(ceng));
    }*/

    const Real igdet = 1.0/g.gdet;

    const Real D = v(crho)*igdet;
    const Real tau = v(ceng)*igdet;
    Real S[3] = {v(cmom_lo)*igdet, v(cmom_lo+1)*igdet, v(cmom_lo+2)*igdet};
    Real Ssq = 0.;
    SPACELOOP2(ii, jj) {
      Ssq += g.gcon[ii][jj]*S[ii]*S[jj];
    }
    const Real gamma = 5./3.;

    Residual res(D,Ssq,tau,gamma);

    bool success;
    //const Real xi = find_root_bisect(res, 0.01*xi_guess, 10.*xi_guess, rel_tolerance, max_iter, __LINE__, success);
    //Real find_root_secant(F func, const Real x_guess, const Real tol, const int maxiter, bool &success)
    const Real xi = find_root_secant(res, xi_guess, rel_tolerance, max_iter, success);

    if (!success) {
      if (is_high_dens) {
        printf("rootfind failed! [%i %i] xi_guess: %e res_guess: %e\n", v.i_, v.j_, xi_guess, res(xi_guess));
        //PARTHENON_FAIL("isudhf");
      }
      return ConToPrimStatus::failure;
    }

    Gamma = sqrt(1./(1. - Ssq/(xi*xi)));
    w = xi/(Gamma*Gamma);
    Real rho = D/Gamma;
    Real P = (gamma - 1.)/gamma*(w - rho);
    Real ug = P/(gamma - 1.);

    //if (v.i_ == 128 && v.j_ == 128) {
    //  printf("xi: %e Gamma: %e res: %e res0: %e success: %i\n", xi, Gamma, res(xi), res(xi_guess), success);
    //}

    Real vcov[3] = {S[0]/xi, S[1]/xi, S[2]/xi};
    Real vcon[3] = {0};
    SPACELOOP2(ii, jj) {
      vcon[ii] += g.gcon[ii][jj]*vcov[jj];
    }

    v(prho) = rho;
    v(peng) = ug;
    v(prs) = P;
    v(pvel_lo) = vcon[0];
    v(pvel_lo + 1) = vcon[1];
    v(pvel_lo + 2) = vcon[2];

    //v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng));
    //v(prs) = eos.PressureFromDensityTemperature(v(prho), v(tmp));
    v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng)/v(prho));
    v(prs) = eos.PressureFromDensityInternalEnergy(v(prho), v(peng)/v(prho));
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp))/v(prs);
                  //Real w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                  //eos.PressureFromDensityInternalEnergy(v(irho,k,j,i), v(ieng,k,j,i)/v(irho,k,j,i));
                  //wsum += w;
                  //sum += w*v(b,iv,k+n,j+m,i+l);

    // TODO(BRR) Set signal speeds more efficiently than by a p2c call
    Real ye_prim = 0.0;
    if (pye > 0) ye_prim = v(pye);
    Real ye_cons;
    //Real S[3];
    Real bcons[3];
    Real sig[3];
    Real bu[] = {0.0, 0.0, 0.0};
    prim2con::p2c(v(prho), vel, bu, v(peng), ye_prim, v(prs), v(gm1),
                  g.gcov4, g.gcon, g.beta, g.lapse, g.gdet,
                  v(crho), S, bcons, v(ceng), ye_cons, sig);
    for (int i = 0; i < sig_hi-sig_lo+1; i++) {
      v(sig_lo+i) = sig[i];
    }

    //if (v.i_ == 138 && v.j_ == 186) {
    /*if (v.i_ == 120 && v.j_ == 149) {
      printf("c2p prim after: %e %e %e %e %e\n", v(prho), v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2),
        v(peng));
      printf("c2p cons after: %e %e %e %e %e\n", v(crho), v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2),
        v(ceng));
    }*/

    if (isnan(rho) || isnan(ug) || isnan(P) || isnan(v(pvel_lo)) || isnan(v(pvel_lo+1)) ||
        isnan(v(pvel_lo+2)) || isnan(v(prs)) || isnan(v(gm1))) {
      if (is_high_dens) {
        printf("A nan! [%i %i] %e %e %e %e %e %e %e %e xi: %e Gamma: %e\n", v.i_, v.j_, rho, ug, P, v(pvel_lo), v(pvel_lo+1),
          v(pvel_lo+2), v(prs), v(gm1), xi, Gamma);
        //PARTHENON_FAIL("suhdf");
      }
      // These will be averaged over, just make sure they aren't NAN
      v(prho) = 0.;
      v(peng) = 0.;
      v(prs) = 0.;
      v(pvel_lo) = 0.;
      v(pvel_lo + 1) = 0.;
      v(pvel_lo + 2) = 0.;
      v(tmp) = 0.;
      v(prs) = 0.;
      v(gm1) = 0.;
      return ConToPrimStatus::failure;
    }

    return ConToPrimStatus::success;
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
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>, VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>, MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol,
                                  const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}

} // namespace con2prim_vanderholst

#endif // CON2PRIM_VANDERHOLST_HPP_
