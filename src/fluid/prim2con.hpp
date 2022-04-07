//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef FLUID_PRIM2CON_HPP_
#define FLUID_PRIM2CON_HPP_

#include "phoebus_utils/relativity_utils.hpp"

namespace prim2con {

KOKKOS_INLINE_FUNCTION
void signal_speeds(const Real &rho, const Real &u, const Real &p, const Real &bsq,
                   const Real v[], const Real &vsq, const Real &gam1, const Real &alpha,
                   const Real beta[], const Real gcon[3][3], Real *sig) {
  const Real rho_rel = rho + u + p;
  const Real vasq = bsq / (rho_rel + bsq);
  Real cssq = gam1 * p / rho_rel;
  cssq += vasq - cssq * vasq;
  const Real vcoff = alpha / (1.0 - vsq * cssq);
  SPACELOOP(m) {
    const Real vpm =
        std::sqrt(cssq * (1.0 - vsq) *
                  (gcon[m][m] * (1.0 - vsq * cssq) - v[m] * v[m] * (1.0 - cssq)));
    const Real vp = vcoff * (v[m] * (1.0 - cssq) + vpm) - beta[m];
    const Real vm = vcoff * (v[m] * (1.0 - cssq) - vpm) - beta[m];
    sig[m] = std::max(std::fabs(vp), std::fabs(vm));
  }
}

KOKKOS_INLINE_FUNCTION
void p2c(const Real &rho, const Real vp[], const Real b[], const Real &u,
         const Real &ye_prim, const Real &p, const Real gam1, const Real gcov[4][4],
         const Real gcon[3][3], const Real beta[], const Real &alpha, const Real &gdet,
         Real &D, Real S[], Real bcons[], Real &tau, Real &ye_cons,
         Real *sig = nullptr) {
  Real vsq = 0.0;
  Real Bsq = 0.0;
  Real Bdotv = 0.0;
  const Real W = phoebus::GetLorentzFactor(vp, gcov);
  const Real iW = 1.0 / W;
  Real v[3] = {vp[0] * iW, vp[1] * iW, vp[2] * iW};
  SPACELOOP2(ii, jj) {
    vsq += gcov[ii + 1][jj + 1] * v[ii] * v[jj];
    Bsq += gcov[ii + 1][jj + 1] * b[ii] * b[jj];
    Bdotv += gcov[ii + 1][jj + 1] * b[ii] * v[jj];
  }
  const Real b0 = W * Bdotv;
  Real bcon[] = {W * Bdotv / alpha, 0.0, 0.0, 0.0};
  SPACELOOP(m) { bcon[m + 1] = b[m] * iW + Bdotv * W * (v[m] - beta[m] / alpha); }
  const Real bsq = Bsq * iW * iW + Bdotv*Bdotv;
  Real bcov[3] = {0.0, 0.0, 0.0};
  SPACELOOP(m) {
    SPACETIMELOOP(n) { bcov[m] += gcov[m + 1][n] * bcon[n]; }
  }

  D = gdet * rho * W;

  const Real rho_rel = (rho + u + p + bsq) * W * W;
  SPACELOOP(m) {
    Real vcov = 0.0;
    SPACELOOP(n) { vcov += gcov[m + 1][n + 1] * v[n]; }
    S[m] = gdet * (rho_rel * vcov - b0 * bcov[m]);
    bcons[m] = gdet * b[m];
  }

#if USE_VALENCIA
  tau = gdet * (rho_rel - (p + 0.5 * bsq) - b0*b0) - D;
#else
  Real ucon[4] = {W / alpha, vp[0] - beta[0] * W / alpha, vp[1] - beta[1] * W / alpha,
                  vp[2] - beta[2] * W / alpha};
  Real ucov[4] = {0};
  SPACETIMELOOP2(mu, nu) { ucov[mu] += gcov[mu][nu] * ucon[nu]; }
  Real bcov0 = 0.;
  SPACETIMELOOP(mu) { bcov0 += gcov[0][mu] * bcon[mu]; }
  // This isn't actually tau it's alpha*T^0_0
  tau =
      gdet * alpha *
          ((rho + u + p + bsq) * ucon[0] * ucov[0] + (p + 0.5 * bsq) - bcon[0] * bcov0) +
      D;
#endif

  ye_cons = D * ye_prim;

  if (sig != nullptr) signal_speeds(rho, u, p, bsq, v, vsq, gam1, alpha, beta, gcon, sig);
}

} // namespace prim2con

#endif
