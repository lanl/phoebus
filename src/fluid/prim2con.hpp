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

namespace prim2con {

KOKKOS_INLINE_FUNCTION
void p2c(const Real &rho, const Real v[], const Real b[], const Real &u,
         const Real &ye_prim, const Real &p, const Real gam1,
         const Real gcov[4][4], const Real gcon[3][3], const Real beta[],
         const Real &alpha, const Real &gdet,
         Real &D, Real S[], Real bcons[], Real &tau, Real &ye_cons, Real sig[]) {
  Real vsq = 0.0;
  Real Bsq = 0.0;
  Real Bdotv = 0.0;
  SPACELOOP(m) {
    SPACELOOP(n) {
      vsq += gcov[m+1][n+1] * v[m] * v[n];
      Bsq += gcov[m+1][n+1] * b[m] * b[n];
      Bdotv += gcov[m+1][n+1] * b[m] * v[n];
    }
  }
  const Real iW = sqrt(1.0 - vsq);
  const Real W = 1.0/iW;
  Real bcon[] = {W * Bdotv / alpha, 0.0, 0.0, 0.0};
  SPACELOOP(m) {
    bcon[m+1] = b[m]*iW + bcon[0] * (v[m] - beta[m]/alpha);
  }
  const Real bsq = (Bsq + alpha*alpha * bcon[0]*bcon[0])*iW*iW;
  Real bcov[3] = {0.0, 0.0, 0.0};
  SPACELOOP(m) {
    SPACETIMELOOP(n) {
      bcov[m] += gcov[m+1][n] * bcon[n];
    }
  }

  D = gdet * rho*W;
  if (isnan(D)) {
    printf("gdet: %e rho: %e W: %e vsq: %e\n", gdet, rho, W, vsq);
  }

  const Real rho_rel = (rho + u + p + bsq)*W*W;
  SPACELOOP(m) {
    Real vcov = 0.0;
    SPACELOOP(n) {
      vcov += gcov[m+1][n+1] * v[n];
    }
    S[m] = gdet * (rho_rel*vcov - alpha*bcon[0]*bcov[m]);
    bcons[m] = gdet * b[m];
  }

  tau = gdet * (rho_rel - (p + 0.5*bsq) - alpha*alpha*bcon[0]*bcon[0]) - D;
  ye_cons = D * ye_prim;

  const Real vasq = bsq*W*W/rho_rel;
  Real cssq = gam1*p/(rho + u + p);
  cssq += vasq - cssq*vasq;
  const Real vcoff = alpha/(1.0 - vsq*cssq);
  SPACELOOP(m) {
    const Real vpm = std::sqrt(cssq*(1.0 - vsq)*(gcon[m][m]*(1.0 - vsq*cssq)) - v[m]*v[m]*(1.0 - cssq));
    const Real vp = vcoff*(v[m]*(1.0 - cssq) + vpm) - beta[m];
    const Real vm = vcoff*(v[m]*(1.0 - cssq) - vpm) - beta[m];
    sig[m] = std::max(std::fabs(vp), std::fabs(vm));
  }
}

}

#endif
