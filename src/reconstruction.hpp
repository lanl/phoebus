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

#ifndef RECONSTRUCTION_HPP_
#define RECONSTRUCTION_HPP_

#include <algorithm>
#include <basic_types.hpp>
#include <kokkos_abstraction.hpp>

namespace PhoebusReconstruction {

enum class ReconType {linear, weno5z, weno5a, mp5};


KOKKOS_FORCEINLINE_FUNCTION
Real koren(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 0.0);
  return std::max(0.0, std::min(2.0*r, std::min((1.0 + 2.0*r)/3.0, 2.0)));
}

KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 0.0);
  return std::max(0.0, std::min(2.0,std::min(2.0*r,0.5*(1+r))));
}
KOKKOS_FORCEINLINE_FUNCTION
Real minmod(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 0.0);
  const Real t = 1.5;
  return std::max(0.0, std::min(t, std::min(t*r,0.5*(1.0+r))));
}

KOKKOS_FORCEINLINE_FUNCTION
Real ltanh(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.0)
    return 2.0*std::tanh(dm/dp);
  return 2.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real barth_jespersen(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 3.0);
  return std::min(std::min(1.0, 4.0/(1.0+r)), 4.0*r/(1.0+r));
}

KOKKOS_FORCEINLINE_FUNCTION
Real overbee_alpha(const Real dm, const Real dp, const Real alpha) {
  if (alpha < 1.0 || alpha > 2.0) {
    PARTHENON_FAIL("alpha not between 1 and 2");
  }
  Real dc = std::abs(0.5*(dp - dm));
  Real dcent = std::min(dc,2.0*std::min(std::abs(dm),std::abs(dp)));
  Real dmin = alpha*std::min(std::abs(dm), std::abs(dp));
  dmin = std::max(dcent,dmin);
  if (std::abs(dp) > 0.)
    return dmin/std::abs(dp);
  return alpha;
}

KOKKOS_FORCEINLINE_FUNCTION
Real overbee(const Real dm, const Real dp) {
  if (std::abs(dp) > 0.)
    return 2.0*std::min(std::abs(dm), std::abs(dp))/std::abs(dp);
  return 2.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real superbee(const Real dm, const Real dp) {
  const Real r = (std::abs(dp) > 0. ? dm/dp : 2.0);
  // superbee
  return std::max(std::max(0.0,std::min(2*r,1.0)), std::min(r,2.0));
}

KOKKOS_FORCEINLINE_FUNCTION
Real nolim(const Real dm, const Real dp) {
  if (fabs(dp) > 0.0) {
    return 0.5*(dp - dm)/dp;
  }
  return 1.0;
}

KOKKOS_FORCEINLINE_FUNCTION
Real phifunc(const Real mind, const Real maxd, const Real gx, const Real gy, const Real dx, const Real dy) {
  Real delta = gx*dx + gy*dy;
  if(delta > 0.) {
    return std::min(1.0,maxd/delta);
  } else if (delta < 0.) {
    return std::min(1.0,mind/delta);
  }
  return 1.0;
}

template <typename T, typename Tq>
KOKKOS_INLINE_FUNCTION
void PiecewiseConstant(const int d, const int n,
                       const int k, const int j, const int i,
                       const T &v, const Tq &ql, const Tq &qr) {
  const int dir = d-1;
  const int di = (d == X1DIR ? 1 : 0);
  const int dj = (d == X2DIR ? 1 : 0);
  const int dk = (d == X3DIR ? 1 : 0);
  ql(dir,n,k+dk,j+dj,i+di) = v(n,k,j,i);
  qr(dir,n,k,j,i) = v(n,k,j,i);
}

template <typename T, typename Tq>
KOKKOS_INLINE_FUNCTION
void PiecewiseLinear(const int d, const int n,
                     const int k, const int j, const int i,
                     const T &v, const Tq &ql, const Tq &qr) {
  const int dir = d-1;
  const int di = (d == X1DIR ? 1 : 0);
  const int dj = (d == X2DIR ? 1 : 0);
  const int dk = (d == X3DIR ? 1 : 0);
  Real dq = v(n,k+dk,j+dj,i+di) - v(n,k,j,i);
  dq = minmod(v(n,k,j,i)-v(n,k-dk,j-dj,i-di), dq)*dq;
  ql(dir,n,k+dk,j+dj,i+di) = v(n,k,j,i) + 0.5*dq;
  qr(dir,n,k,j,i) = v(n,k,j,i) - 0.5*dq;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void WENO5Z(const int d, const int n, const int k, const int j, const int i,
            const T &v, const ParArrayND<Real> &q_l, const ParArrayND<Real> &q_r) {
  constexpr Real w5alpha[3][3] = {{1.0/3.0, -7.0/6.0, 11.0/6.0},
                                  {-1.0/6.0, 5.0/6.0, 1.0/3.0},
                                  {1.0/3.0, 5.0/6.0, -1.0/6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  constexpr Real eps = 1e-100;
  constexpr Real thirteen_thirds = 13.0/3.0;

  const int di = d == X1DIR;
  const int dj = d == X2DIR;
  const int dk = d == X3DIR;

  const Real q0 = v(n,k-2*dk,j-2*dj,i-2*di);
  const Real q1 = v(n,k-dk,j-dj,i-di);
  const Real q2 = v(n,k,j,i);
  const Real q3 = v(n,k+dk,j+dj,i+di);
  const Real q4 = v(n,k+2*dk,j+2*dj,i+2*di);

  Real a = q0 - 2*q1 + q2;
  Real b = q0 - 4.0*q1 + 3.0*q2;
  Real beta0 = thirteen_thirds * a*a + b*b  + eps;
  a = q1 - 2.0*q2 + q3;
  b = q3 - q1;
  Real beta1 = thirteen_thirds * a*a + b*b + eps;
  a = q2 - 2.0*q3 + q4;
  b = q4 - 4.0*q3 + 3.0*q2;
  Real beta2 = thirteen_thirds * a*a + b*b + eps;
  const Real tau5 = std::fabs(beta2 - beta0);

  beta0 = (beta0 + tau5)/beta0;
  beta1 = (beta1 + tau5)/beta1;
  beta2 = (beta2 + tau5)/beta2;
  Real w0 = w5gamma[0]*beta0;
  Real w1 = w5gamma[1]*beta1;
  Real w2 = w5gamma[2]*beta2;
  Real wsum = 1.0/(w0 + w1 + w2);
  Real ql = wsum *
           (w0 * (w5alpha[0][0]*q0 + w5alpha[0][1]*q1 + w5alpha[0][2]*q2)
          + w1 * (w5alpha[1][0]*q1 + w5alpha[1][1]*q2 + w5alpha[1][2]*q3)
          + w2 * (w5alpha[2][0]*q2 + w5alpha[2][1]*q3 + w5alpha[2][2]*q4));

  const Real alpha_l = 3.0*wsum/(w5gamma[0]/(w0+eps) + w5gamma[1]/(w1+eps) + w5gamma[2]/(w2+eps));

  w0 = w5gamma[0]*beta2;
  w1 = w5gamma[1]*beta1;
  w2 = w5gamma[2]*beta0;
  wsum = 1.0/(w0 + w1 + w2);
  Real qr = wsum *
           (w0 * (w5alpha[0][0]*q4 + w5alpha[0][1]*q3 + w5alpha[0][2]*q2)
          + w1 * (w5alpha[1][0]*q3 + w5alpha[1][1]*q2 + w5alpha[1][2]*q1)
          + w2 * (w5alpha[2][0]*q2 + w5alpha[2][1]*q1 + w5alpha[2][2]*q0));
  const Real alpha_r = 3.0*wsum/(w5gamma[0]/(w0+eps) + w5gamma[1]/(w1+eps) + w5gamma[2]/(w2+eps));

  Real dq = q3-q2;
  dq = mc(q2-q1, dq)*dq;

  const Real alpha_lin = 2.0/(1.0/(alpha_l+eps) + 1.0/(alpha_r+eps));
  ql = alpha_lin*ql + (1.0 - alpha_lin) * (q2 + 0.5*dq);
  qr = alpha_lin*qr + (1.0 - alpha_lin) * (q2 - 0.5*dq);

  q_l(d-1,n,k+dk,j+dj,i+di) = ql;
  q_r(d-1,n,k,j,i) = qr;
}

template <typename T, typename Tq>
KOKKOS_INLINE_FUNCTION
void WENO5A(const int d, const int n, const int k, const int j, const int i,
           const T &v, const Tq &ql, const Tq &qr) {

  constexpr Real w5alpha[3][3] = {{1.0/3.0, -7.0/6.0, 11.0/6.0},
                                  {-1.0/6.0, 5.0/6.0, 1.0/3.0},
                                  {1.0/3.0, 5.0/6.0, -1.0/6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  // for WENO-Z type weights. The higher the number, the more diffusive
  // and the more aggressively the method approaches ENO in discontinuous regions
  constexpr Real p = 13;

  const int dir = d-1;
  const int di = (d == X1DIR ? 1 : 0);
  const int dj = (d == X2DIR ? 1 : 0);
  const int dk = (d == X3DIR ? 1 : 0);
  Real q[5];
  Real tl[3];
  Real tr[3];
  Real beta[3];
  Real wl[3];
  Real wr[3];
  for (int l = 0; l < 5; l++) {
    q[l] = v(n,k+(l-2)*dk,j+(l-2)*dj,i+(l-2)*di);
  }

  // ENO polynomials
  tl[0] = w5alpha[0][0]*q[0] + w5alpha[0][1]*q[1] + w5alpha[0][2]*q[2];
  tl[1] = w5alpha[1][0]*q[1] + w5alpha[1][1]*q[2] + w5alpha[1][2]*q[3];
  tl[2] = w5alpha[2][0]*q[2] + w5alpha[2][1]*q[3] + w5alpha[2][2]*q[4];

  tr[0] = w5alpha[0][0]*q[4] + w5alpha[0][1]*q[3] + w5alpha[0][2]*q[2];
  tr[1] = w5alpha[1][0]*q[3] + w5alpha[1][1]*q[2] + w5alpha[1][2]*q[1];
  tr[2] = w5alpha[2][0]*q[2] + w5alpha[2][1]*q[1] + w5alpha[2][2]*q[0];

  // Smoothness indicators from Tchekhovskoy et al. 2007
  beta[0] = (13. / 12.) * std::pow(q[0] - 2. * q[1] + q[2], 2) +
            0.25 * std::pow(q[0] - 4. * q[1] + 3. * q[2], 2);
  beta[1] = (13. / 12.) * std::pow(q[1] - 2. * q[2] + q[3], 2) +
            0.25 * std::pow(q[3] - q[1], 2);
  beta[2] = (13. / 12.) * std::pow(q[2] - 2. * q[3] + q[4], 2) +
            0.25 * std::pow(q[4] - 4. * q[3] + 3. * q[2], 2);

  // Nonlinearity/shock detectors for WENO-A
  Real phi = std::sqrt(std::abs(beta[0] - 2*beta[1] + beta[2]));
  Real PHI = std::min(1.0,phi);
  Real tau5 = std::fabs(beta[2] - beta[0]);

  // Normalizations
  Real wsuml = 0.0;
  Real wsumr = 0.0;

  // WENO-A scheme introduced in Wang, Wang, and Don, 2019.
  for (int l = 0; l < 3; l++) {
    wl[l] = w5gamma[l]*std::max(1., PHI*std::pow(tau5/beta[l],p));
    wr[l] = w5gamma[l]*std::max(1., PHI*std::pow(tau5/beta[2-l],p));
    wsuml += wl[l];
    wsumr += wr[l];
  }
  for (int l = 0; l < 3; l++) {
    wl[l] /= wsuml;
    wr[l] /= wsumr;
  }

  ql(dir,n,k+dk,j+dj,i+di) = wl[0]*tl[0] + wl[1]*tl[1] + wl[2]*tl[2];
  qr(dir,n,k,j,i) = wr[0]*tr[0] + wr[1]*tr[1] + wr[2]*tr[2];
}

// MP5, lifted shamelessly from nubhlight, which was lifted shamelessly from PLUTO
#define MINMOD(a, b) ((a) * (b) > 0.0 ? (fabs(a) < fabs(b) ? (a) : (b)) : 0.0)
KOKKOS_INLINE_FUNCTION
double Median(double a, double b, double c) {
  return (a + MINMOD(b - a, c - a));
}
KOKKOS_INLINE_FUNCTION
double mp5_subcalc(
    double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2) {
  double        f, d2, d2p, d2m;
  double        dMMm, dMMp;
  double        scrh1, scrh2, Fmin, Fmax;
  double        fAV, fMD, fLC, fUL, fMP;
  constexpr double alpha = 4.0, epsm = 1.e-12;

  f = 2.0 * Fjm2 - 13.0 * Fjm1 + 47.0 * Fj + 27.0 * Fjp1 - 3.0 * Fjp2;
  f /= 60.0;

  fMP = Fj + MINMOD(Fjp1 - Fj, alpha * (Fj - Fjm1));

  if ((f - Fj) * (f - fMP) <= epsm)
    return f;

  d2m = Fjm2 + Fj - 2.0 * Fjm1; // Eqn. 2.19
  d2  = Fjm1 + Fjp1 - 2.0 * Fj;
  d2p = Fj + Fjp2 - 2.0 * Fjp1; // Eqn. 2.19

  scrh1 = MINMOD(4.0 * d2 - d2p, 4.0 * d2p - d2);
  scrh2 = MINMOD(d2, d2p);
  dMMp  = MINMOD(scrh1, scrh2); // Eqn. 2.27
  scrh1 = MINMOD(4.0 * d2m - d2, 4.0 * d2 - d2m);
  scrh2 = MINMOD(d2, d2m);
  dMMm  = MINMOD(scrh1, scrh2); // Eqn. 2.27

  fUL = Fj + alpha * (Fj - Fjm1);                   // Eqn. 2.8
  fAV = 0.5 * (Fj + Fjp1);                          // Eqn. 2.16
  fMD = fAV - 0.5 * dMMp;                           // Eqn. 2.28
  fLC = 0.5 * (3.0 * Fj - Fjm1) + 4.0 / 3.0 * dMMm; // Eqn. 2.29

  scrh1 = fmin(Fj, Fjp1);
  scrh1 = fmin(scrh1, fMD);
  scrh2 = fmin(Fj, fUL);
  scrh2 = fmin(scrh2, fLC);
  Fmin  = fmax(scrh1, scrh2); // Eqn. (2.24a)

  scrh1 = fmax(Fj, Fjp1);
  scrh1 = fmax(scrh1, fMD);
  scrh2 = fmax(Fj, fUL);
  scrh2 = fmax(scrh2, fLC);
  Fmax  = fmin(scrh1, scrh2); // Eqn. 2.24b

  f = Median(f, Fmin, Fmax); // Eqn. 2.26
  return f;
}
#undef MINMOD

template <typename T, typename Tq>
KOKKOS_INLINE_FUNCTION
void MP5(const int d, const int n,
         const int k, const int j, const int i,
         const T &v, const Tq &ql, const Tq &qr) {
  const int dir = d-1;
  const int di = (d == X1DIR ? 1 : 0);
  const int dj = (d == X2DIR ? 1 : 0);
  const int dk = (d == X3DIR ? 1 : 0);
  ql(dir,n,k+dk,j+dj,i+di) = mp5_subcalc(v(n,k-2*dk,j-2*dj,i-2*di),
                                         v(n,k-dk,j-dj,i-di),
                                         v(n,k,j,i),
                                         v(n,k+dk,j+dj,i+di),
                                         v(n,k+2*dk,j+2*dj,i+2*di));
  qr(dir,n,k,j,i) = mp5_subcalc(v(n,k+2*dk,j+2*dj,i+2*di),
                                v(n,k+dk,j+dj,i+di),
                                v(n,k,j,i),
                                v(n,k-dk,j-dj,i-di),
                                v(n,k-2*dk,j-2*dj,i-2*di));
}

} // PhoebusReconstruction

#endif // RECONSTRUCTION_HPP_
