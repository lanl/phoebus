#include "con2prim.hpp"

#include <cstdio>

namespace con2prim {

//static int calls = 0;
//static Real avg_iters = 0.0;

template <typename Data_t,typename T>
ConToPrimStatus ConToPrim<Data_t,T>::Solve(const VarAccessor<T> &v, const CellGeom &g, const bool print) const {
  // converge on rho and T
  // constraints: rho <= D, T > 0

  const Real igdet = 1./g.gdet;
  const Real D = v(crho)*igdet;
  const Real tau = v(ceng)*igdet;

  // electron fraction
  if(pye > 0) v(pye) = v(cye)/v(crho);

  Real BdotS = 0.0;
  Real Bsq = 0.0;
  // bfield
  if (pb_hi > 0) {
    // set primitive fields
    for (int i = 0; i < 3; i++) {
      v(pb_lo+i) = v(cb_lo+i)*igdet;
    }
    // take some dot products
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Bsq += g.gcov[i][j] * v(pb_lo+i)*v(pb_lo+j);
      }
      BdotS += v(pb_lo+i)*v(cmom_lo+i);
    }
    // don't forget S_j has a \sqrt{gamma} in it...get rid of it here
    BdotS *= igdet;
  }
  const Real BdotSsq = BdotS*BdotS;

  Real Ssq = 0.0;
  Real W = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Ssq += g.gcon[i][j] * v(cmom_lo+i)*v(cmom_lo+j);
      W += g.gcov[i][j] * v(pvel_lo+i)*v(pvel_lo+j);
    }
  }
  Ssq *= igdet*igdet;
  W = sqrt(1.0/(1.0 - W));
  Real rho_guess = D/W;
  Real T_guess = v(tmp);

  Residual Rfunc(D, tau, Bsq, Ssq, BdotSsq, eos);

  int iter = 0;
  bool converged = false;
  Real res[2], resp[2];
  Real jac[2][2];
  constexpr Real delta_fact_min = 1.e-8;
  Real delta_fact = 1.e-6;
  constexpr Real delta_adj = 1.2;
  constexpr Real idelta_adj = 1./delta_adj;
  Rfunc(rho_guess, T_guess, res);
  do {
    Real drho = delta_fact*rho_guess;
    Real idrho = 1./drho;
    Rfunc(rho_guess + drho, T_guess, resp);
    jac[0][0] = (resp[0] - res[0])*idrho;
    jac[1][0] = (resp[1] - res[1])*idrho;
    Real dT = delta_fact*T_guess;
    Real idT = 1./dT;
    Rfunc(rho_guess, T_guess+dT, resp);
    jac[0][1] = (resp[0] - res[0])*idT;
    jac[1][1] = (resp[1] - res[1])*idT;

    const Real det = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]);
    const Real idet = 1./det;
    if (std::abs(det) < 1.e-16) {
      delta_fact *= delta_adj;
      iter++;
      continue;
    }
    Real delta_rho = -(res[0]*jac[1][1] - jac[0][1]*res[1])*idet;
    Real delta_T = -(jac[0][0]*res[1] - jac[1][0]*res[0])*idet;

    if (std::abs(delta_rho)/rho_guess < rel_tolerance &&
        std::abs(delta_T)/T_guess < rel_tolerance) {
          converged = true;
    }
    if (print) {
      printf("%d %g %g %g %g %g %g\n",
             iter, rho_guess, T_guess, delta_rho, delta_T, res[0], res[1]);
    }

    if (rho_guess + delta_rho < 0.0) {
      delta_rho = -0.1*rho_guess;
    }
    if (rho_guess + delta_rho > D) {
      delta_rho = D-rho_guess;
    }
    if (T_guess + delta_T < 0.0) {
      delta_T = -0.1*T_guess;
    }

    const Real res0 = res[0]*res[0] + res[1]*res[1];
    Rfunc(rho_guess + delta_rho, T_guess + delta_T, res);
    Real res1 = res[0]*res[0] + res[1]*res[1];
    Real alpha = 1.0;
    int cnt = 0;
    while (res1 >= res0 && cnt < 5) {
           //(std::abs(alpha*delta_rho/rho_guess) > delta_fact*rho_guess ||
           //std::abs(alpha*delta_T/T_guess) > delta_fact*T_guess)) {
      alpha *= 0.5;
      Rfunc(rho_guess + alpha*delta_rho, T_guess + alpha*delta_T, res);
      res1 = res[0]*res[0] + res[1]*res[1];
      cnt++;
    }

    rho_guess += alpha*delta_rho;
    T_guess += alpha*delta_T;
    iter++;

    if (delta_fact > delta_fact_min) delta_fact *= idelta_adj;

  } while(converged != true && iter < max_iter);

  if(!converged) {
    printf("ConToPrim failed state: %g %g %g %g %g %g %g\n",
           rho_guess, T_guess, v(crho),
           v(cmom_lo), v(cmom_lo+1), v(cmom_lo+2),
           v(ceng));
    if (!print) Solve(v,g,true);
    return ConToPrimStatus::failure;
  }

  v(tmp) = T_guess;
  v(prho) = rho_guess;
  v(prs) = eos.PressureFromDensityTemperature(rho_guess, T_guess);
  v(peng) = rho_guess*eos.InternalEnergyFromDensityTemperature(rho_guess, T_guess);
  const Real H = rho_guess + v(peng) + v(prs);
  v(gm1) = eos.BulkModulusFromDensityTemperature(rho_guess, T_guess)/v(prs);

  W = D/rho_guess;
  Real W2 = W*W;
  const Real z = (rho_guess + v(peng) + v(prs))*W2;
  const Real izbsq = 1./(z + Bsq);
  for (int i = 0; i < 3; i++) {
    Real sconi = 0.0;
    for (int j = 0; j < 3; j++) {
      sconi += g.gcon[i][j]*v(cmom_lo+j);
    }
    sconi *= igdet;
    v(pvel_lo+i) = izbsq*(sconi + BdotS*v(pb_lo+i)/z);
  }

  // cell-centered signal speeds
  Real vasq = BdotS/(H*W); // this is just bcon[0]*lapse
  vasq = (Bsq + vasq*vasq); // this is now b^2 * W^2
  vasq /= (H+vasq); // now this is the alven speed squared
  Real cssq = v(gm1)*v(prs)/H;
  cssq += vasq - cssq*vasq; // estimate of fast magneto-sonic speed
  const Real vsq = 1.0 - 1.0/(W2);
  const Real vcoff = g.lapse/(1.0 - vsq*cssq);
  for (int i = 0; i < 3; i++) {
    const Real vpm = sqrt(cssq*(1.0 - vsq)*(g.gcon[i][i]*(1.0 - vsq*cssq) - v(pvel_lo+i)*v(pvel_lo+i)*(1.0 - cssq)));
    Real vp = vcoff*(v(pvel_lo+i)*(1.0 - cssq) + vpm) - g.beta[i];
    Real vm = vcoff*(v(pvel_lo+i)*(1.0 - cssq) - vpm) - g.beta[i];
    v(sig_lo+i) = std::max(std::fabs(vp), std::fabs(vm));
  }

  return ConToPrimStatus::success;
}

template class ConToPrim<MeshBlockData<Real>,VariablePack<Real>>;
//template class ConToPrim<MeshData<Real>,MeshBlockPack<Real>>;

} // namespace con2prim
