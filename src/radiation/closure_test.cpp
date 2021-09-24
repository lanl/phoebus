#include <stdio.h> 

#include "closure.hpp"

using namespace radiationClosure; 

struct Vec { 
  Real data[NDSPACE]; 
  inline Real& operator()(const int idx){return data[idx];}
  inline const Real& operator()(const int idx) const {return data[idx];}
};

struct Tens2 { 
  Real data[NDSPACE][NDSPACE]; 
  inline Real& operator()(const int i, const int j){return data[i][j];} 
  inline const Real& operator()(const int i, const int j) const {return data[i][j];} 
};

int main(int /*nargin*/, char** /*args*/) {
  
  // Set up background state
  Vec con_v = {0.01, -0.7, 0.0}; 
  //Vec con_v = {0.0, 0.0, 0.0}; 
  Tens2 cov_gamma = {{{1.0, 0.1, 0.0}, {0.1, 1.0, 0.0}, {0.0, 0.0, 1.0}}}; 
  
  Closure<Vec, Tens2> cl(con_v, cov_gamma); 
  
  const Real pi = 3.14159;
  
  // Assume a fluid frame state  
  Real J = 1.0;
  Vec cov_H = {{0.3, 0.0, 0.1}}; 
  Tens2 con_tilPi;
  //Vec con_tilf; 
  //cl.M1FluidPressureTensor(J, cov_H, &con_tilPi, &con_tilf);

  // Calculate comoving frame state 
  Real E;
  Vec F;
  //cl.Prim2Con(J, cov_H, con_tilPi, &E, &F);  

  cl.Prim2ConM1(J, cov_H, &E, &F, &con_tilPi); 
  
  //Real E = 1.0;
  //Vec F = {{0.000001, 0.0, 0.0}};  
  //Real fXi, fPhi;
  //Real xi = 0.8; 
  //Real phi = pi;
  //cl.SolveClosure(E, F, &xi, &phi); 

  /*
  Real Jac[2][2];
  Real f[2], dx[2];
  Real zeta = std::log(1/0.3-1.0);
  int iter = 0;
  const Real eps = std::sqrt(3.0*std::numeric_limits<Real>::epsilon()); 
  const Real delta = 3.0*std::numeric_limits<Real>::epsilon();
  do {
    Real fPhi_up, fPhi_lo;
    Real fXi_up, fXi_lo;
    
    const Real zeta_up = zeta*(1.0 + eps) + delta;
    const Real zeta_lo = zeta*(1.0 - eps) - delta;
    xi = 1/(1 + std::exp(zeta));
    Real xi_up = 1/(1 + std::exp(zeta_up)); 
    Real xi_lo = 1/(1 + std::exp(zeta_lo));
    cl.M1Residuals(E, F, xi_up, phi, &fXi_up, &fPhi_up);
    cl.M1Residuals(E, F, xi_lo, phi, &fXi_lo, &fPhi_lo);
    Jac[0][0] =  (fXi_up - fXi_lo)/(zeta_up - zeta_lo);
    Jac[1][0] =  (fPhi_up - fPhi_lo)/(zeta_up - zeta_lo);
    
    const Real phi_up = phi*(1.0 + eps) + delta;
    const Real phi_lo = phi*(1.0 - eps) - delta;
    cl.M1Residuals(E, F, xi, phi_up, &fXi_up, &fPhi_up);
    cl.M1Residuals(E, F, xi, phi_lo, &fXi_lo, &fPhi_lo);
    Jac[0][1] =  (fXi_up - fXi_lo)/(phi_up - phi_lo);
    Jac[1][1] =  (fPhi_up - fPhi_lo)/(phi_up - phi_lo);
    
    cl.M1Residuals(E, F, xi, phi, &fXi, &fPhi);
    f[0] = fXi; 
    f[1] = fPhi; 
    
    //printf("{{ %e, %e },{ %e, %e}}\n",Jac[0][0], Jac[0][1], Jac[1][0], Jac[1][1]);
    //printf("xi : %e phi/pi : %e fXi : %e fPhi : %e\n", xi, phi/pi, fXi, fPhi);  
    
    SolveAxb2x2(Jac, f, dx); 
    
    zeta -= dx[0];  
    phi -= dx[1];  
    
    ++iter;
  } while (iter<50 && std::abs(dx[0])/(std::abs(zeta) + delta) > eps);

  printf("iter : %i xi : %e phi/pi : %e fXi : %e fPhi : %e\n", iter, xi, phi/pi, fXi, fPhi);  
  */  
  // Calculate rest frame quantities using closure 
  //cl.M1FluidPressureTensor(E, F, xi, phi, &con_tilPi, &con_tilf); 
  Real J_out; 
  Vec H_out;
  //cl.Con2Prim(E, F, con_tilPi, &J_out, &H_out);
  cl.Con2PrimM1(E, F, &J_out, &H_out, &con_tilPi);
  printf("    E : %e     F_i : (%e, %e, %e) \n", E, F(0), F(1), F(2));
  printf("    J : %e     H_i : (%e, %e, %e) \n", J, cov_H(0), cov_H(1), cov_H(2));
  printf("J_out : %e H_out_i : (%e, %e, %e) \n", J_out, H_out(0), H_out(1), H_out(2));

  /* 
  // Set up background state
  Real con_v[3] = {0.1, 0.3, 0.3}; 
  //Real con_v[3] = {0.0, 0.0, 0.0}; 
  Real cov_gamma[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}; 
  CellBackground bg(con_v, cov_gamma);

  // Calculate rest frame quantities that will serve as reference state 
  Real J = 1.0; 
  Real cov_H[3] = {0.3, 1.015, 0.1};  
  Real H2(0.0), vH(0.0); 
  SPACELOOP(i) { SPACELOOP(j){ H2 += cov_H[i]*cov_H[j]*bg.con_gamma[i][j]; }} 
  SPACELOOP(i) { vH += cov_H[i]*bg.con_v[i]; }
  H2 -= vH*vH;
  std::cout << H2 << std::endl;
  Real con_tilPi[3][3] = {{0.5, 0.3, 0.0}, {0.3, 0.1, 0.0}, {0.0, 0.0, 0.0}}; 
  Real con_H[3]; 
  bg.raise3Vector(cov_H, con_H);
  Real xi = std::sqrt(H2/(J*J));
  Real athin = 0.5*(3*closure(xi) - 1);
  SPACELOOP(i) {
    SPACELOOP(j) {
      con_tilPi[i][j] = athin*(con_H[i]*con_H[j]/H2 
                        - (bg.W2*bg.con_v[i]*bg.con_v[j] + bg.con_gamma[i][j])/3);
    }
  }
  // Calculate E and F from rest frame state   
  Real E;
  Real cov_F[3];
  RadPrim2Con(J, cov_H, con_tilPi, bg, &E, cov_F);
 

  // Invert and find new guess at inverse state
  Real J_out; 
  Real cov_H_out[3];
  RadCon2Prim(E, cov_F, con_tilPi, bg, &J_out, cov_H_out); 
  printf("\n");
  printf("   J : %f (%f)   E : %f \n", J_out, J, E);
  printf(" H^1 : %f (%f) F^1 : %f \n", cov_H_out[0], cov_H[0], cov_F[0]);
  printf(" H^2 : %f (%f) F^2 : %f \n", cov_H_out[1], cov_H[1], cov_F[1]);
  printf(" H^3 : %f (%f) F^3 : %f \n", cov_H_out[2], cov_H[2], cov_F[2]);
  
  Real delta[3][3];
  SPACELOOP(i) {
    SPACELOOP(j) {
      delta[i][j] = 0.0; 
      SPACELOOP(k) { 
        delta[i][j] += bg.cov_gamma[i][k]*bg.con_gamma[k][j];
      }
    }
  }
  
  Real xi_out = findM1Xi(E, cov_F, bg, xi);
  printf("\n xi : %f (%f) \n", xi_out, xi);

  printf("\n gamma^{kj}\n");
  SPACELOOP(i) printf(" %f %f %f\n", bg.con_gamma[i][0], bg.con_gamma[i][1], bg.con_gamma[i][2]);
  printf("\n");
  
  printf("\n gamma_{ik}gamma^{kj}\n");
  SPACELOOP(i) printf(" %f %f %f\n", delta[i][0], delta[i][1], delta[i][2]);
  printf("\n");
  */
  return 0; 
}