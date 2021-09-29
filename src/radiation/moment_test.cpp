#include <stdio.h> 

#include "moments.hpp"

using namespace radiationMoments; 

int main(int /*nargin*/, char** /*args*/) {
  
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
  return 0; 
}