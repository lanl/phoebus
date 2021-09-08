#include <stdio.h> 

#include "moments.hpp"

using namespace radiationMoments; 

int main(int /*nargin*/, char** /*args*/) {
  Real con_v[3]; 
  con_v[0] = 0.6; con_v[1] = 0.3; con_v[2] = 0.1;  
  
  Real cov_gamma[3][3] = {{1.0, 0.1, 0.0}, {0.1, 2.0, 0.0}, {0.0, 0.0, 2.0}}; 
  Real con_tilPi[3][3] = {{0.5, 0.3, 0.0}, {0.3, 0.1, 0.0}, {0.0, 0.0, 0.0}}; 
  
  CellBackground bg(con_v, cov_gamma);

  Real J = 1.0; 
  Real cov_H[3] = {0.1, 0.3, 0};  
  
  Real E;
  Real cov_F[3];
  RadPrim2Con(J, cov_H, con_tilPi, bg, &E, cov_F);

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
  
  printf("\ngamma^{kj}\n");
  SPACELOOP(i) printf(" %f %f %f\n", bg.con_gamma[i][0], bg.con_gamma[i][1], bg.con_gamma[i][2]);
  printf("\n");
  
  printf("\ngamma_{ik}gamma^{kj}\n");
  SPACELOOP(i) printf(" %f %f %f\n", delta[i][0], delta[i][1], delta[i][2]);
  printf("\n");
  return 0; 
}