// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

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
  Tens2 cov_gamma = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}}; 
  
  Closure<Vec, Tens2> cl(con_v, cov_gamma); 
  
  const Real pi = 3.14159;
  
  // Assume a fluid frame state  
  Real J = 1.0;
  Vec cov_tilH = {{0.8, 0.0, 0.1}}; 
  Tens2 con_tilPi;

  // Calculate comoving frame state 
  Real E;
  Vec F;
  cl.Prim2ConM1(J, cov_tilH, &E, &F, &con_tilPi); 
    
  // re-Calculate rest frame quantities using closure 
  // to check for self-consistency
  Real J_out; 
  Vec H_out;
  cl.Con2PrimM1(E, F, &J_out, &H_out, &con_tilPi);
  
  printf("    E : %e     F_i : (%e, %e, %e) \n", E, F(0), F(1), F(2));
  printf("    J : %e  tilH_i : (%e, %e, %e) \n", J, cov_tilH(0), cov_tilH(1), cov_tilH(2));
  printf("J_out : %e H_out_i : (%e, %e, %e) \n", J_out, H_out(0), H_out(1), H_out(2));

  return 0; 
}