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

/*
Brief : Various finite difference computations for handling
        spatial derivative.
Date : Jul.19.2023
Author : Hyun Lim
*/

#ifndef FD_COMPUTE_HPP_
#define FD_COMPUTE_HPP_

#include "fd_stencil.hpp"

//TODO : Coupling parthenon looping i.e. Kokkos

// Define stride and index
int stride[3];
Real idx[3];

//Dissipation constant
Real diss;

//First derivative (second order centered scheme)
inline Real Dx(int dif, Real &u) {
  
  for(auto n1){
    constexpr int n2 - 
 
  }    

}



#endif // FD_COMPUTE_HPP_
