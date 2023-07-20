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
Breif: General header for Z4c formulation
       In this impelementation, we follow the Z4c formulation that is reported in
       Bernuzzi & Hilditch (2010) and Hilditch et al. (2013).
       The Z4c system is comprised of dynamical variable:
       chi: Conformal factor (scalar)
       gt : Induced 3-metric (rank-2)
       Kh : Trace of extrinsic curvature (scalar)
       At : Traceless of extrinsic curvature (rank-2)
       Theta : Temporal projection of Z4 vector (scalar)
       Gt : Conformaly-related function (3-vector)

       In addtion, we follow typical gauge choice:
       Bona-Masso for lapse, alpha (scalar) \TODO :maybe 1+log?
       Gamma-driver for shift, beta (3-vector) and B (3-vector)       
Date : Jul.19.2023
Author : Hyun Lim
*/

#ifndef Z4C_HPP_
#define Z4C_HPP_

#include "fd_compute.hpp"

//TODO : Link to Phoebus

#endif // Z4C_HPP_
