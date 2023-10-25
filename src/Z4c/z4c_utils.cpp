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
Breif : Various utils for Z4c implementation
Date : Oct.2.2023
Author : Hyun Lim
*/

// stdlib
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Parthenon
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Phoebus
#ifndef PHOEBUS_IN_UNIT_TESTS
#include "geometry/geometry.hpp"
#endif // PHOEBUS_UNIT_TESTS

#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/robust.hpp"

// Z4c header to contain some utils including FD computations
#include "z4c.hpp"

// Some macro
// TODO : maybe define different way?
// Random number in [-1,1]
std::default_random_engine generator{137};
std::uniform_real_distribution<double> distribution(-1.,1.);
#define RANDOMNUMBER (distribution(generator))

#define NDIM    (3)

#define IX_IL                                                                 \
  pmy_block->is

#define IX_IU                                                                 \
  pmy_block->ive

#define IX_JL                                                                 \
  pmy_block->js

#define IX_JU                                                                 \
  pmy_block->jve

#define IX_KL                                                                 \
  pmy_block->ks

#define IX_KU                                                                 \
  pmy_block->kve

#define GSIZEI                                                                \
  (NGHOST)

#define GSIZEJ                                                                \
  ((pmy_block->block_size.nx2 > 1) ? (NGHOST) : (0))

#define GSIZEK                                                                \
  ((pmy_block->block_size.nx3 > 1) ? (NGHOST) : (0))

// 2D loop over k and j in the interior of the block
#define ILOOP2(k,j)                                                           \
  for(int k = IX_KL; k <= IX_KU; ++k)                         \
  for(int j = IX_JL; j <= IX_JU; ++j)

// 2D loop over k and j on the whole block
#define GLOOP2(k,j)                                                           \
  for(int k = IX_KL - GSIZEK; k <= IX_KU + GSIZEK; ++k)       \
  for(int j = IX_JL - GSIZEJ; j <= IX_JU + GSIZEJ; ++j)

// 1D loop over i in the interior of the block
#define ILOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = IX_IL; i <= IX_IU; ++i)

// 1D loop over i on the whole block
#define GLOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = IX_IL - GSIZEI; i <= IX_IU + GSIZEI; ++i)

// 3D loop over the interior of the block
#define ILOOP3(k,j,i)                                                         \
    ILOOP2(k,j)                                                               \
    ILOOP1(i)

// 3D loop over the whole block
#define GLOOP3(k,j,i)                                                         \
    GLOOP2(k,j)                                                               \
    GLOOP1(i)

#endif

using namespace parthenon::package::prelude;
using parthenon::AllReduce;
using parthenon::MetadataFlag;

//Initialization of vars for robust stability test.
// Note that noise amplitude must be resacled with respect to
// appropriately based on resolution
void Z4c::ADMRobustStability(AthenaArray<Real> & u_adm) {
  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Real const amp = opt.AwA_amplitude;

  // Flat spacetime
  ADMMinkowski(u_adm);

  GLOOP2(k,j) {
    // g_ab
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        GLOOP1(i) {
          adm.g_dd(a,b,k,j,i) += RANDOMNUMBER * amp;
        }
      }
    // K_ab
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        GLOOP1(i) {
          adm.K_dd(a,b,k,j,i) += RANDOMNUMBER * amp;
        }
      }
  }
}

void Z4c::GaugeRobStab(AthenaArray<Real> & u) {
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);
  z4c.alpha.Fill(1.);
  z4c.beta_u.Fill(0.);

  Real const amp = opt.AwA_amplitude;

  GLOOP2(k,j) {
    GLOOP1(i) {
      // lapse
      z4c.alpha(k,j,i) += RANDOMNUMBER * amp;
      // shift
      z4c.beta_u(0,k,j,i) += RANDOMNUMBER * amp;
      z4c.beta_u(1,k,j,i) += RANDOMNUMBER * amp;
      z4c.beta_u(2,k,j,i) += RANDOMNUMBER * amp;
    }
  }
}



#endif //PHOEBUS_UNIT_TEST

