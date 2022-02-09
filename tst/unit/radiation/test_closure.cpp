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

#include "catch2/catch.hpp"
#include "radiation/closure.hpp"

// parthenon includes
#include <coordinates/coordinates.hpp>
#include <defs.hpp>
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>

// phoebus includes
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "geometry/tetrads.hpp"
#include <geometry/minkowski.hpp>
#include "../../test_utils.hpp"

using namespace Geometry;
using namespace radiation; 

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

const Real pi = acos(-1);

TEST_CASE("M1 Closure", "[radiation][closure]") { 
  GIVEN("A background fluid state and values for J and tilde H_i") {
    THEN("We can perform a Prim2Con followed by a Con2Prim and recover J and tilde H_i") {
      int n_wrong = 0;
      int n_total = 0;
      FILE *fptr;
      fptr = fopen("Closure_error.out","w");

      for (double phiv = 0; phiv <= 0.0; phiv += pi*1.e-2) {
        for (double vmag = 0.0; vmag < 0.6; vmag += 1.e-2) { 
          for (double xi = 1.e-10; xi<=1.0; xi +=1.e-2) {
         
            // Set up background state
            Vec con_v = {vmag*cos(phiv), vmag*sin(phiv), 0.0}; 
            Tens2 cov_gamma = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}}; 
            LocalThreeGeometry<Vec, Tens2> g(cov_gamma); 
            Closure<Vec, Tens2> cl(con_v, &g); 
  
  
            // Assume a fluid frame state  
            Real J = 1.0;
            Vec cov_tilH = {{xi*J/sqrt(g.con_gamma(0,0) - con_v(0)*con_v(0)), 0.0, 0.0}}; 
            Tens2 con_tilPi;

            // Calculate comoving frame state 
            Real E;
            Vec F;
            cl.GetCovTilPiFromPrimM1(J, cov_tilH, &con_tilPi);
            cl.Prim2Con(J, cov_tilH, con_tilPi, &E, &F); 
              
            
            // re-Calculate rest frame quantities using closure 
            // to check for self-consistency
            Real J_out; 
            Vec H_out;
            
            Real xig, phig; 
            cl.GetM1GuessesFromEddington(E, F, &xig, &phig); 
            cl.GetCovTilPiFromConM1(E, F, xig, phig, &con_tilPi);
            cl.Con2Prim(E, F, con_tilPi, &J_out, &H_out);
            
            //if (result.status == radiation::Status::failure) throw 2;
            bool bad_solution = true;
            if ((fabs(J - J_out)/J < 1.e-5) && (fabs(cov_tilH(0) - H_out(0))/J < 1.e-5)) bad_solution = false; 
            
            if (bad_solution) {
              fprintf(fptr, " \033[1;31m[FAILURE]\033[0m\n");
              fprintf(fptr, " \033[1;32mvmag: %e v^i : (%e, %e, %e) xi : %e \033[0m\n", vmag, con_v(0), con_v(1), con_v(2), xi);
              fprintf(fptr, "    E : %e     F_i : (%e, %e, %e) \n", E, F(0), F(1), F(2));
              fprintf(fptr, "    J : %e  tilH_i : (%e, %e, %e) \n", J, cov_tilH(0), cov_tilH(1), cov_tilH(2));
              fprintf(fptr, "J_out : %e H_out_i : (%e, %e, %e) \n\n", J_out, H_out(0), H_out(1), H_out(2));
              ++n_wrong;
            }
            ++n_total;
          }
        }
      }
      fclose(fptr);
      printf("Total points : %i  Points wrong : %i \% wrong: %e\n", n_total, n_wrong, (double)n_wrong/ n_total);
      REQUIRE(n_wrong == 0); 
    }
  }
}
