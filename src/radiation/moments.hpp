//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef MOMENTS_HPP_
#define MOMENTS_HPP_

#define NDSPACE 3
#define KOKKOS_FORCEINLINE_FUNCTION inline 

#define SPACELOOP(i) for (int i = 0; i < NDSPACE; ++i) 
#include <cmath>

/// TODO: Add parthenon includes

/// TODO: Add Opacity includes 

/// TODO: Add phoebus includes and switch to Phoebus geometry and variable arrays 

/// TODO: Should this just be part of the radiation namespace? 

namespace radiationMoments { 

typedef double Real;
// Taken from https://pharr.org/matt/blog/2019/11/03/difference-of-floats 
// meant to reduce floating point rounding error in cancellations,
// returns ab - cd
template <class T> 
inline 
T DifferenceOfProducts(T a, T b, T c, T d) {
    T cd = c * d;
    T err = std::fma(-c, d, cd); // Fused multiply add that is rounded only once
    T dop = std::fma(a, b, -cd);
    return dop + err;
}

template<class T> 
inline 
void matrixInverse3x3(T A[3][3], T Ainv[3][3]) {
  Ainv[0][0] = DifferenceOfProducts(A[1][1], A[2][2], A[1][2], A[2][1]);
  Ainv[1][0] =-DifferenceOfProducts(A[1][0], A[2][2], A[1][2], A[2][0]);
  Ainv[2][0] = DifferenceOfProducts(A[1][0], A[2][1], A[1][1], A[2][0]);
  Ainv[0][1] =-DifferenceOfProducts(A[0][1], A[2][2], A[0][2], A[2][1]);
  Ainv[1][1] = DifferenceOfProducts(A[0][0], A[2][2], A[0][2], A[2][0]);
  Ainv[2][1] =-DifferenceOfProducts(A[0][0], A[2][1], A[0][1], A[2][0]);
  Ainv[0][2] = DifferenceOfProducts(A[0][1], A[1][2], A[0][2], A[1][1]);
  Ainv[1][2] =-DifferenceOfProducts(A[0][0], A[1][2], A[0][2], A[1][0]);
  Ainv[2][2] = DifferenceOfProducts(A[0][0], A[1][1], A[0][1], A[1][0]);
  const T det = std::fma(A[0][0], Ainv[0][0], 
      DifferenceOfProducts(A[2][0], Ainv[2][0], A[1][0], -Ainv[1][0])); 
  const T invDet = 1.0/det; 
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) { 
      Ainv[i][j] *= invDet;
    }
  }
}

enum class ClosureStatus { success=0, failure=1 }; 

// Contains static information about the geometry and fluid state needed 
// for the radiation moments 
struct CellBackground {
  CellBackground(const Real con_v_in[NDSPACE], const Real cov_gamma_in[NDSPACE][NDSPACE])
  {
    SPACELOOP(i) {
      SPACELOOP(j) {
        cov_gamma[i][j] = cov_gamma_in[i][j];
      }
    }
    SPACELOOP(i) con_v[i] = con_v_in[i];
    
    matrixInverse3x3(cov_gamma, con_gamma); 

    lower3Vector(con_v, cov_v);
    double v2 = 0.0; 
    SPACELOOP(i) v2 += con_v[i]*cov_v[i]; 
    W = 1/std::sqrt(1 - v2);  
    W2 = W*W; 
    W3 = W*W2; 
    W4 = W*W3; 
  }
  double W, W2, W3, W4;
  double cov_v[NDSPACE]; 
  double con_v[NDSPACE]; 
  double cov_gamma[NDSPACE][NDSPACE]; 
  double con_gamma[NDSPACE][NDSPACE]; 
  
  void lower3Vector(const Real con_U[NDSPACE], Real cov_U[NDSPACE]) const {
    SPACELOOP(i) {
      cov_U[i] = 0.0;
      SPACELOOP(j) { 
        cov_U[i] += con_U[j]*cov_gamma[i][j];
      }
    }
  }

  void raise3Vector(const Real cov_U[NDSPACE], Real con_U[NDSPACE]) const {
    SPACELOOP(i) {
      con_U[i] = 0.0;
      SPACELOOP(j) { 
        con_U[i] += cov_U[j]*con_gamma[i][j];
      }
    }
  }

};

KOKKOS_FORCEINLINE_FUNCTION
void GetTilPiContractions(const Real con_TilPi[NDSPACE][NDSPACE],
                          const CellBackground &bg,
                          Real con_vPi[NDSPACE], Real* vvPi) {
*vvPi = 0.0;
SPACELOOP(i) { 
    con_vPi[i] = 0.0;
    SPACELOOP(j) {
      con_vPi[i] += bg.cov_v[j]*con_TilPi[i][j]; 
    }
    *vvPi += bg.cov_v[i]*con_vPi[i];
  }
}

KOKKOS_FORCEINLINE_FUNCTION
void RadPrim2Con(const Real J, const Real cov_H[NDSPACE], 
                 const Real con_TilPi[NDSPACE][NDSPACE], 
                 const CellBackground &bg, 
                 Real *E, Real cov_F[NDSPACE]) {
  
  double vvPi, con_vPi[NDSPACE], cov_vPi[NDSPACE]; 
  GetTilPiContractions(con_TilPi, bg, con_vPi, &vvPi); 
  bg.lower3Vector(con_vPi, cov_vPi);
  
  double vH = 0.0; 
  SPACELOOP(i) vH += bg.con_v[i]*cov_H[i];

  *E = (4*bg.W2 - 1 + 3*bg.W2*vvPi)/3*J + 2*bg.W*vH;
  SPACELOOP(i) cov_F[i] = 4*bg.W2/3*bg.cov_v[i]*J + bg.W*bg.cov_v[i]*vH 
                          + bg.W*cov_H[i] + J*cov_vPi[i];
}

KOKKOS_FORCEINLINE_FUNCTION 
void RadCon2Prim(const Real E, const Real cov_F[NDSPACE], 
                 const Real con_TilPi[NDSPACE][NDSPACE], 
                 const CellBackground& bg, 
                 Real* J, Real cov_tilH[NDSPACE]) {
  
  // Calculate contractions of three velocity with projected fluid frame pressure tensor
  double vvPi, con_vPi[NDSPACE], cov_vPi[NDSPACE]; 
  GetTilPiContractions(con_TilPi, bg, con_vPi, &vvPi); 
  bg.lower3Vector(con_vPi, cov_vPi);
  
  double vF = 0.0; 
  SPACELOOP(i) vF += bg.con_v[i]*cov_F[i];
  
  // lam is proportional to the determinant of the 2x2 linear system relating 
  // E and v_i F^i to J and v_i H^i for fixed tilde pi^ij 
  const Real lam = (2.0*bg.W2 + 1.0)/3.0 + (2*bg.W4 - 3*bg.W2)*vvPi; 
  
  // zeta = v_i H^i
  // const Real zeta = -bg.W/lam*(((4*bg.W2-4)/3 + vvPi)*E 
  //                             -((4*bg.W2-1)/3 + bg.W2*vvPi)*vF);
  // a = 4 W^2/3 J + W zeta  
  const Real a = bg.W2/lam*((4*bg.W2/3 - vvPi)*E - ((4*bg.W2+1)/3 - bg.W2*vvPi)*vF); 
  
  // Calculate fluid rest frame (i.e. primitive) quantities
  *J = ((2*bg.W2 - 1)*E - 2*bg.W2*vF)/lam; 
  SPACELOOP(i) cov_tilH[i] = (cov_F[i] - (*J)*cov_vPi[i] - bg.cov_v[i]*a)/bg.W; 
   
}

KOKKOS_FORCEINLINE_FUNCTION 
void getContravariantP(const Real J, const Real cov_tilH[NDSPACE], 
                         const Real con_TilPi[NDSPACE][NDSPACE], 
                         const CellBackground& bg, 
                         Real con_P[NDSPACE][NDSPACE]) {
  Real con_tilH[NDSPACE]; 
  bg.raise3Vector(cov_tilH, con_tilH);
  SPACELOOP(i) {
    SPACELOOP(j) {
      con_P[i][j] = (4/3*bg.W2*bg.con_v[i]*bg.con_v[j] + bg.con_gamma[i][j]/3)*J 
          + bg.W*(bg.con_v[i]*con_tilH[j] + bg.con_v[j]*con_tilH[i])
          + J*con_TilPi[i][j];
    }
  }
}

} // namespace radiationMoments 

#endif // MOMENTS_HPP_