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

#ifndef CLOSURE_HPP_
#define CLOSURE_HPP_

#define NDSPACE 3
#define KOKKOS_FORCEINLINE_FUNCTION inline 
#define KOKKOS_INLINE_FUNCTION inline 

#define SPACELOOP(i) for (int i = 0; i < NDSPACE; ++i) 

#include <cmath>
#include <iostream>
#include <type_traits>
#include <limits>

/// TODO: Add parthenon includes

/// TODO: Add Opacity includes 

/// TODO: Add phoebus includes and switch to Phoebus geometry and variable arrays 

/// TODO: Should this just be part of the radiation namespace? 

namespace radiationClosure { 

typedef double Real;

// Taken from https://pharr.org/matt/blog/2019/11/03/difference-of-floats 
// meant to reduce floating point rounding error in cancellations,
// returns ab - cd
template <class T> 
inline 
T DifferenceOfProducts(T a, T b, T c, T d) {
    T cd = c * d;
    T err = std::fma(-c, d, cd); // Round off error correction for cd
    T dop = std::fma(a, b, -cd);
    return dop + err;
}

template<class T> 
inline 
auto matrixInverse3x3_dos(T A, T& Ainv) 
    -> typename std::remove_reference<decltype(A(0,0))>::type {
  Ainv(0,0) = DifferenceOfProducts(A(1,1), A(2,2), A(1,2), A(2,1));
  Ainv(1,0) =-DifferenceOfProducts(A(1,0), A(2,2), A(1,2), A(2,0));
  Ainv(2,0) = DifferenceOfProducts(A(1,0), A(2,1), A(1,1), A(2,0));
  Ainv(0,1) =-DifferenceOfProducts(A(0,1), A(2,2), A(0,2), A(2,1));
  Ainv(1,1) = DifferenceOfProducts(A(0,0), A(2,2), A(0,2), A(2,0));
  Ainv(2,1) =-DifferenceOfProducts(A(0,0), A(2,1), A(0,1), A(2,0));
  Ainv(0,2) = DifferenceOfProducts(A(0,1), A(1,2), A(0,2), A(1,1));
  Ainv(1,2) =-DifferenceOfProducts(A(0,0), A(1,2), A(0,2), A(1,0));
  Ainv(2,2) = DifferenceOfProducts(A(0,0), A(1,1), A(0,1), A(1,0));
  const auto det = std::fma(A(0,0), Ainv(0,0), 
      DifferenceOfProducts(A(2,0), Ainv(2,0), A(1,0), -Ainv(1,0))); 
  const auto invDet = 1.0/det; 
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) { 
      Ainv(i,j) *= invDet;
    }
  }
  return det;
}

template<class T>
inline 
void SolveAxb2x2(T A[2][2], T b[2], T x[2]) {
  const auto invDet = 1.0/DifferenceOfProducts(A[0][0], A[1][1], A[0][1], A[1][0]); 
  x[0] = invDet*DifferenceOfProducts(A[1][1], b[0], A[0][1], b[1]);
  x[1] = invDet*DifferenceOfProducts(A[0][0], b[1], A[1][0], b[0]);
}

enum class Status { success=0, failure=1 };

template <class Vec, class Tens2> 
class Closure {
 public:
  template<class V, class T2> 
  Closure(const V con_v_in, const T2 cov_gamma_in);
  
  // Con2Prim for given rest frame stress tensor 
  template<class VA, class VB, class T2> 
  Status Prim2Con(const Real J, const VA cov_H, const T2 con_tilPi, Real* E, VB* cov_F);
  template<class VA, class VB, class T2> 
  Status Con2Prim(const Real E, const VA cov_F, const T2 con_tilPi, Real* J, VB* cov_H);
  
 
  // Con2Prim and Prim2Con for closure
  template<class VA, class VB, class T2> 
  Status Con2PrimM1(const Real E, const VA cov_F, Real xi_guess, Real phi_guess, 
                    Real* J, VB* cov_H, T2* con_tilPi);
  
  template<class VA, class VB, class T2>  
  Status Con2PrimM1(const Real E, const VA cov_F, Real* J, VB* cov_H, T2* con_tilPi) {
    return Con2PrimM1(E, cov_F, 0.5, 3.14159, J, cov_H, con_tilPi);
  }

  template<class VA, class VB, class T2>  
  Status Prim2ConM1(const Real J, const VA cov_H, Real* E, VB* cov_F, T2* con_tilPi);
  
  // Calculate the momentum density flux
  template<class V, class T2A, class T2B> 
  Status getContravariantPFromPrim(const Real J, const V cov_tilH, const T2A con_TilPi, 
                            T2B* con_P);
   
 protected:
  Real W, W2, W3, W4;
  Vec cov_v; 
  Vec con_v; 
  Real gdet, alpha;
  Vec con_beta;
  Tens2 cov_gamma; 
  Tens2 con_gamma; 
  
  template<class V> 
  Status M1Residuals(const Real E, const V cov_F, 
                              const Real xi, const Real phi, 
                              Real* fXi, Real* fPhi);
  
  template<class V> 
  Status M1FluidPressureTensor(const Real J, const V cov_H, 
                              Tens2* con_tilPi, Vec* con_tilf);

  template<class V> 
  Status M1FluidPressureTensor(const Real E, const V cov_F, 
                              const Real xi, const Real phi, 
                              Tens2* con_tilPi, Vec* con_tilf);
  template<class V> 
  Status SolveClosure(const Real E, const V cov_F, 
                      Real* xi_out, Real* phi_out,
                      const Real xi_guess = 0.5, const Real phi_guess = 3.14159);
  template<class VA, class VB>
  KOKKOS_FORCEINLINE_FUNCTION 
  void lower3Vector(const VA& con_U, VB* cov_U) const {
    SPACELOOP(i) (*cov_U)(i) = 0.0;
    SPACELOOP(i) { SPACELOOP(j) { (*cov_U)(i) += con_U(j)*cov_gamma(i,j); }}
  }
  
  template<class VA, class VB>
  KOKKOS_FORCEINLINE_FUNCTION 
  void raise3Vector(const VA& cov_U, VB* con_U) const {
    SPACELOOP(i) (*con_U)(i) = 0.0;
    SPACELOOP(i) { SPACELOOP(j) { (*con_U)(i) += cov_U(j)*con_gamma(i,j); }}
  }
  
  template<class T2> 
  KOKKOS_FORCEINLINE_FUNCTION
  void GetTilPiContractions(const T2& con_tilPi, Vec* cov_vTilPi, Real* vvTilPi) {
    Vec con_vTilPi; 
    *vvTilPi = 0.0; 
    SPACELOOP(i) {
      con_vTilPi(i) = 0.0;
      SPACELOOP(j) {
        con_vTilPi(i) += cov_v(j)*con_tilPi(i,j); 
      }
      *vvTilPi += cov_v(i)*con_vTilPi(i);
    }
    lower3Vector(con_vTilPi, cov_vTilPi);
  } 
  
  KOKKOS_FORCEINLINE_FUNCTION 
  Real closure(Real xi) {
    // MEFD Closure 
    //return (1.0 - 2*std::pow(xi, 2) + 4*std::pow(xi,3))/3.0;
    return (1.0 + 2*xi*xi)/3; 
  }

 private: 
};

template<class Vec, class Tens2> 
template<class V, class T2> 
Closure<Vec, Tens2>::Closure(const V con_v_in, const T2 cov_gamma_in) { 
  SPACELOOP(i) {
    SPACELOOP(j) {
      cov_gamma(i,j) = cov_gamma_in(i,j);
    }
  }
  
  SPACELOOP(i) con_v(i) = con_v_in(i);

  gdet = matrixInverse3x3_dos(cov_gamma, con_gamma); 

  lower3Vector(con_v, &cov_v);
  Real v2 = 0.0; 
  SPACELOOP(i) v2 += con_v(i)*cov_v(i); 
  W = 1/std::sqrt(1 - v2);  
  W2 = W*W; 
  W3 = W*W2; 
  W4 = W*W3; 
}

template<class Vec, class Tens2> 
template<class VA, class VB, class T2> 
Status Closure<Vec, Tens2>::Prim2Con(const Real J, const VA cov_H, 
                                              const T2 con_tilPi, 
                                              Real* E, VB* cov_F) {
  Real vvPi;
  Vec cov_vPi; 
  GetTilPiContractions(con_tilPi, &cov_vPi, &vvPi); 
  
  Real vH = 0.0; 
  SPACELOOP(i) vH += con_v(i)*cov_H(i);

  *E = (4*W2 - 1 + 3*W2*vvPi)/3*J + 2*W*vH;
  SPACELOOP(i) (*cov_F)(i) = 4*W2/3*cov_v(i)*J + W*cov_v(i)*vH 
                          + W*cov_H(i) + J*cov_vPi(i);                                          
  return Status::success;
}

template<class Vec, class Tens2> 
template<class VA, class VB, class T2> 
Status Closure<Vec, Tens2>::Con2Prim(const Real E, const VA cov_F, 
                                              const T2 con_tilPi, 
                                              Real* J, VB* cov_tilH) {
  Real vvTilPi;
  Vec cov_vTilPi; 
  GetTilPiContractions(con_tilPi, &cov_vTilPi, &vvTilPi);
  
  double vF = 0.0; 
  SPACELOOP(i) vF += con_v(i)*cov_F(i);
  
  // lam is proportional to the determinant of the 2x2 linear system relating 
  // E and v_i F^i to J and v_i H^i for fixed tilde pi^ij 
  const Real lam = (2.0*W2 + 1.0)/3.0 + (2*W4 - 3*W2)*vvTilPi; 
  
  // zeta = v_i H^i
  // const Real zeta = -W/lam*(((4*W2-4)/3 + vvPi)*E 
  //                   -((4*W2-1)/3 + W2*vvPi)*vF);
  // a = 4 W^2/3 J + W zeta  
  const Real a = W2/lam*((4*W2/3 - vvTilPi)*E - ((4*W2+1)/3 - W2*vvTilPi)*vF); 
  
  // Calculate fluid rest frame (i.e. primitive) quantities
  *J = ((2*W2 - 1)*E - 2*W2*vF)/lam; 
  SPACELOOP(i) (*cov_tilH)(i) = (cov_F(i) - (*J)*cov_vTilPi(i) - cov_v(i)*a)/W; 
  return Status::success;
}

template<class Vec, class Tens2> 
template<class VA, class VB, class T2>  
Status Closure<Vec, Tens2>::Prim2ConM1(const Real J, const VA cov_H, 
                                       Real* E, VB* cov_F, T2* con_tilPi) {
  Vec con_tilf;
  M1FluidPressureTensor(J, cov_H, con_tilPi, &con_tilf);   
  Prim2Con(J, cov_H, *con_tilPi, E, cov_F);
  
  return Status::success;
}

template<class Vec, class Tens2> 
template<class VA, class VB, class T2>  
Status Closure<Vec, Tens2>::Con2PrimM1(const Real E, const VA cov_F, 
                                       const Real xi_guess, const Real phi_guess,
                                       Real* J, VB* cov_H, T2* con_tilPi){
  Real xi, phi;
  Vec con_tilf;
  auto status = SolveClosure(E, cov_F, &xi, &phi, xi_guess, phi_guess);
  M1FluidPressureTensor(E, cov_F, xi, phi, con_tilPi, &con_tilf); 
  Con2Prim(E, cov_F, *con_tilPi, J, cov_H);
  return status;
}

template<class Vec, class Tens2> 
template<class V, class T2A, class T2B> 
KOKKOS_FORCEINLINE_FUNCTION 
Status Closure<Vec, Tens2>::getContravariantPFromPrim(const Real J, const V cov_tilH, 
                                                      const T2A con_TilPi, 
                                                      T2B* con_P) {
  Vec con_tilH; 
  raise3Vector(cov_tilH, con_tilH);
  SPACELOOP(i) {
    SPACELOOP(j) {
      (*con_P)(i,j) = (4/3*W2*con_v(i)*con_v(j) + con_gamma(i,j)/3)*J 
          + W*(con_v(i)*con_tilH(j) + con_v(j)*con_tilH(i))
          + J*con_TilPi(i,j);
    }
  }
  return Status::success;
}


template<class Vec, class Tens2> 
template<class V> 
Status Closure<Vec, Tens2>::SolveClosure(const Real E, const V cov_F, Real* xi_out, Real* phi_out,
                                         const Real xi_guess, const Real phi_guess) {
  
  const int max_iter = 50; 
  const Real tol = 1.e3*std::numeric_limits<Real>::epsilon();

  Real zeta = std::log(1/xi_guess-1.0);
  Real xi = xi_guess; 
  Real phi = phi_guess; 

  int iter = 0;
  const Real eps = std::sqrt(3.0*std::numeric_limits<Real>::epsilon()); 
  const Real delta = 3.0*std::numeric_limits<Real>::epsilon();
  Real err_zeta, err_phi;
  Real Jac[2][2], f[2], dx[2];
  do {
    Real fPhi, fPhi_up, fPhi_lo;
    Real fXi, fXi_up, fXi_lo;
    
    const Real zeta_up = zeta*(1.0 + eps) + delta;
    const Real zeta_lo = zeta*(1.0 - eps) - delta;
    xi = 1/(1 + std::exp(zeta));
    Real xi_up = 1/(1 + std::exp(zeta_up)); 
    Real xi_lo = 1/(1 + std::exp(zeta_lo));
    M1Residuals(E, cov_F, xi_up, phi, &fXi_up, &fPhi_up);
    M1Residuals(E, cov_F, xi_lo, phi, &fXi_lo, &fPhi_lo);
    Jac[0][0] =  (fXi_up - fXi_lo)/(zeta_up - zeta_lo);
    Jac[1][0] =  (fPhi_up - fPhi_lo)/(zeta_up - zeta_lo);
    
    const Real phi_up = phi*(1.0 + eps) + delta;
    const Real phi_lo = phi*(1.0 - eps) - delta;
    M1Residuals(E, cov_F, xi, phi_up, &fXi_up, &fPhi_up);
    M1Residuals(E, cov_F, xi, phi_lo, &fXi_lo, &fPhi_lo);
    Jac[0][1] =  (fXi_up - fXi_lo)/(phi_up - phi_lo);
    Jac[1][1] =  (fPhi_up - fPhi_lo)/(phi_up - phi_lo);
    
    M1Residuals(E, cov_F, xi, phi, &fXi, &fPhi);
    f[0] = fXi; 
    f[1] = fPhi; 
    
    SolveAxb2x2(Jac, f, dx); 
    
    zeta -= dx[0];  
    phi -= dx[1];  
    
    err_zeta = std::abs(dx[0])/(std::abs(zeta) + delta);
    err_phi = std::abs(dx[1])/(std::abs(phi) + delta); 
    
    ++iter;
  
  } while ( iter < max_iter &&  (err_zeta > tol || err_phi > tol) );
  
  *xi_out = 1/(1+std::exp(zeta)); 
  *phi_out = phi; 

  if (iter < max_iter) return Status::success;
  return Status::failure; 

}

template<class Vec, class Tens2> 
template<class V> 
Status Closure<Vec, Tens2>::M1Residuals(const Real E, const V cov_F, 
                                        const Real xi, const Real phi,
                                        Real* fXi, Real* fPhi) {

  // Calculate rest frame fluid variables
  Tens2 con_tilPi; 
  Vec con_tilf; 
  M1FluidPressureTensor(E, cov_F, xi, phi, &con_tilPi, &con_tilf); 
  
  Real J; 
  Vec cov_tilH;
  Con2Prim(E, cov_F, con_tilPi, &J, &cov_tilH); 
  
  // Construct the residual functions 
  Real H(0.0), vTilH(0.0), Hf(0.0), vTilf(0.0);
  SPACELOOP(i) { SPACELOOP(j) { H += cov_tilH(i)*cov_tilH(j)*con_gamma(i,j); } } 
  SPACELOOP(i) vTilH += con_v(i)*cov_tilH(i); 
  SPACELOOP(i) Hf += cov_tilH(i)*con_tilf(i); 
  SPACELOOP(i) vTilf += cov_v(i)*con_tilf(i); 
  Hf -= vTilf*vTilH; 
  H = std::sqrt(H - vTilH*vTilH); 
  //printf("tilH = (%f, %f, %f) \n", cov_tilH(0), cov_tilH(1), cov_tilH(2));
  //printf("Hf = %e H = %e \n", Hf, H); 
  *fXi = xi*J - H;  
  *fPhi = Hf - H;
  
  return Status::success;
}

template<class Vec, class Tens2> 
template<class V> 
Status Closure<Vec, Tens2>::M1FluidPressureTensor(const Real J, const V cov_tilH, 
                                                Tens2* con_tilPi, Vec* con_tilf) {
  Vec con_tilH;
  raise3Vector(cov_tilH, &con_tilH); 
  Real H(0.0), vH(0.0); 
  SPACELOOP(i) H += cov_tilH(i)*con_tilH(i);
  SPACELOOP(i) vH += cov_tilH(i)*con_v(i);
  H = std::sqrt(H - vH*vH);  
  SPACELOOP(i) (*con_tilf)(i) = con_tilH(i)/H; 

  const Real athin = 0.5*(3*closure(H/J) - 1); 

  // Calculate the projected rest frame radiation pressure tensor 
  SPACELOOP(i) {
    SPACELOOP(j) {
      (*con_tilPi)(i,j) = (*con_tilf)(i)*(*con_tilf)(j) 
                       - (W2*con_v(i)*con_v(j) + con_gamma(i,j))/3; 
      (*con_tilPi)(i,j) *= athin;  
    }
  }  
  return Status::success;  

}

template<class Vec, class Tens2> 
template<class V> 
Status Closure<Vec, Tens2>::M1FluidPressureTensor(const Real E, const V cov_F, 
                                                const Real xi, const Real phi, 
                                                Tens2* con_tilPi, Vec* con_tilf) {
  
  // Build projected basis vectors for flux direction 
  // These vectors are actually fixed, so there is no need to recalculate 
  // them every time this function is called in the root find
  Vec con_F;
  raise3Vector(cov_F, &con_F); 
  Real Fmag(0.0), vl(0.0), v2(0.0);
  SPACELOOP(i) Fmag += cov_F(i)*con_F(i); 
  Fmag = std::sqrt(Fmag); 
  SPACELOOP(i) vl += cov_F(i)*con_v(i); 
  vl /= Fmag; 
  SPACELOOP(i) v2 += cov_v(i)*con_v(i); 
  const Real lam = 1/(W*(1-vl));
  const Real aa = std::sqrt(v2*(1 + std::numeric_limits<Real>::epsilon()) - vl*vl);
  //const Real aa = std::sqrt(v2 - vl*vl);
  const Real invDenom = 1/(aa + std::numeric_limits<Real>::min());
  Vec con_tilg, con_tild; 
  SPACELOOP(i) {
    con_tilg(i) = con_F(i)/Fmag*lam - W*con_v(i);
    con_tild(i) = (con_F(i)/Fmag*(1-lam/W) + con_v(i))*invDenom;
  }
  
  // Build projected flux direction from basis vectors and given angle 
  const Real s = sin(phi); 
  const Real c = cos(phi); 
  SPACELOOP(i) (*con_tilf)(i) = -c*con_tilg(i) + s*con_tild(i);
  //printf("v2 = %e vl*vl = %e lam = %e invDenom = %e \n", v2, vl*vl, lam, invDenom); 
  //printf("tilg = (%f, %f, %f) \n", con_tilg(0), con_tilg(1), con_tilg(2));
  //printf("tild = (%e, %e, %e) \n", con_tild(0), con_tild(1), con_tild(2));
  //printf("tilf = (%f, %f, %f) \n", (*con_tilf)(0), (*con_tilf)(1), (*con_tilf)(2));

  // Calculate the closure interpolation 
  const Real athin = 0.5*(3*closure(xi) - 1);

  // Calculate the projected rest frame radiation pressure tensor 
  SPACELOOP(i) {
    SPACELOOP(j) {
      (*con_tilPi)(i,j) = (*con_tilf)(i)*(*con_tilf)(j) 
                       - (W2*con_v(i)*con_v(j) + con_gamma(i,j))/3; 
      (*con_tilPi)(i,j) *= athin;  
    }
  }  
  return Status::success; 
}

} // namespace radiationMoments 

#endif // MOMENTS_HPP_