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

#ifndef LOCAL_THREE_GEOMETRY_HPP_
#define LOCAL_THREE_GEOMETRY_HPP_

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>
#include <limits>

namespace radiation
{

  using namespace LinearAlgebra;
  using namespace robust;


  struct Vec { 
    // Do not add any member data to this struct since it is initialized in many places in 
    // the code using initializer lits 
    Real data[Geometry::NDSPACE]; 
    KOKKOS_FORCEINLINE_FUNCTION
    Real& operator()(const int idx){return data[idx];}
    KOKKOS_FORCEINLINE_FUNCTION
    const Real& operator()(const int idx) const {return data[idx];}
  };
  
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator+(Vec a, Vec b) {Vec out; SPACELOOP(i) out(i) = a(i) + b(i); return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator-(Vec a, Vec b) {Vec out; SPACELOOP(i) out(i) = a(i) - b(i); return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator-(Vec a) {Vec out; SPACELOOP(i) out(i) = -a(i); return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator*(Vec a, Real b) {Vec out; SPACELOOP(i) out(i) = a(i)*b; return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator*(Real a, Vec b) {Vec out; SPACELOOP(i) out(i) = a*b(i); return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Vec operator/(Vec a, Real b) {Vec out; SPACELOOP(i) out(i) = ratio(a(i), b); return out;} 
  
  struct Tens2 {
    // Do not add any member data to this struct since it is initialized in many places in 
    // the code using initializer lits 
    Real data[Geometry::NDSPACE][Geometry::NDSPACE]; 
    KOKKOS_FORCEINLINE_FUNCTION 
    Real& operator()(const int i, const int j){return data[i][j];} 
    KOKKOS_FORCEINLINE_FUNCTION
    const Real& operator()(const int i, const int j) const {return data[i][j];} 
  };
  
  KOKKOS_FORCEINLINE_FUNCTION 
  Tens2 operator+(Tens2 a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j) + b(i,j);} return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Tens2 operator-(Tens2 a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j) - b(i,j);} return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Tens2 operator*(Real a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a*b(i,j);} return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Tens2 operator*(Tens2 a, Real b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j)*b;} return out;} 
  KOKKOS_FORCEINLINE_FUNCTION 
  Tens2 operator/(Tens2 a, Real b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = ratio(a(i,j), b);} return out;}

  struct LocalThreeGeometry {
  
    Real gdet;
    Tens2 cov_gamma;
    Tens2 con_gamma;
    Real alpha;
    Vec con_beta;

    template <class T2>
    KOKKOS_FUNCTION LocalThreeGeometry(const T2 cov_gamma_in, const Real alpha_in = 1.0, 
                                       const Vec cBeta_in = {0,0,0}) {
      SPACELOOP2(i,j){ cov_gamma(i, j) = cov_gamma_in(i, j);}
      gdet = matrixInverse3x3(cov_gamma, con_gamma);
      alpha = alpha_in; 
      con_beta = cBeta_in;
    }
    
    template<class T, class C> 
    KOKKOS_FUNCTION LocalThreeGeometry(const T& geom, const C& face, 
                                       int b, int k, int j, int i) {
      geom.Metric(face, b, k, j, i, cov_gamma.data); 
      gdet = matrixInverse3x3(cov_gamma, con_gamma);
      geom.ContravariantShift(face, b, k, j, i, con_beta.data);
      alpha = geom.Lapse(face, b, k, j, i); 
    }

    //-------------------------------------------------------------------------------------
    /// Routines for raising, lowering, contracting, etc. three vectors.
    template <class VA, class VB>
    KOKKOS_FORCEINLINE_FUNCTION void lower3Vector(const VA &con_U, VB *cov_U) const {
      SPACELOOP(i) (*cov_U)(i) = 0.0;
      SPACELOOP2(i,j){ (*cov_U)(i) += con_U(j) * cov_gamma(i, j); }
    }

    template <class VA, class VB>
    KOKKOS_FORCEINLINE_FUNCTION void raise3Vector(const VA &cov_U, VB *con_U) const {
      SPACELOOP(i) (*con_U)(i) = 0.0;
      SPACELOOP2(i,j){ (*con_U)(i) += cov_U(j) * con_gamma(i, j); }     
    }

    template <class VA, class VB>
    KOKKOS_FORCEINLINE_FUNCTION
    Real contractCon3Vectors(const VA &con_A, const VB &con_B) const {
      Real s(0.0);
      SPACELOOP2(i,j){ s += cov_gamma(i, j) * con_A(i) * con_B(j); }
      return s;
    }

    template <class VA, class VB>
    KOKKOS_FORCEINLINE_FUNCTION
    Real contractCov3Vectors(const VA &cov_A, const VB &cov_B) const
    {
      Real s(0.0);
      SPACELOOP2(i,j){ s += con_gamma(i, j) * cov_A(i) * cov_B(j); }
      return s;
    }

    template <class VA, class VB>
    KOKKOS_FORCEINLINE_FUNCTION
    Real contractConCov3Vectors(const VA &con_A, const VB &cov_B) const {
      Real s(0.0);
      SPACELOOP(i) { s += con_A(i) * cov_B(i); }
      return s;
    }
  };

  using LocaLGeometry = LocalThreeGeometry; 
} // namespace radiation

#endif // LOCAL_THREE_GEOMETRY_HPP_
