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

#ifndef CLOSURE_M1_HPP_
#define CLOSURE_M1_HPP_

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/closure.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>
#include <limits>

namespace radiation
{

  using namespace LinearAlgebra;
  using namespace robust;

  /// Store results of M1 closure root find
  struct M1Result
  {
    Status status;
    int iter;
    Real xi, phi;
    Real fXi, fPhi;
    Real errXi, errPhi;
  };
  
  /// Holds methods for closing the radiation moment equations as well as calculating radiation
  /// moment source terms.
  template <class Vec, class Tens2, bool ENERGY_CONSERVE = true>
  class ClosureM1 : public ClosureEdd<Vec, Tens2, ENERGY_CONSERVE> 
  {
  protected:
    KOKKOS_FORCEINLINE_FUNCTION
    Real closure(Real xi) {
      // MEFD Closure
      return (1.0 - 2*std::pow(xi, 2) + 4*std::pow(xi,3))/3.0;
      //return (1.0 + 2 * xi * xi) / 3;
    }
  
  public:
    //-------------------------------------------------------------------------------------
    /// Constructor just calculates the inverse 3-metric, covariant three-velocity, and the
    /// Lorentz factor for the given background state.
    KOKKOS_FUNCTION
    ClosureM1(const Vec con_v_in, LocalThreeGeometry<Vec, Tens2>* g) 
        : ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>(con_v_in, g) {}  
    
    KOKKOS_FUNCTION 
    Status GetCovTilPiFromPrim(const Real J, const Vec cov_H, Tens2 *con_tilPi) {
      Vec con_tilf; 
      M1FluidPressureTensor(J, cov_H, con_tilPi, &con_tilf); 
      return Status::success; 
    }
    
    KOKKOS_FUNCTION 
    Status GetCovTilPiFromCon(const Real E, const Vec cov_F, Real& xi, Real& phi, Tens2 *con_tilPi) {
      Vec con_tilf;
      auto status = SolveClosure(E, cov_F, &xi, &phi, xi, phi); 
      M1FluidPressureTensor(cov_F, xi, phi, con_tilPi, &con_tilf); 
      return status.status; 
    } 
    
    KOKKOS_FUNCTION
    void GetM1GuessesFromEddington(const Real E, const Vec cov_F, Real *xi, Real *phi);

    // Make base template class variables dependent names (pretty ugly, I know)
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::Con2Prim;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::Prim2Con;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::v2;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::W;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::W2;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::cov_v;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::con_v;
    using ClosureEdd<Vec, Tens2, ENERGY_CONSERVE>::gamma;
  
  protected:
  
    //-------------------------------------------------------------------------------------
    /// Calculate \tilde \pi^{ij} and \tilde f_i = \tilde H_i/\sqrt{H_\alpha H^\alpha} from
    /// J and \tilde H_i.
    KOKKOS_FUNCTION
    Status M1FluidPressureTensor(const Real J, const Vec cov_H,
                                 Tens2 *con_tilPi, Vec *con_tilf);
  
    //-------------------------------------------------------------------------------------
    /// Calculate the residuals of the M1 root equations
    KOKKOS_FUNCTION
    Status M1Residuals(const Real E, const Vec cov_F,
                       const Real xi, const Real phi,
                       const Vec con_tilg, const Vec con_tild,
                       Real *fXi, Real *fPhi);

    //-------------------------------------------------------------------------------------
    /// Calculate \tilde \pi^{ij} and \tilde f_i = \tilde H_i/\sqrt{H_\alpha H^\alpha} from
    /// E, F_i, \xi = H/J, and \phi that are a root of the M1 residual equations.
    KOKKOS_FUNCTION
    Status M1FluidPressureTensor(const Vec cov_F,
                                 const Real xi, const Real phi,
                                 Tens2 *con_tilPi, Vec *con_tilf) {
      Vec con_tilg, con_tild;
      GetBasisVectors(cov_F, &con_tilg, &con_tild);
      return M1FluidPressureTensor(cov_F, xi, phi, con_tilg, con_tild, con_tilPi, con_tilf);
    }

    //-------------------------------------------------------------------------------------
    /// Calculate \tilde \pi^{ij} and \tilde f_i = \tilde H_i/\sqrt{H_\alpha H^\alpha} from
    /// E, F_i, \xi = H/J, and \phi that are a root of the M1 residual equations. Basis
    /// vectors are explicitly passed so they don't need to be recalculated repeatedly.
    KOKKOS_FUNCTION
    Status M1FluidPressureTensor(const Vec cov_F,
                                 const Real xi, const Real phi,
                                 const Vec con_tilg, const Vec con_tild,
                                 Tens2 *con_tilPi, Vec *con_tilf);

    //-------------------------------------------------------------------------------------
    /// Perform NR iteration to find a root (xi, phi) of the M1 residual equations for a
    /// given E and F_i
    KOKKOS_FUNCTION
    M1Result SolveClosure(Real E, Vec cov_F,
                          Real *xi_out, Real *phi_out,
                          const Real xi_guess = 0.5, const Real phi_guess = 3.14159);

    //-------------------------------------------------------------------------------------
    /// Calculate the basis vectors for \tilde f_i as described in the notes.
    KOKKOS_FUNCTION
    Status GetBasisVectors(const Vec cov_F, Vec *con_tilg, Vec *con_tild);

  };
  
  //---- Method templates instantiated below 

  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  void ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::GetM1GuessesFromEddington(const Real E, const Vec cov_F,
                                           Real *xi, Real *phi) {
    Vec con_tilf;
    
    // Get the basis vectors for the search
    Vec con_tilg, con_tild;
    GetBasisVectors(cov_F, &con_tilg, &con_tild);

    // Perform an Eddington approximation Con2Prim to get initial guesses
    Tens2 con_PiEdd;
    SPACELOOP2(i, j) con_PiEdd(i,j) = 0.0;
    Real JEdd;
    Vec cov_HEdd;
    Con2Prim(E, cov_F, con_PiEdd, &JEdd, &cov_HEdd);
    Real vHEdd = gamma->contractConCov3Vectors(con_v, cov_HEdd);
    Real HEdd = sqrt(gamma->contractCov3Vectors(cov_HEdd, cov_HEdd) - vHEdd * vHEdd);

    *xi = std::min(ratio(HEdd, JEdd), 0.99);
    *phi = 1.000001 * acos(-1);
  }
  
  //----------------
  // Protected functions below
  //----------------
  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  M1Result ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::SolveClosure(Real E, Vec cov_F, Real *xi_out, Real *phi_out,
                                             const Real xi_guess, const Real phi_guess) {
    const int max_iter = 30;
    const Real tol = 1.e6 * std::numeric_limits<Real>::epsilon();
    const Real eps = std::sqrt(10 * std::numeric_limits<Real>::epsilon());
    const Real delta = 20 * std::numeric_limits<Real>::epsilon();

    // Normalize input and make sure we are slightly away from zero
    cov_F(0) = ratio(cov_F(0), E) + 1.e3 * sgn(cov_F(0)) * std::numeric_limits<Real>::min();
    cov_F(1) = ratio(cov_F(1), E) + 1.e3 * sgn(cov_F(1)) * std::numeric_limits<Real>::min();
    cov_F(2) = ratio(cov_F(2), E) + 1.e3 * sgn(cov_F(2)) * std::numeric_limits<Real>::min();
    E = 1.0;

    // Get the basis vectors for the search
    Vec con_tilg, con_tild;
    GetBasisVectors(cov_F, &con_tilg, &con_tild);

    int iter = 0;
    Real fXi = 1.e3;
    Real fPhi = 1.e3;
    Real Jac[2][2], f[2], dx[2];
    Real err_phi, err_xi;
    Real xi = xi_guess;
    Real phi = phi_guess;
    phi = acos(-1.0);
    do {
      Real fPhi_up, fPhi_lo;
      Real fXi_up, fXi_lo;

      const Real xi_up = std::min(xi * (1.0 + eps) + delta, 1.0);
      const Real xi_lo = std::max(xi * (1.0 - eps) - delta, 0.0);
      M1Residuals(E, cov_F, xi_up, phi, con_tilg, con_tild, &fXi_up, &fPhi_up);
      M1Residuals(E, cov_F, xi_lo, phi, con_tilg, con_tild, &fXi_lo, &fPhi_lo);
      Jac[0][0] = (fXi_up - fXi_lo) / (xi_up - xi_lo);
      Jac[1][0] = (fPhi_up - fPhi_lo) / (xi_up - xi_lo);

      const Real phi_up = phi * (1.0 + eps) + delta;
      const Real phi_lo = phi * (1.0 - eps) - delta;
      M1Residuals(E, cov_F, xi, phi_up, con_tilg, con_tild, &fXi_up, &fPhi_up);
      M1Residuals(E, cov_F, xi, phi_lo, con_tilg, con_tild, &fXi_lo, &fPhi_lo);
      Jac[0][1] = (fXi_up - fXi_lo) / (phi_up - phi_lo);
      Jac[1][1] = (fPhi_up - fPhi_lo) / (phi_up - phi_lo);

      M1Residuals(E, cov_F, xi, phi, con_tilg, con_tild, &fXi, &fPhi);
      f[0] = fXi;
      f[1] = fPhi;

      SolveAxb2x2(Jac, f, dx);

      // Keep phi fixed for the first two updates
      if (iter < max_iter) {
        dx[0] = fXi / Jac[0][0];
        dx[1] = 0.0;
        fPhi = 0.0;
      }

      Real corrected_delta_xi = std::min(dx[0], xi - delta * xi);
      corrected_delta_xi = std::max(corrected_delta_xi, xi - (1 - delta * (1 - xi)));
      Real alpha = ratio(corrected_delta_xi, dx[0]);
      alpha = std::min(alpha, 0.5 * 3.14159 / fabs(dx[1]));
      xi -= alpha * dx[0];
      phi -= alpha * dx[1];

      err_xi = std::abs(dx[0]) / (std::abs(xi) + delta);
      err_phi = std::abs(dx[1]) / (std::abs(phi) + delta);
      ++iter;

    } while (iter < max_iter && (fabs(fXi) > tol || fabs(fPhi) > tol));

    *xi_out = xi;
    *phi_out = phi;
    M1Result result{Status::success, iter, xi, phi, fXi, fPhi, err_xi, err_phi};
    if (std::isnan(xi) || iter>=max_iter) {
      if (!(xi < 1.e-4) && !(std::fabs(fXi) < 1.e-3)) {
            printf("Con2Prim (Fail) : E = %e F = (%e, %e, %e) \n"
                   "                 xi = %e phi = %e fXi = %e fPhi = %e v = (%e, %e, %e) xig = %e phig = %e\n", 
                   E, cov_F(0), cov_F(1), cov_F(2), xi, phi, fXi, fPhi, 
                   con_v(0), con_v(1), con_v(2), xi_guess, phi_guess);
      }
      result.status = Status::failure;
    }
    return result;
  }

  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  Status ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::M1Residuals(const Real E, const Vec cov_F,
                                          const Real xi, const Real phi,
                                          const Vec con_tilg, const Vec con_tild,
                                          Real *fXi, Real *fPhi) {
    // Calculate rest frame fluid variables
    Tens2 con_tilPi;
    Vec con_tilf;
    M1FluidPressureTensor(cov_F, xi, phi, con_tilg, con_tild, &con_tilPi, &con_tilf);
    Real J;
    Vec cov_tilH;
    Con2Prim(E, cov_F, con_tilPi, &J, &cov_tilH);

    // Construct the residual functions
    Real H(0.0), vTilH(0.0), Hf(0.0), vTilf(0.0);
    SPACELOOP2(i,j){ H += cov_tilH(i) * cov_tilH(j) * gamma->con_gamma(i, j); }
    SPACELOOP(i) vTilH += con_v(i) * cov_tilH(i);
    SPACELOOP(i) Hf += cov_tilH(i) * con_tilf(i);
    SPACELOOP(i) vTilf += cov_v(i) * con_tilf(i);
    Hf -= vTilf * vTilH;
    H = std::sqrt(H - vTilH * vTilH);
    *fXi = xi - ratio(H, J);
    *fPhi = Hf - H;

    return Status::success;
  }

  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  Status ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::M1FluidPressureTensor(const Real J, const Vec cov_tilH,
                                                    Tens2 *con_tilPi, Vec *con_tilf) {
    Vec con_tilH;
    gamma->raise3Vector(cov_tilH, &con_tilH);
    Real H(0.0), vH(0.0);
    SPACELOOP(i) H += cov_tilH(i) * con_tilH(i);
    SPACELOOP(i) vH += cov_tilH(i) * con_v(i);
    H = std::sqrt(H - vH * vH);
    SPACELOOP(i) (*con_tilf)(i) = ratio(con_tilH(i), H);
    const Real xi = std::max(std::min(1.0, ratio(H, J)), 0.0);
    const Real athin = 0.5 * (3 * closure(xi) - 1);
    // Calculate the projected rest frame radiation pressure tensor
    SPACELOOP2(i,j) {
      (*con_tilPi)(i, j) = (*con_tilf)(i) * (*con_tilf)(j) - (W2 * con_v(i) * con_v(j) + gamma->con_gamma(i, j)) / 3;
      (*con_tilPi)(i, j) *= athin;
    }

    return Status::success;
  }

  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  Status ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::M1FluidPressureTensor(const Vec /*cov_F*/,
                                                    const Real xi, const Real phi,
                                                    const Vec con_tilg, const Vec con_tild,
                                                    Tens2 *con_tilPi, Vec *con_tilf) {

    // Build projected flux direction from basis vectors and given angle
    const Real s = sin(phi);
    const Real c = cos(phi);
    SPACELOOP(i) (*con_tilf)(i) = -c * con_tilg(i) + s * con_tild(i);

    // Calculate the closure interpolation
    const Real athin = 0.5 * (3 * closure(xi) - 1);

    // Calculate the projected rest frame radiation pressure tensor
    SPACELOOP2(i,j)
    {
      (*con_tilPi)(i, j) = (*con_tilf)(i) * (*con_tilf)(j) - (W2 * con_v(i) * con_v(j) + gamma->con_gamma(i, j)) / 3;
      (*con_tilPi)(i, j) *= athin;
    }

    return Status::success;
  }

  template <class Vec, class Tens2, bool ENERGY_CONSERVE>
  KOKKOS_FUNCTION
  Status ClosureM1<Vec, Tens2, ENERGY_CONSERVE>::GetBasisVectors(const Vec cov_F,
                                              Vec *con_tilg, Vec *con_tild) {
    // Build projected basis vectors for flux direction
    // These vectors are actually fixed, so there is no need to recalculate
    // them every step in the root find 
    Vec con_F;
    gamma->raise3Vector(cov_F, &con_F);
    Real Fmag(0.0), vl(0.0), v2(0.0);
    SPACELOOP(i) Fmag += cov_F(i) * con_F(i);
    Fmag = std::sqrt(Fmag);
    SPACELOOP(i) vl += cov_F(i) * con_v(i);
    vl = ratio(vl, Fmag);
    SPACELOOP(i) v2 += cov_v(i) * con_v(i);
    const Real lam = ratio(1.0, W * (1 - vl));

    // This goes to zero if F and v lie along one another and can create problems, probably should be
    // checking elsewhere that they are not along one another
    const Real aa = std::sqrt(v2 * (1 + 10 * std::numeric_limits<Real>::epsilon()) - vl * vl);
    SPACELOOP(i) {
      (*con_tilg)(i) = ratio(con_F(i) * lam, Fmag) - W * con_v(i);
      (*con_tild)(i) = ratio(ratio(con_F(i), Fmag) * (1 - lam / W) + con_v(i), aa);
    }
    //if (aa < 1.e-6) (*con_tild) = {{0,0,0}};

    return Status::success;
  }
} // namespace radiation

#endif // CLOSURE_HPP_
