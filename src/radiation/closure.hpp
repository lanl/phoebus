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

#ifndef CLOSURE_HPP_
#define CLOSURE_HPP_

#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/robust.hpp"
#include "radiation/local_three_geometry.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

//namespace radiation
//{

  /*using namespace LinearAlgebra;
  using namespace robust;

  /// Status of closure calculation
  enum class ClosureStatus
  {
    success = 0,
    failure = 1
  };

  enum class ClosureType {Eddington, M1, MOCMC};

  // enums and struct for specifying behavior of closure class
  enum class ClosureEquation {energy_conserve, number_conserve};
  enum class ClosureVerbosity {quiet, v1, v2};

  template <ClosureEquation EQ = ClosureEquation::energy_conserve,
            ClosureVerbosity VB = ClosureVerbosity::quiet>
  struct ClosureSettings {
    static const ClosureEquation eqn_type = EQ;
    static const ClosureVerbosity verbosity = VB;
  };

  /// Holds methods for closing the radiation moment equations as well as calculating radiation
  /// moment source terms.
  template <class Vec, class Tens2, class SET = ClosureSettings<> >
  class ClosureEdd
  {
  public:

    using LocalGeometryType = LocalThreeGeometry;

    //-------------------------------------------------------------------------------------
    /// Constructor just calculates the inverse 3-metric, covariant three-velocity, and the
    /// Lorentz factor for the given background state.
    KOKKOS_FUNCTION
    ClosureEdd(const Vec con_v_in, LocalGeometryType* g);

    //-------------------------------------------------------------------------------------
    /// Calculate the update values dE and cov_dF for a linear, implicit source term update
    /// from the starred state using the optical depths tauJ = lapse kappa_a dt and
    /// tauH = lapse (kappa_a + kappa_s) dt
    KOKKOS_FUNCTION
    ClosureStatus LinearSourceUpdate(const Real Estar, const Vec cov_Fstar,
                              const Tens2 con_tilPi, const Real JBB,
                              const Real tauJ, const Real tauH,
                              Real *dE, Vec *cov_dF);

    //-------------------------------------------------------------------------------------
    /// Calculate the fluxes for the conservation equations from J, \tilde H_i, and
    /// \tilde pi^{ij}.
    KOKKOS_FUNCTION
    ClosureStatus getFluxesFromPrim(const Real J, const Vec cov_tilH, const Tens2 con_tilPi,
                             Vec *con_F, Tens2 *concov_P);

    //-------------------------------------------------------------------------------------
    /// Calculate the momentum density flux P^i_j from J, \tilde H_i, and \tilde pi^{ij}.
    KOKKOS_FUNCTION
    ClosureStatus getConCovPFromPrim(const Real J, const Vec cov_tilH, const Tens2 con_tilPi,
                              Tens2 *concov_P);

    //-------------------------------------------------------------------------------------
    /// Calculate the momentum density flux P^{ij} from J, \tilde H_i, and \tilde pi^{ij}.
    KOKKOS_FUNCTION
    ClosureStatus getConPFromPrim(const Real J, const Vec cov_tilH, const Tens2 con_tilPi,
                           Tens2 *con_P);

    //-------------------------------------------------------------------------------------
    /// Transform from J, \tilde H_i, and \tilde \pi_{ij} to E and F_i. Should be used for
    /// Eddington approximation (i.e. \tilde \pi_{ij} = 0) and MoCMC where \tilde \pi^{ij}
    /// is externally specified
    KOKKOS_FUNCTION
    ClosureStatus Prim2Con(const Real J, const Vec cov_H, const Tens2 con_tilPi, Real *E, Vec *cov_F);

    //-------------------------------------------------------------------------------------
    /// Transform from E, F_i, and \tilde \pi_{ij} to J and \tilde H_i. Should be used for
    /// Eddington approximation (i.e. \tilde \pi_{ij} = 0) and MoCMC where \tilde \pi^{ij}
    /// is externally specified
    KOKKOS_FUNCTION
    ClosureStatus Con2Prim(Real E, const Vec cov_F, const Tens2 con_tilPi, Real *J, Vec *cov_H);

    KOKKOS_FUNCTION
    ClosureStatus GetCovTilPiFromPrim(const Real J, const Vec cov_tilH, Tens2* con_tilPi) {
      SPACELOOP2(i,j) (*con_tilPi)(i,j) = 0.0;
      return ClosureStatus::success;
    }

    KOKKOS_FUNCTION
    ClosureStatus GetCovTilPiFromCon(const Real E, const Vec cov_F, Real& xi, Real& phi, Tens2* con_tilPi) {
      SPACELOOP2(i,j) (*con_tilPi)(i,j) = 0.0;
      xi = 0.0;
      phi = acos(-1);
      return ClosureStatus::success;
    }

    Real v2, W, W2;
    Vec cov_v;
    Vec con_v;
    LocalGeometryType* gamma;

  protected:

    KOKKOS_FORCEINLINE_FUNCTION
    void GetTilPiContractions(const Tens2 &con_tilPi, Vec *cov_vTilPi, Real *vvTilPi) {
      Vec con_vTilPi;
      *vvTilPi = 0.0;
      SPACELOOP(i) {
        con_vTilPi(i) = 0.0;
        SPACELOOP(j) {
          con_vTilPi(i) += cov_v(j) * con_tilPi(i, j);
        }
        *vvTilPi += cov_v(i) * con_vTilPi(i);
      }
      gamma->lower3Vector(con_vTilPi, cov_vTilPi);
    }*/

namespace radiation {

using namespace LinearAlgebra;
using namespace robust;

/// Status of closure calculation
enum class ClosureStatus { success = 0, failure = 1 };

enum class ClosureType { Eddington, M1, MOCMC };

// enums and struct for specifying behavior of closure class
enum class ClosureEquation { energy_conserve, number_conserve };
enum class ClosureVerbosity { quiet, v1, v2 };

template <ClosureEquation EQ = ClosureEquation::energy_conserve,
          ClosureVerbosity VB = ClosureVerbosity::quiet>
struct ClosureSettings {
  static const ClosureEquation eqn_type = EQ;
  static const ClosureVerbosity verbosity = VB;
};

/// Holds methods for closing the radiation moment equations as well as calculating
/// radiation moment source terms.
template <class Vec, class Tens2, class SET = ClosureSettings<>>
class ClosureEdd {
 public:
  using LocalGeometryType = LocalThreeGeometry;

  //-------------------------------------------------------------------------------------
  /// Constructor just calculates the inverse 3-metric, covariant three-velocity, and the
  /// Lorentz factor for the given background state.
  KOKKOS_FUNCTION
/*<<<<<<< HEAD
  ClosureEdd<Vec, Tens2, SET>::ClosureEdd(const Vec con_v_in, LocalGeometryType* g)
      : gamma(g)
  {
    SPACELOOP(i) con_v(i) = con_v_in(i);

    gamma->lower3Vector(con_v, &cov_v);
    v2 = 0.0;
    SPACELOOP(i) v2 += con_v(i) * cov_v(i);
    W = 1 / std::sqrt(1 - v2);
    W2 = W * W;
  }

  template <class Vec, class Tens2, class SET>
  KOKKOS_FUNCTION
  ClosureStatus ClosureEdd<Vec, Tens2, SET>::LinearSourceUpdate(const Real Estar, const Vec cov_Fstar,
                                                 const Tens2 con_tilPi, const Real JBB,
                                                 const Real tauJ, const Real tauH,
                                                 Real *dE, Vec *cov_dF) {

    Real vvTilPi;
    Vec cov_vTilPi;
    GetTilPiContractions(con_tilPi, &cov_vTilPi, &vvTilPi);

    Real A[2][2], x[2], b[2];
    A[0][0] = (4*W2-1)/3 + W2*vvTilPi + W*tauJ;
    A[0][1] = 2*W + tauH;
    A[1][0] = A[0][0] - 1 - tauJ/W;
    A[1][1] = A[0][1] - 1/W;

    if (SET::eqn_type == ClosureEquation::number_conserve) {
      A[0][0] = W + tauJ;
      A[0][1] = 1.0;
    }

    Real vFstar = gamma->contractConCov3Vectors(con_v, cov_Fstar);

    if (SET::eqn_type == ClosureEquation::energy_conserve) {
      b[0] = Estar + W*tauJ*JBB;
    }
    else if (SET::eqn_type == ClosureEquation::number_conserve) {
      b[0] = Estar + tauJ*JBB;
    }
    b[1] = vFstar + (W2-1)/W*tauJ*JBB;

    SolveAxb2x2(A, b, x);
    //if (std::isnan(b[0])) throw 1;

    Real &J = x[0];
    Real &zeta = x[1]; //v^i \tilde H_i

    if (SET::eqn_type == ClosureEquation::energy_conserve) {
      *dE = (4*W2 - 1 + 3*W2*vvTilPi) / 3 * J + 2*W*zeta - Estar;
    }
    else if (SET::eqn_type == ClosureEquation::number_conserve) {
      *dE = W*J + zeta - Estar;
    }
    SPACELOOP(i) (*cov_dF)(i) = (cov_v(i)*tauH*(4*W2/3*J + W*zeta) + tauH*cov_vTilPi(i)*J
                   + tauJ*W2*cov_v(i)*(JBB-J) + W*cov_Fstar(i))/(W+tauH) - cov_Fstar(i);

    return ClosureStatus::success;
  }
=======*/
  ClosureEdd(const Vec con_v_in, LocalGeometryType *g);

  //-------------------------------------------------------------------------------------
  /// Calculate the update values dE and cov_dF for a linear, implicit source term update
  /// from the starred state using the optical depths tauJ = lapse kappa_a dt and
  /// tauH = lapse (kappa_a + kappa_s) dt
  KOKKOS_FUNCTION
  ClosureStatus LinearSourceUpdate(const Real Estar, const Vec cov_Fstar,
                                   const Tens2 con_tilPi, const Real JBB, const Real tauJ,
                                   const Real tauH, Real *dE, Vec *cov_dF);

  //-------------------------------------------------------------------------------------
  /// Calculate the fluxes for the conservation equations from J, \tilde H_i, and
  /// \tilde pi^{ij}.
  KOKKOS_FUNCTION
  ClosureStatus getFluxesFromPrim(const Real J, const Vec cov_tilH, const Tens2 con_tilPi,
                                  Vec *con_F, Tens2 *concov_P);
//>>>>>>> origin/main

  //-------------------------------------------------------------------------------------
  /// Calculate the momentum density flux P^i_j from J, \tilde H_i, and \tilde pi^{ij}.
  /*KOKKOS_FUNCTION
  ClosureStatus ClosureEdd<Vec, Tens2, SET>::getFluxesFromPrim(const Real J, const Vec cov_tilH,
                                                                 const Tens2 con_tilPi, Vec *con_F,
                                                                 Tens2 *concov_P) {
    getConCovPFromPrim(J, cov_tilH, con_tilPi, concov_P);
    if (SET::eqn_type == ClosureEquation::energy_conserve) {
      Real E;
      Vec cov_F;
      Prim2Con(J, cov_tilH, con_tilPi, &E, &cov_F);
      gamma->raise3Vector(cov_F, con_F);
    }
    else if (SET::eqn_type == ClosureEquation::number_conserve) {
      SPACELOOP(i) (*con_F)(i) = W*con_v(i)*J + cov_tilH(i);
    }
    SPACELOOP2(i,j) (*concov_P)(i,j) *= gamma->alpha;
    SPACELOOP(i) (*con_F)(i) *= gamma->alpha;
    return ClosureStatus::success;
  }

  template <class Vec, class Tens2, class SET>
  KOKKOS_FORCEINLINE_FUNCTION
  ClosureStatus ClosureEdd<Vec, Tens2, SET>::getConPFromPrim(const Real J, const Vec cov_tilH,
                                              const Tens2 con_tilPi, Tens2 *con_P) {
    Vec con_tilH;
    gamma->raise3Vector(cov_tilH, &con_tilH);
    SPACELOOP2(i,j) (*con_P)(i, j) = 4.0 / 3.0 * W2 * con_v(i) * con_v(j) * J
            + W * (con_v(i) * con_tilH(j) + con_v(j) * con_tilH(i)) + J * con_tilPi(i, j)
            + J/3.0*gamma->con_gamma(i,j);*/

  ClosureStatus getConCovPFromPrim(const Real J, const Vec cov_tilH,
                                   const Tens2 con_tilPi, Tens2 *concov_P);

  //-------------------------------------------------------------------------------------
  /// Calculate the momentum density flux P^{ij} from J, \tilde H_i, and \tilde pi^{ij}.
  KOKKOS_FUNCTION
  ClosureStatus getConPFromPrim(const Real J, const Vec cov_tilH, const Tens2 con_tilPi,
                                Tens2 *con_P);

  //-------------------------------------------------------------------------------------
  /// Transform from J, \tilde H_i, and \tilde \pi_{ij} to E and F_i. Should be used for
  /// Eddington approximation (i.e. \tilde \pi_{ij} = 0) and MoCMC where \tilde \pi^{ij}
  /// is externally specified
  KOKKOS_FUNCTION
  ClosureStatus Prim2Con(const Real J, const Vec cov_H, const Tens2 con_tilPi, Real *E,
                         Vec *cov_F);

  //-------------------------------------------------------------------------------------
  /// Transform from E, F_i, and \tilde \pi_{ij} to J and \tilde H_i. Should be used for
  /// Eddington approximation (i.e. \tilde \pi_{ij} = 0) and MoCMC where \tilde \pi^{ij}
  /// is externally specified
  KOKKOS_FUNCTION
  ClosureStatus Con2Prim(Real E, const Vec cov_F, const Tens2 con_tilPi, Real *J,
                         Vec *cov_H);

  KOKKOS_FUNCTION
  ClosureStatus GetCovTilPiFromPrim(const Real J, const Vec cov_tilH, Tens2 *con_tilPi) {
    SPACELOOP2(i, j) (*con_tilPi)(i, j) = 0.0;
    return ClosureStatus::success;
  }

  KOKKOS_FUNCTION
  ClosureStatus GetCovTilPiFromCon(const Real E, const Vec cov_F, Real &xi, Real &phi,
                                   Tens2 *con_tilPi) {
    SPACELOOP2(i, j) (*con_tilPi)(i, j) = 0.0;
    xi = 0.0;
    phi = acos(-1);
//>>>>>>> origin/main
    return ClosureStatus::success;
  }

  Real v2, W, W2;
  Vec cov_v;
  Vec con_v;
  LocalGeometryType *gamma;

 protected:
  KOKKOS_FORCEINLINE_FUNCTION
  void GetTilPiContractions(const Tens2 &con_tilPi, Vec *cov_vTilPi, Real *vvTilPi) {
    Vec con_vTilPi;
    *vvTilPi = 0.0;
    SPACELOOP(i) {
//<<<<<<< HEAD
//      SPACELOOP(j) {
//        (*concov_P)(i, j) = 4.0 / 3.0 * W2 * con_v(i) * cov_v(j) * J
//            + W * (con_v(i) * cov_tilH(j) + cov_v(j) * con_tilH(i)) + J * concov_tilPi(i, j);
//      }
//      (*concov_P)(i, i) += J / 3.0;
//=======
      con_vTilPi(i) = 0.0;
      SPACELOOP(j) { con_vTilPi(i) += cov_v(j) * con_tilPi(i, j); }
      *vvTilPi += cov_v(i) * con_vTilPi(i);
//>>>>>>> origin/main
    }
    gamma->lower3Vector(con_vTilPi, cov_vTilPi);
  }

//<<<<<<< HEAD
  /*template <class Vec, class Tens2, class SET>
  KOKKOS_FUNCTION
  ClosureStatus ClosureEdd<Vec, Tens2, SET>::Prim2Con(const Real J, const Vec cov_H,
                                       const Tens2 con_tilPi, Real *E, Vec *cov_F) {
    Real vvPi;
    Vec cov_vPi;
    GetTilPiContractions(con_tilPi, &cov_vPi, &vvPi);

    Real vH = 0.0;
    SPACELOOP(i) vH += con_v(i) * cov_H(i);
    if (SET::eqn_type == ClosureEquation::energy_conserve) {
      *E = (4 * W2 - 1 + 3 * W2 * vvPi) / 3 * J + 2 * W * vH;
    }
    else if (SET::eqn_type == ClosureEquation::number_conserve) {
      *E = W*J + vH;
    }
    SPACELOOP(i) (*cov_F)(i) = 4 * W2 / 3 * cov_v(i) * J + W * cov_v(i) * vH + W * cov_H(i) + J * cov_vPi(i);
    return ClosureStatus::success;
  }

  template <class Vec, class Tens2, class SET>
  KOKKOS_FUNCTION
  ClosureStatus ClosureEdd<Vec, Tens2, SET>::Con2Prim(Real E, const Vec cov_F,
                                       const Tens2 con_tilPi,
                                       Real *J, Vec *cov_tilH) {
    printf("c2p: E: %e covF: %e %e %e\n", E, cov_F(0), cov_F(1), cov_F(2));
    Real vvTilPi;
    Vec cov_vTilPi;
    GetTilPiContractions(con_tilPi, &cov_vTilPi, &vvTilPi);

    double vF = 0.0;
    SPACELOOP(i) vF += con_v(i) * cov_F(i);

    if (SET::eqn_type == ClosureEquation::number_conserve) E = E/W + vF;

    // lam is proportional to the determinant of the 2x2 linear system relating
    // E and v_i F^i to J and v_i H^i for fixed tilde pi^ij
    const Real lam = (2.0 * W2 + 1.0) / 3.0 + (2.0 * W2 - 3.0) * W2 * vvTilPi;

    // zeta = v_i H^i
    // const Real zeta = -W/lam*(((4*W2-4)/3 + vvPi)*E
    //                   -((4*W2-1)/3 + W2*vvPi)*vF);
    // a = 4 W^2/3 J + W zeta
    const Real a = ratio(W2, lam) * ((4 * W2 / 3 - vvTilPi) * E - ((4 * W2 + 1) / 3 - W2 * vvTilPi) * vF);

    // Calculate fluid rest frame (i.e. primitive) quantities
    *J = ratio((2 * W2 - 1) * E - 2 * W2 * vF, lam);
    SPACELOOP(i) (*cov_tilH)(i) = (cov_F(i) - (*J) * cov_vTilPi(i) - cov_v(i) * a) / W;
    return ClosureStatus::success;
//=======
*/
  private:
};

template <class Vec, class Tens2, class SET>
KOKKOS_FUNCTION ClosureEdd<Vec, Tens2, SET>::ClosureEdd(const Vec con_v_in,
                                                        LocalGeometryType *g)
    : gamma(g) {
  SPACELOOP(i) con_v(i) = con_v_in(i);

  gamma->lower3Vector(con_v, &cov_v);
  v2 = 0.0;
  SPACELOOP(i) v2 += con_v(i) * cov_v(i);
  W = 1 / std::sqrt(1 - v2);
  W2 = W * W;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::LinearSourceUpdate(
    const Real Estar, const Vec cov_Fstar, const Tens2 con_tilPi, const Real JBB,
    const Real tauJ, const Real tauH, Real *dE, Vec *cov_dF) {

  Real vvTilPi;
  Vec cov_vTilPi;
  GetTilPiContractions(con_tilPi, &cov_vTilPi, &vvTilPi);

  Real A[2][2], x[2], b[2];
  A[0][0] = (4 * W2 - 1) / 3 + W2 * vvTilPi + W * tauJ;
  A[0][1] = 2 * W + tauH;
  A[1][0] = A[0][0] - 1 - tauJ / W;
  A[1][1] = A[0][1] - 1 / W;

  if (SET::eqn_type == ClosureEquation::number_conserve) {
    A[0][0] = W + tauJ;
    A[0][1] = 1.0;
  }

  Real vFstar = gamma->contractConCov3Vectors(con_v, cov_Fstar);

  if (SET::eqn_type == ClosureEquation::energy_conserve) {
    b[0] = Estar + W * tauJ * JBB;
  } else if (SET::eqn_type == ClosureEquation::number_conserve) {
    b[0] = Estar + tauJ * JBB;
  }
  b[1] = vFstar + (W2 - 1) / W * tauJ * JBB;

  SolveAxb2x2(A, b, x);
  // if (std::isnan(b[0])) throw 1;

  Real &J = x[0];
  Real &zeta = x[1]; // v^i \tilde H_i

  if (SET::eqn_type == ClosureEquation::energy_conserve) {
    *dE = (4 * W2 - 1 + 3 * W2 * vvTilPi) / 3 * J + 2 * W * zeta - Estar;
  } else if (SET::eqn_type == ClosureEquation::number_conserve) {
    *dE = W * J + zeta - Estar;
  }
  SPACELOOP(i)
  (*cov_dF)(i) =
      (cov_v(i) * tauH * (4 * W2 / 3 * J + W * zeta) + tauH * cov_vTilPi(i) * J +
       tauJ * W2 * cov_v(i) * (JBB - J) + W * cov_Fstar(i)) /
          (W + tauH) -
      cov_Fstar(i);

  return ClosureStatus::success;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::getFluxesFromPrim(
    const Real J, const Vec cov_tilH, const Tens2 con_tilPi, Vec *con_F,
    Tens2 *concov_P) {
  getConCovPFromPrim(J, cov_tilH, con_tilPi, concov_P);
  if (SET::eqn_type == ClosureEquation::energy_conserve) {
    Real E;
    Vec cov_F;
    Prim2Con(J, cov_tilH, con_tilPi, &E, &cov_F);
    gamma->raise3Vector(cov_F, con_F);
  } else if (SET::eqn_type == ClosureEquation::number_conserve) {
    SPACELOOP(i) (*con_F)(i) = W * con_v(i) * J + cov_tilH(i);
  }
  SPACELOOP2(i, j) (*concov_P)(i, j) *= gamma->alpha;
  SPACELOOP(i) (*con_F)(i) *= gamma->alpha;
  return ClosureStatus::success;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FORCEINLINE_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::getConPFromPrim(
    const Real J, const Vec cov_tilH, const Tens2 con_tilPi, Tens2 *con_P) {
  Vec con_tilH;
  gamma->raise3Vector(cov_tilH, &con_tilH);
  SPACELOOP2(i, j)
  (*con_P)(i, j) = 4.0 / 3.0 * W2 * con_v(i) * con_v(j) * J +
                   W * (con_v(i) * con_tilH(j) + con_v(j) * con_tilH(i)) +
                   J * con_tilPi(i, j) + J / 3.0 * gamma->con_gamma(i, j);

  return ClosureStatus::success;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FORCEINLINE_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::getConCovPFromPrim(
    const Real J, const Vec cov_tilH, const Tens2 con_tilPi, Tens2 *concov_P) {
  Vec con_tilH;
  Tens2 concov_tilPi;
  SPACELOOP2(i, j) {
    concov_tilPi(i, j) = 0.0;
    SPACELOOP(k) concov_tilPi(i, j) += con_tilPi(i, k) * gamma->cov_gamma(k, j);
  }
  gamma->raise3Vector(cov_tilH, &con_tilH);
  SPACELOOP(i) {
    SPACELOOP(j) {
      (*concov_P)(i, j) = 4.0 / 3.0 * W2 * con_v(i) * cov_v(j) * J +
                          W * (con_v(i) * cov_tilH(j) + cov_v(j) * con_tilH(i)) +
                          J * concov_tilPi(i, j);
    }
    (*concov_P)(i, i) += J / 3.0;
  }
  return ClosureStatus::success;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::Prim2Con(const Real J,
                                                                    const Vec cov_H,
                                                                    const Tens2 con_tilPi,
                                                                    Real *E, Vec *cov_F) {
  Real vvPi;
  Vec cov_vPi;
  GetTilPiContractions(con_tilPi, &cov_vPi, &vvPi);

  Real vH = 0.0;
  SPACELOOP(i) vH += con_v(i) * cov_H(i);
  if (SET::eqn_type == ClosureEquation::energy_conserve) {
    *E = (4 * W2 - 1 + 3 * W2 * vvPi) / 3 * J + 2 * W * vH;
  } else if (SET::eqn_type == ClosureEquation::number_conserve) {
    *E = W * J + vH;
//>>>>>>> origin/main
  }
  SPACELOOP(i)
  (*cov_F)(i) =
      4 * W2 / 3 * cov_v(i) * J + W * cov_v(i) * vH + W * cov_H(i) + J * cov_vPi(i);
  return ClosureStatus::success;
}

template <class Vec, class Tens2, class SET>
KOKKOS_FUNCTION ClosureStatus ClosureEdd<Vec, Tens2, SET>::Con2Prim(
    Real E, const Vec cov_F, const Tens2 con_tilPi, Real *J, Vec *cov_tilH) {
  Real vvTilPi;
  Vec cov_vTilPi;
  GetTilPiContractions(con_tilPi, &cov_vTilPi, &vvTilPi);

  double vF = 0.0;
  SPACELOOP(i) vF += con_v(i) * cov_F(i);

  if (SET::eqn_type == ClosureEquation::number_conserve) E = E / W + vF;

  // lam is proportional to the determinant of the 2x2 linear system relating
  // E and v_i F^i to J and v_i H^i for fixed tilde pi^ij
  const Real lam = (2.0 * W2 + 1.0) / 3.0 + (2.0 * W2 - 3.0) * W2 * vvTilPi;

  // zeta = v_i H^i
  // const Real zeta = -W/lam*(((4*W2-4)/3 + vvPi)*E
  //                   -((4*W2-1)/3 + W2*vvPi)*vF);
  // a = 4 W^2/3 J + W zeta
  const Real a = ratio(W2, lam) *
                 ((4 * W2 / 3 - vvTilPi) * E - ((4 * W2 + 1) / 3 - W2 * vvTilPi) * vF);

  // Calculate fluid rest frame (i.e. primitive) quantities
  *J = ratio((2 * W2 - 1) * E - 2 * W2 * vF, lam);
  SPACELOOP(i) (*cov_tilH)(i) = (cov_F(i) - (*J) * cov_vTilPi(i) - cov_v(i) * a) / W;
  return ClosureStatus::success;
}

} // namespace radiation

#endif // CLOSURE_HPP_
