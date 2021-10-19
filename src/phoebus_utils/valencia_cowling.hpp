// Valencia formulation in Cowling approximation, for debugging. See BHAC code paper for details.
// Note that in phoebus we don't have beta^i_,j

#pragma once

#include "geometry/geometry.hpp"

class ValenciaCowling {

public:
  KOKKOS_INLINE_FUNCTION
  ValenciaCowling(const Real alpha_, const Real betacon_[3], const Real gammacov_[3][3],
                  const Real gammacon_[3][3], const Real dgcov_[4][4][4],
                  const Real gradlnalpha_[4], const Real rho_, const Real u_, const Real P_,
                  const Real vcon_[3]) {
    alpha = alpha_;
    rho = rho_;
    u = u_;
    P = P_;
    SPACELOOP(ii) {
      betacon[ii] = betacon_[ii];
      vcon[ii] = vcon_[ii];
      SPACELOOP(jj) {
        gammacov[ii][jj] = gammacov_[ii][jj];
        gammacon[ii][jj] = gammacon_[ii][jj];
      }
    }
    SPACETIMELOOP(mu) {
      gradlnalpha[mu] = gradlnalpha_[mu];
      SPACETIMELOOP2(nu, lam) {
        dgcov[mu][nu][lam] = dgcov_[mu][nu][lam];
      }
    }

    SPACELOOP(ii) {
      vcov[ii] = 0.;
      vtildecon[ii] = alpha*vcon[ii] - betacon[ii];
    }

    vsq = 0.;
    SPACELOOP2(ii, jj) {
      vsq += gammacov[ii][jj]*vcon[ii]*vcon[jj];
      vcov[ii] += gammacov[ii][jj]*vcon[jj];
    }
    Gamma = 1./(sqrt(1. - vsq));

    h = 1. + u/rho + P/rho;

    D = rho*Gamma;
    SPACELOOP(ii) {
      Scov[ii] = rho*h*Gamma*Gamma*vcov[ii];
      Scon[ii] = 0.;
    }
    SPACELOOP2(ii, jj) {
      Scon[ii] += gammacon[ii][jj]*Scov[jj];
      Wconcon[ii][jj] = 0.;
      Wconcov[ii][jj] = 0.;
    }

    SPACELOOP2(ii, jj) {
      Wconcon[ii][jj] = Scon[ii]*vcon[jj] + P*gammacov[ii][jj];
    }

    SPACELOOP3(ii, jj, kk) {
      Wconcov[ii][jj] += Wconcon[ii][kk]*gammacov[kk][jj];
    }

    tau = rho*h*Gamma*Gamma - P - D;

    U[0] = rho*Gamma;
    U[1] = Scov[0];
    U[2] = Scov[1];
    U[3] = Scov[2];
    U[4] = tau;

    for (int dir = 0; dir < 3; dir++) {
      F[dir][0] = D*vtildecon[dir]/alpha;
      SPACELOOP(jj) {
        F[dir][jj+1] = Wconcov[dir][jj] - betacon[dir]*Scov[jj]/alpha;
      }
      F[dir][4] = Scon[dir] - vcon[dir]*D - betacon[dir]*tau/alpha;
    }
  }

  // Geometry
  Real alpha;
  Real betacon[3];
  Real gammacov[3][3];
  Real gammacon[3][3];
  Real dgcov[4][4][4];
  Real gradlnalpha[4];

  // Fluid
  Real rho;
  Real u;
  Real P;
  Real vcon[3];
  Real vcov[3];
  Real vtildecon[3];
  Real vsq;
  Real Gamma;
  Real h;
  Real D;
  Real Scov[3];
  Real Scon[3];
  Real tau;
  Real Wconcon[3][3];
  Real Wconcov[3][3];

  // order: rho mom1 mom2 mom3 ener
  Real U[5];
  // F[dir][rho mom1 mom2 mom3 ener]
  Real F[3][5];
};
