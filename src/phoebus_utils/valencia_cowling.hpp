// Valencia formulation in Cowling approximation, for debugging. See BHAC code paper for details.
// Note that in phoebus we don't have beta^i_,j

#pragma once

#include "geometry/geometry.hpp"

class ValenciaCowling {

public:
  KOKKOS_INLINE_FUNCTION
  ValenciaCowling(const Real alpha_, const Real betacon_[3], const Real gammacov_[3][3],
                  const Real gammacon_[3][3], const Real dgcov_[4][4][4],
                  const Real dlnalpha_[4], const Real rho_, const Real u_, const Real P_,
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
      dlnalpha[mu] = dlnalpha_[mu];
      SPACETIMELOOP2(nu, lam) {
        dgcov[mu][nu][lam] = dgcov_[mu][nu][lam];
      }
    }

    Real dbetacov[3][3] = {0};
    SPACELOOP2(ii, jj) {
      dbetacon[ii][jj] = 0.;
      dbetacov[ii][jj] = dgcov[ii+1][0][jj+1];
    }

    SPACELOOP3(ii, jj, kk) {
      dbetacon[ii][jj] += dbetacov[kk][jj]*gammacon[kk][ii];
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
      Wconcon[ii][jj] = Scon[ii]*vcon[jj] + P*gammacon[ii][jj];
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

    for (int i = 0; i < 5; i++) {
      S[i] = 0.;
    }
    SPACELOOP(jj) {
      SPACELOOP2(ii, kk) {
        S[jj + 1] += 0.5*Wconcon[ii][kk]*dgcov[ii+1][kk+1][jj+1];
      }
      SPACELOOP(ii) {
        S[jj + 1] += Scov[ii]*dbetacon[ii][jj]/alpha;
      }
      S[jj + 1] -= (tau + D)*dlnalpha[jj+1];
    }
    SPACELOOP3(ii, jj, kk) {
      S[4] += 1./(2.*alpha)*Wconcon[ii][kk]*betacon[jj]*dgcov[ii+1][kk+1][jj+1];
    }
    SPACELOOP2(ii, jj) {
      S[4] += Wconcov[jj][ii]*dbetacon[ii][jj]/alpha;
    }
    SPACELOOP(jj) {
      S[4] -= Scon[jj]*dlnalpha[jj+1];
    }
  }

  // Geometry
  Real alpha;
  Real betacon[3];
  Real gammacov[3][3];
  Real gammacon[3][3];
  Real dgcov[4][4][4];
  Real dlnalpha[4];
  Real dbetacon[3][3];

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
  // order: rho mom1 mom2 mom3 ener
  Real S[5];
};
