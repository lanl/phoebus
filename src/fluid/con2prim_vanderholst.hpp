//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef CON2PRIM_VANDERHOLST_HPP_
#define CON2PRIM_VANDERHOLST_HPP_

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <singularity-eos/eos/eos.hpp>

#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"

namespace con2prim_vanderholst {

enum class ConToPrimStatus {success, failure};
struct FailFlags {
  static constexpr Real success = 1.0;
  static constexpr Real fail = 0.0;
};

template <typename F>
KOKKOS_INLINE_FUNCTION
Real find_root_secant(F func, const Real x_guess, const Real tol, const int maxiter, bool &success,
  const Real eps = 1.e-5) {
  int niter = 0;

  Real x0 = x_guess;
  Real x1 = (1. + eps)*x_guess;

  while (fabs(x0 - x1)/fabs(x0) > tol) {
    Real x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0));
    x0 = x1;
    x1 = x2;
    niter++;
    if (niter == maxiter) {
      success = false;
      return x1;
    }
  }

  success = true;
  if (isnan(x1)) {
    success = false;
  }
  return x1;
}

KOKKOS_INLINE_FUNCTION
Real xi_to_Gamma(const Real &xi, const Real &SdB, const Real &Bsq, const Real Scov[3],
                 const Real Bcov[3], const Real gcon[3][3]) {
  Real threev[3] = {0};
  SPACELOOP(ii) {
    threev[ii] = Scov[ii] + 1./xi*SdB*Bcov[ii];
  }
  Real ans = 0.;
  SPACELOOP2(ii, jj) {
    ans += gcon[ii][jj]*threev[ii]*threev[jj];
  }
  ans /= std::pow(xi + Bsq,2);
  ans = 1. - ans;
  return 1./std::sqrt(ans);
}

KOKKOS_INLINE_FUNCTION
Real Gamma_to_sie(const Real &Gamma, const Real &xi, const Real vcon[3], const Real &D, const Real &tau,
                  const Real Scov[3], const Real Bcon[3], const Real gcov4[4][4],
                  const Real gcon[3][3], const Real &Bsq, const Real &SdB) {
    Real rho = D/Gamma;
    Real h = xi/(Gamma*Gamma);
    const Real muK = 1./(h*Gamma);
    Real bKcon[3] = {0};

    const Real sD = 1.0/std::sqrt(D);
    SPACELOOP(ii) {
      bKcon[ii] = Bcon[ii]/sD;
      printf("Bcon[%i]: %e\n", ii, Bcon[ii]);
    }
    Real bKsq = 0.;
    SPACELOOP2(ii, jj) {
      bKsq += gcov4[ii+1][jj+1]*bKcon[ii]*bKcon[jj];
    }
    printf("bKsq: %e\n", bKsq);
    Real rKperpcon[3] = {0.};
    SPACELOOP(ii) {
      SPACELOOP(jj) {
        rKperpcon[ii] -= bKcon[jj]*Scov[jj];
      }
      rKperpcon[ii] *= bKcon[ii]/(bKsq*D + 1.e-100);
      Real Sconii = 0.;
      SPACELOOP(jj) {
        Sconii += gcon[ii][jj]*Scov[jj];
      }
      rKperpcon[ii] += Sconii/D;
      printf("rKperpcon[%i]: %e\n", ii, rKperpcon[ii]);
      printf("[%i] bKcon: %e Scov: %e\n", ii, bKcon[ii], Scov[ii]);
    }
    Real rKperpsq = 0.;
    SPACELOOP2(ii, jj) {
      rKperpsq += gcov4[ii+1][jj+1]*rKperpcon[ii]*rKperpcon[jj];
    }
    Real rbarKsq = 0.;
    SPACELOOP2(ii, jj) {
      rbarKsq += vcon[ii]*vcon[jj]*gcov4[ii][jj];
    }
    rbarKsq *= D*Gamma*Gamma*h*h;
    printf("rbarKsq: %e\n", rbarKsq);

    const Real qK = tau/D;
    const Real xK = 1./(1. + muK*bKsq);
    const Real qKbar = qK - 0.5*bKsq - 0.5*muK*muK*xK*xK*bKsq*rKperpsq;

    printf("qK: %e xK: %e qKbar: %e muK: %e rKperpsq: %e\n", qK, xK, qKbar, muK, rKperpsq);

    return Gamma*(qKbar - muK*rbarKsq) + Gamma - 1.;
}

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real D, const Real Ssq, const Real tau, const Real gamma) :
    D_(D), Ssq_(Ssq), tau_(tau), gamma_(gamma) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real xi) {
    const Real Gamma = sqrt(1./(1. - Ssq_/(xi*xi)));
    return xi - (gamma_ - 1.)/gamma_*(xi/(Gamma*Gamma) - D_/Gamma) - tau_ - D_;
  }

  private:
    const Real D_;
    const Real Ssq_;
    const Real tau_;
    const Real gamma_;
};

template <typename T>
class VarAccessor {
 public:
  KOKKOS_FUNCTION
  VarAccessor(const T &var, const int k, const int j, const int i)
              : var_(var), b_(0), k_(k), j_(j), i_(i) {}
  VarAccessor(const T &var, const int b, const int k, const int j, const int i)
              : var_(var), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int n) const {
    return var_(b_, n, k_, j_, i_);
  }
  const int b_, i_, j_, k_;
 private:
  const T &var_;
};

struct CellGeom {
  template<typename CoordinateSystem>
  KOKKOS_FUNCTION
  CellGeom(const CoordinateSystem &geom,
           const int k, const int j, const int i)
    : gdet(geom.DetGamma(CellLocation::Cent,k,j,i)),
      lapse(geom.Lapse(CellLocation::Cent,k,j,i)) {
    geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
    SPACELOOP2(m,n) {
      gcov[m][n] = gcov4[m+1][n+1];
    }
    geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
  }
  template<typename CoordinateSystem>
  CellGeom(const CoordinateSystem &geom,
           const int b, const int k, const int j, const int i)
    : gdet(geom.DetGamma(CellLocation::Cent,b,k,j,i)),
      lapse(geom.Lapse(CellLocation::Cent,b,k,j,i)) {
    geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, gcov4);
    SPACELOOP2(m,n) {
      gcov[m][n] = gcov4[m+1][n+1];
    }
    geom.MetricInverse(CellLocation::Cent, b, k, j, i, gcon);
    geom.ContravariantShift(CellLocation::Cent, b, k, j, i, beta);
  }
  Real gcov[3][3]; // TODO(BRR) remove
  Real gcov4[4][4];
  Real gcon[3][3];
  Real beta[3];
  const Real gdet;
  const Real lapse;
};

template <typename Data_t, typename T>
class ConToPrim {
 public:
  ConToPrim(Data_t *rc, const Real tol, const int max_iterations)
    : var(rc->PackVariables(Vars(), imap)),
      prho(imap[fluid_prim::density].first),
      crho(imap[fluid_cons::density].first),
      pvel_lo(imap[fluid_prim::velocity].first),
      pvel_hi(imap[fluid_prim::velocity].second),
      cmom_lo(imap[fluid_cons::momentum].first),
      cmom_hi(imap[fluid_cons::momentum].second),
      peng(imap[fluid_prim::energy].first),
      ceng(imap[fluid_cons::energy].first),
      pb_lo(imap[fluid_prim::bfield].first),
      pb_hi(imap[fluid_prim::bfield].second),
      cb_lo(imap[fluid_cons::bfield].first),
      cb_hi(imap[fluid_cons::bfield].second),
      pye(imap[fluid_prim::ye].second),
      cye(imap[fluid_cons::ye].second),
      prs(imap[fluid_prim::pressure].first),
      tmp(imap[fluid_prim::temperature].first),
      sig_lo(imap[internal_variables::cell_signal_speed].first),
      sig_hi(imap[internal_variables::cell_signal_speed].second),
      gm1(imap[fluid_prim::gamma1].first),
      rel_tolerance(tol),
      max_iter(max_iterations),
      h0sq_(1.0) { }

  std::vector<std::string> Vars() {
    return std::vector<std::string>({fluid_prim::density, fluid_cons::density,
                                    fluid_prim::velocity, fluid_cons::momentum,
                                    fluid_prim::energy, fluid_cons::energy,
                                    fluid_prim::bfield, fluid_cons::bfield,
                                    fluid_prim::ye, fluid_cons::ye,
                                    fluid_prim::pressure, fluid_prim::temperature,
                                    internal_variables::cell_signal_speed, fluid_prim::gamma1});
  }

  template <typename CoordinateSystem, class... Args>
  KOKKOS_INLINE_FUNCTION
  ConToPrimStatus Solve(const CoordinateSystem &geom, const singularity::EOS &eos,
                        Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);

    const Real igdet = 1.0/g.gdet;

    // First update magnetic field
    if (pb_hi > 0) {
      // first set the primitive b-fields
      SPACELOOP(ii) {
        v(pb_lo+ii) = v(cb_lo+ii)*igdet;
      }
    }

    // Calculate initial guess
    Real h = 1. + v(peng)/v(prho) +  v(prs)/v(prho);
    Real w = v(prho)*h;
    //Real vsq = 0.;
    Real vel[3] = {v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2)};
    Real Gamma = phoebus::GetLorentzFactor(vel, g.gcov);
    //SPACELOOP2(ii, jj) {
    //  vsq += g.gcov[ii][jj]*vel[ii]*vel[jj];
    //}
    //vsq = (vsq > 0.9996) ? 0.9996 : vsq;
    //Real Gamma = 1./sqrt(1. - vsq);
    Real xi_guess = Gamma*Gamma*w;

    printf("Initial: rho: %e ug: %e P: %e\n", v(prho), v(peng), v(prs));

    //if (v.i_ == 128 && v.j_ == 128) {
    //  printf("h: %e w: %e vel: %e %e %e Gamma: %e\n", h,w,vel[0],vel[1],vel[2],Gamma);
    //}

    if (isnan(xi_guess)) {
      return ConToPrimStatus::failure;
    }

    const Real D = v(crho)*igdet;
    //  printf("D: %e\n", D);
    #if USE_VALENCIA
    const Real tau = v(ceng)*igdet;
    //printf("tau: %e\n", tau);
    #else
    Real Qcov[4] = {(v(ceng) - v(crho))*igdet,
                      v(cmom_lo)*igdet,
                      v(cmom_lo+1)*igdet,
                      v(cmom_lo+2)*igdet};
    Real ncon[4] = {1./g.lapse, -g.beta[0]/g.lapse, -g.beta[1]/g.lapse, -g.beta[2]/g.lapse};
    Real tau = 0.;
    SPACETIMELOOP(mu) {
      tau -= Qcov[mu]*ncon[mu];
    }
    tau -= D;
    #endif // USE_VALENCIA
    Real Scov[3] = {v(cmom_lo)*igdet, v(cmom_lo+1)*igdet, v(cmom_lo+2)*igdet};
    Real Ssq = 0.;
    SPACELOOP2(ii, jj) {
      Ssq += g.gcon[ii][jj]*Scov[ii]*Scov[jj];
    }

    Real Bcon[3] = {0.};
    Real Bcov[3] = {0.};
    Real Bsq = 0.;
    Real SdB = 0.;
    if (pb_hi > 0) {
      SPACELOOP(ii) {
        Bcon[ii] = v(pb_lo+ii);
      }
      SPACELOOP2(ii, jj) {
        Bcov[ii] += g.gcov4[ii+1][jj+1]*Bcon[jj];
      }
      SPACELOOP(ii) {
        Bsq += Bcon[ii]*Bcov[ii];
        SdB += Scov[ii]*Bcon[ii];
      }
    }


    Real gamma = 5./3.;
    Residual res(D,Ssq,tau,gamma);

    bool success;
    const Real xi = find_root_secant(res, xi_guess, rel_tolerance, max_iter, success);

    if (!success) {
      return ConToPrimStatus::failure;
    }

    Gamma = xi_to_Gamma(xi, SdB, Bsq, Scov, Bcov, g.gcon);

    printf("Gamma: %e\n", Gamma);
    {
      Real vcov[3] = {Scov[0]/xi, Scov[1]/xi, Scov[2]/xi};
      Real vcon[3] = {0};
      SPACELOOP2(ii, jj) {
        vcon[ii] += g.gcon[ii][jj]*vcov[jj];
      }
      Real sie = Gamma_to_sie(Gamma, xi, vcon, D, tau, Scov, Bcon, g.gcov4, g.gcon, Bsq, SdB);
      printf("sie: %e ug: %e\n", sie, v(prho)*sie);
    }
    exit(-1);

   // Gamma = sqrt(1./(1. - Ssq/(xi*xi)));

    //if (v.i_ == 128 && v.j_ == 128) {
    //  printf("xi = %e Gamma = %e\n",xi, Gamma);
    //  printf("[%i %i %i]\n", v.i_, v.j_, v.k_);
    //}
    w = xi/(Gamma*Gamma);
    Real rho = D/Gamma;
    Real vcov[3] = {Scov[0]/xi, Scov[1]/xi, Scov[2]/xi};
    Real vcon[3] = {0};
    SPACELOOP2(ii, jj) {
      vcon[ii] += g.gcon[ii][jj]*vcov[jj];
    }

    // Follow Kastaun et al. 2021 to calculate internal energy density
    /*h = w/rho;
    const Real muK = 1./(h*Gamma);
    Real bKcon[3] = {0};

    const Real sD = 1.0/std::sqrt(D);
    SPACELOOP(ii) {
      bKcon[ii] = v(pb_lo+ii)/sD;
    }
    Real bKsq = 0.;
    SPACELOOP2(ii, jj) {
      bKsq += g.gcov4[ii+1][jj+1]*bKcon[ii]*bKcon[jj];
    }
    Real rKperpcon[3] = {0.};
    SPACELOOP(ii) {
      SPACELOOP(jj) {
        rKperpcon[ii] -= bKcon[jj]*S[jj];
      }
      rKperpcon[ii] *= bKcon[ii]/(bKsq*D);
      Real Sconii = 0.;
      SPACELOOP(jj) {
        Sconii += g.gcon[ii][jj]*S[jj];
      }
      rKperpcon[ii] += Sconii/D;
    }
    Real rKperpsq = 0.;
    SPACELOOP2(ii, jj) {
      rKperpsq += g.gcov4[ii+1][jj+1]*rKperpcon[ii]*rKperpcon[jj];
    }
    Real rbarKsq = 0.;
    SPACELOOP(ii) {
      rbarKsq += vcon[ii]*vcov[ii];
    }
    rbarKsq *= D*Gamma*Gamma*h*h;

    const Real qK = tau/D;
    const Real xK = 1./(1. + muK*bKsq);
    const Real qKbar = qK - 0.5*bKsq - 0.5*muK*muK*xK*xK*bKsq*rKperpsq;

    Real sie = Gamma*(qKbar - muK*rbarKsq) + Gamma - 1.;
    printf("ug: %e\n", ug);
    exit(-1);*/
    const Real sie = 0.;
    Real ug = 1.;

    //Real P = (gamma - 1.)/gamma*(w - rho);
    //Real P = (gamma - 1.)/gamma*(w - rho);
    //Real ug = P/(gamma - 1.);
    Real P = eos.PressureFromDensityInternalEnergy(rho, sie);


    v(prho) = rho;
    v(peng) = ug;
    v(prs) = P;
    v(pvel_lo) = Gamma*vcon[0];
    v(pvel_lo + 1) = Gamma*vcon[1];
    v(pvel_lo + 2) = Gamma*vcon[2];
    //if (v.i_ == 128 && v.j_ == 128) {
    //  printf("rho: %e ug: %e P: %e vel: %e %e %e\n", rho, ug, P, v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2));
      //exit(-1);
   // }

    v(tmp) = eos.TemperatureFromDensityInternalEnergy(v(prho), v(peng)/v(prho));
    v(prs) = eos.PressureFromDensityInternalEnergy(v(prho), v(peng)/v(prho));
    v(gm1) = eos.BulkModulusFromDensityTemperature(v(prho), v(tmp))/v(prs);

    // TODO(BRR) Set signal speeds more efficiently than by a p2c call
    Real ye_prim = 0.0;
    if (pye > 0) ye_prim = v(pye);
    Real ye_cons;
    //Real S[3];
    Real bcons[3];
    Real sig[3];
    Real bu[] = {0.0, 0.0, 0.0};
    prim2con::p2c(v(prho), vel, bu, v(peng), ye_prim, v(prs), v(gm1),
                  g.gcov4, g.gcon, g.beta, g.lapse, g.gdet,
                  v(crho), Scov, bcons, v(ceng), ye_cons, sig);
    for (int i = 0; i < sig_hi-sig_lo+1; i++) {
      v(sig_lo+i) = sig[i];
    }
    SPACELOOP(ii) {
      v(pb_lo+ii) = 0.;
    }

    if (isnan(rho) || isnan(ug) || isnan(P) || isnan(v(pvel_lo)) || isnan(v(pvel_lo+1)) ||
        isnan(v(pvel_lo+2)) || isnan(v(prs)) || isnan(v(gm1))) {
      //printf("%s:%i\n", __FILE__, __LINE__);
      //printf("%e %e %e %e %e %e %e %e %e\n",
      //  rho, ug, P, v(pvel_lo), v(pvel_lo+1), v(pvel_lo+2), v(prs), v(gm1));
      //exit(-1);
      return ConToPrimStatus::failure;
    }

    return ConToPrimStatus::success;
  }

  int NumBlocks() {
    return var.GetDim(5);
  }

  private:
  PackIndexMap imap;
  const T var;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int pb_lo, pb_hi;
  const int cb_lo, cb_hi;
  const int pye, cye;
  const int prs, tmp, sig_lo, sig_hi, gm1;
  const Real rel_tolerance;
  const int max_iter;
  const Real h0sq_;
};

using C2P_Block_t = ConToPrim<MeshBlockData<Real>, VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>, MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol,
                                  const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}

} // namespace con2prim_vanderholst

#endif // CON2PRIM_VANDERHOLST_HPP_
