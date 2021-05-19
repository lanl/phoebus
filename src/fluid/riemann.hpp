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

#ifndef FLUID_RIEMANN_HPP_
#define FLUID_RIEMANN_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "compile_constants.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"


namespace riemann {

enum class solver {LLF, HLL};

struct FaceGeom {
  KOKKOS_INLINE_FUNCTION
  FaceGeom(const Geometry::CoordSysMeshBlock &g, const CellLocation loc,
           const int d, const int k, const int j, const int i)
           : alpha(g.Lapse(loc,k,j,i)), gdet(g.DetG(loc,k,j,i)) {
    auto gcon = reinterpret_cast<Real (*)[3]>(&gcov[0][0]);
    g.MetricInverse(loc,k,j,i,gcon);
    gdd = gcon[d-1][d-1];
    g.SpacetimeMetric(loc,k,j,i,gcov);
    g.ContravariantShift(loc,k,j,i,beta);

    // TEMP
    g.MetricInverse(loc,k,j,i,gcon_);
  }
  const Real alpha;
  const Real gdet;
  Real gcov[4][4];
  Real beta[3];
  Real gdd;
  Real gcon_[3][3];
};

class FluxState {
 public:
  FluxState(MeshBlockData<Real> *rc) : FluxState(rc, PackIndexMap()) {}

  static void ReconVars(std::vector<std::string> &vars) {
    for (const auto &v : vars) {
      recon_vars.push_back(v);
    }
  }
  static void ReconVars(const std::string &var) {
    recon_vars.push_back(var);
  }
  static void FluxVars(std::vector<std::string> &vars) {
    for (const auto &v : vars) {
      flux_vars.push_back(v);
    }
  }
  static void FluxVars(const std::string &var) {
    flux_vars.push_back(var);
  }

  static std::vector<std::string> ReconVars() {
    return recon_vars;
  }
  static std::vector<std::string> FluxVars() {
    return flux_vars;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int NumConserved() const {
    return ncons;
  }

  KOKKOS_INLINE_FUNCTION
  void prim_to_flux(const int d, const int k, const int j, const int i, const FaceGeom &g,
                    const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F) const {
    const int dir = d-1;
    const Real rho = q(dir,prho,k,j,i);
    const Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_lo+2,k,j,i)};
    const Real &vel = vcon[dir];
    Real Bcon[] = {0.0, 0.0, 0.0};
    const Real u = q(dir,peng,k,j,i);
    const Real P = q(dir,prs,k,j,i);
    const Real gamma1 = q(dir,gm1,k,j,i);

    for (int m = pb_lo; m <= pb_hi; m++) {
      Bcon[m-pb_lo] = q(dir, m, k, j, i);
    }

    Real BdotB = 0.0;
    Real Bdotv = 0.0;
    Real vsq = 0.0;
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        vsq += g.gcov[m+1][n+1]*vcon[m]*vcon[n];
        Bdotv += g.gcov[m+1][n+1]*Bcon[m]*vcon[n];
        BdotB += g.gcov[m+1][n+1]*Bcon[m]*Bcon[n];
      }
    }
    const Real W = 1.0/sqrt(1-vsq);

    const Real vtil = vel - g.beta[dir]/g.alpha;

    // density
    U[crho] = rho*W;
    F[crho] = U[crho]*vtil;

    // composition
    if (cye>0) {
      U[cye] = U[crho]*q(dir,pye,k,j,i);
      F[cye] = U[cye]*vtil;
    }

    Real b0 = W*Bdotv; // this is really b0*alpha
    const Real bsq = (BdotB + b0*b0)/(W*W);

    // conserved momentum
    const Real rhohWsq = (rho + u + P)*W*W;
    const Real bsqWsq = bsq*W*W;
    for (int m = 0; m < 3; m++) {
      Real bcovm = g.gcov[m+1][0] * b0/g.alpha;
      Real vcovm = 0.0;
      for (int n = 1; n < 4; n++) {
        bcovm += g.gcov[m+1][n] * (Bcon[n-1]/W + b0*(vcon[n-1]-g.beta[n-1]/g.alpha));
        vcovm += g.gcov[m+1][n] * vcon[n-1];
      }
      U[cmom_lo+m] = (rhohWsq+bsqWsq)*vcovm - b0*bcovm;
      F[cmom_lo+m] = U[cmom_lo+m]*vtil + (P + 0.5*bsq)*Delta(dir,m) - bcovm*Bcon[dir]/W;
      //printf("Fold[%i] = %e Fnew = %e\n", cmom_lo+m, F[cmom_lo+m],
      //  rhohWsq*vcovm*(g.alpha*vcon[dir]-g.beta[dir]) + g.alpha*P*Delta(dir,m));
    }

    // energy
    U[ceng] = rhohWsq + bsqWsq - (P + 0.5*bsq) - b0*b0 - U[crho];
    //if (i == 64 && j == 64)
    //printf("Calcing ceng: %e + %e - %e - %e - %e = %e\n", rhohWsq, bsqWsq, P+0.5*bsq, b0*b0, U[crho], U[ceng]);
    F[ceng] = U[ceng]*vtil + (P + 0.5*bsq)*vel - Bdotv*Bcon[dir];

    // magnetic fields
    for (int m = cb_lo; m <= cb_hi; m++) {
      U[m] = Bcon[m-cb_lo];
      F[m] = U[m]*vtil - Bcon[dir]*(vcon[m-cb_lo] - g.beta[m-cb_lo]/g.alpha);
    }

    const Real vasq = bsqWsq/(rhohWsq+bsqWsq);
    const Real cssq = gamma1*P*W*W/rhohWsq;
    Real cmsq = cssq + vasq*(1.0 - cssq);
    cmsq = (cmsq > 0.0 ? cmsq : 1.e-16); // TODO(JCD): what should this 1.e-16 be?
    cmsq = (cmsq > 1.0 ? 1.0 : cmsq);

    const Real vcoff = g.alpha/(1.0 - vsq*cmsq);
    const Real v0 = vel*(1.0 - cmsq);
    const Real vpm = sqrt(cmsq*(1.0  - vsq)*(g.gdd*(1.0 - vsq*cmsq) - vel*v0));
    vp = vcoff*(v0 + vpm) - g.beta[dir];
    vm = vcoff*(v0 - vpm) - g.beta[dir];

/*    if (j == 64 && i == 64) {
//    {
      //printf("Diag cons and flux!\n");
 //     printf("Original U: %e %e %e %e %e\n", U[crho], U[ceng], U[cmom_lo], U[cmom_lo+1], U[cmom_lo+2]);
  //    printf("Original F: %e %e %e %e %e\n", F[crho], F[ceng], F[cmom_lo], F[cmom_lo+1], F[cmom_lo+2]);
      //printf("rho: %e u: %e v: %e %e %e\n", rho, u, vcon[0], vcon[1], vcon[2]);
      Real vcov[3] = {0, 0, 0};
      Real vsq = 0.0;
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          vcov[m] += g.gcov[m+1][n+1]*vcon[n];
          vsq += g.gcov[m+1][n+1]*vcon[m]*vcon[n];
        }
      }
      //printf("vcov: %e %e %e\n", vcov[0], vcov[1], vcov[2]);
      Real Gamma = 1./sqrt(1. - vsq);
      //printf("Gamma: %e\n", Gamma);

      //printf("dir: %i\n", dir);

      Real vtil = vcon[dir];

      Real NewU[5] = {0, 0, 0, 0, 0};
      Real NewF[5] = {0, 0, 0, 0, 0};

      NewU[0] = rho*Gamma;
      NewF[0] = rho*vtil;

      double h = 1 + u/rho + P/rho;
      Real Scon[3], Scov[3];
      for (int m = 0; m < 3; m++) {
        Scon[m] = rho*h*Gamma*Gamma*vcon[m];
        Scov[m] = rho*h*Gamma*Gamma*vcov[m];
      }
      Real Sconi = Scon[dir];//rho*h*Gamma*Gamma*vcon[dir];
      Real Scovi = Scov[dir];//rho*h*Gamma*Gamma*vcov[dir];

      //printf("old rhohWsq: %e new: %e\n", rhohWsq, rho*h*Gamma*Gamma);

      Real Wij_UU[3][3];
      Real Wij_UD[3][3] = {0};
      //Real gcon[3][3];
      //g.MetricInverse(loc_,k,j,i,gcon);
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          Wij_UU[m][n] = Scon[m]*vcon[n] + P*g.gcon_[m][n];
        }
      }
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          for (int kap = 0; kap < 3; kap++) {
            Wij_UD[n][m] += g.gcov[m+1][kap+1]*Wij_UU[n][kap];
          }
        }
      }


      NewU[1] = rho*h*Gamma*Gamma - P - NewU[0];
      NewF[1] = Sconi - vcon[dir]*NewU[0];

      NewU[2] = Scov[0];
      NewU[3] = Scov[1];
      NewU[4] = Scov[2];

      NewF[2] = Wij_UD[dir][0];
      NewF[3] = Wij_UD[dir][1];
      NewF[4] = Wij_UD[dir][2];

//      F[cmom_lo+1] = NewF[3];

//      printf("New U:      %e %e %e %e %e\n", NewU[0], NewU[1], NewU[2], NewU[3], NewU[4]);
//      printf("New F:      %e %e %e %e %e\n", NewF[0], NewF[1], NewF[2], NewF[3], NewF[4]);

//      exit(-1);
    }*/
  }

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
  const Geometry::CoordSysMeshBlock geom;
 private:
  const int prho, pvel_lo, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, ceng, cb_lo, cb_hi, cye, ncons;
  static std::vector<std::string> recon_vars, flux_vars;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
    : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
      ql(rc->Get("ql").data),
      qr(rc->Get("qr").data),
      geom(Geometry::GetCoordinateSystem(rc)),
      prho(imap[fluid_prim::density].first),
      pvel_lo(imap[fluid_prim::velocity].first),
      peng(imap[fluid_prim::energy].first),
      pb_lo(imap[fluid_prim::bfield].first),
      pb_hi(imap[fluid_prim::bfield].second),
      pye(imap[fluid_prim::ye].second),
      prs(imap[fluid_prim::pressure].first),
      gm1(imap[fluid_prim::gamma1].first),
      crho(imap[fluid_cons::density].first),
      cmom_lo(imap[fluid_cons::momentum].first),
      ceng(imap[fluid_cons::energy].first),
      cb_lo(imap[fluid_cons::bfield].first),
      cb_hi(imap[fluid_cons::bfield].second),
      cye(imap[fluid_cons::ye].first),
      ncons(5 + (pb_hi-pb_lo+1) + (cye>0)) {
    PARTHENON_REQUIRE_THROWS(ncons <= NCONS_MAX, "ncons exceeds NCONS_MAX.  Reconfigure to increase NCONS_MAX.");
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int Delta(const int i, const int j) const {
    return i==j;
  }
};

KOKKOS_INLINE_FUNCTION
Real llf(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  CellLocation loc = DirectionToFaceID(d);
  FaceGeom g(fs.geom, loc, d, k, j, i);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr);

  const Real cmax = std::max(std::max(-vml,vpl), std::max(-vmr,vpr));

  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = 0.5*(Fl[m] + Fr[m] - cmax*(Ur[m] - Ul[m])) * g.gdet;
  }
  return cmax;
}

KOKKOS_INLINE_FUNCTION
Real hll(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  CellLocation loc = DirectionToFaceID(d);
  FaceGeom g(fs.geom, loc, d, k, j, i);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr);

  const Real cl = std::min(std::min(vml, vmr), 0.0);
  const Real cr = std::max(std::max(vpl, vpr), 0.0);

  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = (cr*Fl[m] - cl*Fr[m] + cr*cl*(Ur[m] - Ul[m]))/(cr - cl) * g.gdet;
  }
  return std::max(-cl,cr);
}

} // namespace riemann

#endif // FLUID_RIEMANN_HPP_
