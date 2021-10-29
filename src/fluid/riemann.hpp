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
#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

#include "phoebus_utils/valencia_cowling.hpp"


namespace riemann {

enum class solver {LLF, HLL};

struct FaceGeom {
  KOKKOS_INLINE_FUNCTION
  FaceGeom(const Coordinates_t &coords, const Geometry::CoordSysMeshBlock &g, const CellLocation loc,
           const int d, const int k, const int j, const int i)
           : alpha(g.Lapse(loc,k,j,i)), gdet(g.DetG(loc,k,j,i)),
             gammadet(g.DetGamma(loc,k,j,i)) {
    auto gcon = reinterpret_cast<Real (*)[3]>(&gcov[0][0]);
    g.MetricInverse(loc,k,j,i,gcon);
    gdd = gcon[d-1][d-1];
    g.SpacetimeMetric(loc,k,j,i,gcov);
    g.ContravariantShift(loc,k,j,i,beta);
    X[1] = (loc == CellLocation::Face1 ? coords.x1f(k, j, i) : coords.x1v(k, j, i));
    X[2] = (loc == CellLocation::Face2 ? coords.x2f(k, j, i) : coords.x2v(k, j, i));
    X[2] = (loc == CellLocation::Face3 ? coords.x3f(k, j, i) : coords.x3v(k, j, i));

    g.MetricInverse(loc, k, j, i, gammacon);
    g.Metric(loc, k, j, i, gammacov);
  }
  const Real alpha;
  const Real gdet;
  const Real gammadet;
  Real gcov[4][4];
  Real beta[3];
  Real gdd;
  Real X[4];

  Real gammacon[3][3];
  Real gammacov[3][3];
  Real dgcov[4][4][4];
  Real gradlnalpha[4];
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
                    const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F,
                    const Real rho_floor, const Real sie_floor, const Real sie_max,
                    const Real gam_max) const {
    const int dir = d-1;

    /*Real rho = q(dir,prho,k,j,i);
    Real u = q(dir,peng,k,j,i);
    if (rho < rho_floor) {
      rho = rho_floor;
      u = 0.1*rho_floor;
    }*/

    const Real rho = std::max(q(dir,prho,k,j,i),rho_floor);
    Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_lo+2,k,j,i)};
    const Real &vel = vcon[dir];
    Real Bcon[] = {0.0, 0.0, 0.0};
    const Real u = (q(dir,peng,k,j,i)/rho > sie_floor
                      ? (q(dir,peng,k,j,i)/rho > sie_max
                          ? rho*sie_max
                          : q(dir,peng,k,j,i))
                      : rho*sie_floor);
    // TODO(BRR) nasty hack to test consistent prs
    //const Real P = (5./3. - 1.)*u;
    const Real P = std::max(q(dir,prs,k,j,i), 0.0);
    const Real gamma1 = q(dir,gm1,k,j,i);
    PARTHENON_REQUIRE(!isnan(gamma1), "gamma1 is nan?");

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
    const Real vsq_max = 1.0 - 1.0/(gam_max*gam_max);
    if (vsq > vsq_max) {
      Real scale = vsq_max/vsq;
      vsq *= scale;
      scale = std::sqrt(scale);
      Bdotv *= scale;
      for (int m = 0; m < 3; m++) vcon[m] *= scale;
    }
    const Real W = 1.0/sqrt(1-vsq);

    const Real vtil = vel - g.beta[dir]/g.alpha;

    // density
    U[crho] = rho*W;
    F[crho] = U[crho]*vtil;
    /*if (j == 127 + 4 && d == 1 && i > 180 && i < 183) {
      //printf("[%i] vtil = %e vel = %e u^1 = %e F[RHO] = %e\n", i, vtil, vel, vtil*W/g.alpha, F[crho]);
      printf("[%i] v: %28.18e vtilde: %28.18e beta: %28.18e alpha: %28.18e\n", i, vel, vtil, g.beta[0], g.alpha);
      printf("[%i] wanted v: %28.18e\n", i, g.beta[0]/g.alpha);
    }*/

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
    }

    // energy
    U[ceng] = rhohWsq + bsqWsq - (P + 0.5*bsq) - b0*b0 - U[crho];
    F[ceng] = U[ceng]*vtil + (P + 0.5*bsq)*vel - Bdotv*Bcon[dir];

    // magnetic fields
    for (int m = cb_lo; m <= cb_hi; m++) {
      U[m] = Bcon[m-cb_lo];
      F[m] = U[m]*vtil - Bcon[dir]*(vcon[m-cb_lo] - g.beta[m-cb_lo]/g.alpha);
    }

    const Real vasq = bsqWsq/robust::make_positive(rhohWsq+bsqWsq);
    const Real cssq = robust::make_bounded(gamma1*P*W*W/robust::make_positive(rhohWsq), 0.0, 1.0);
    Real cmsq = robust::make_bounded(cssq + vasq*(1.0 - cssq), 0.0, 1.0);
    //cmsq = (cmsq > 0.0 ? cmsq : 1.e-16); // TODO(JCD): what should this 1.e-16 be?
    //cmsq = (cmsq > 1.0 ? 1.0 : cmsq);

    const Real vcoff = g.alpha/(1.0 - vsq*cmsq);
    const Real v0 = vel*(1.0 - cmsq);
    const Real vpm = sqrt(robust::make_positive(cmsq*(1.0  - vsq)*(g.gdd*(1.0 - vsq*cmsq) - vel*v0)));
    vp = vcoff*(v0 + vpm) - g.beta[dir];
    vm = vcoff*(v0 - vpm) - g.beta[dir];

    // TODO(BRR) break the code
    /*F[crho] = 0.;
    F[ceng] = 0.;
    SPACELOOP(ii) {
      F[cmom_lo + ii] = 0.;
    }*/

    if (isnan(vp) || isnan(vm)) {
      printf("[%i %i %i] Nan in waves! %e %e %e %e %e %e [%i]\n", k,j,i,vp, vm, vcoff, v0, vpm, g.beta[dir], dir);
      printf("p: %e %e %e %e %e\n", rho, vel, u, P, gamma1);
      printf("vsq: %e cmsq: %e vasq: %e cssq: %e\n", vsq, cmsq, vasq, cssq);
      PARTHENON_FAIL("stupid code");
    }

    // TODO(BRR) use my own fluxes
    /*Real vcov[3] = {0};
    SPACELOOP2(ii, jj) {
      vcov[ii] += g.gammacov[ii][jj]*vcon[jj];
    }
    Real vtildecon[3] = {0};
    SPACELOOP(ii) {
      vtildecon[ii] = g.alpha*vcon[ii] - g.beta[ii];
    }
    Real Scov[3] = {0};
    SPACELOOP(ii) {
      Scov[ii] =  (rho + u + P)*W*W*vcov[ii];
    }
    Real Scon[3] = {0};
    SPACELOOP2(ii, jj) {
      Scon[ii] += g.gammacon[ii][jj]*Scov[jj];
    }
    Real Wconcon[3][3] = {0};
    SPACELOOP2(ii, jj) {
      Wconcon[ii][jj] = Scon[ii]*vcon[jj] + P*g.gammacon[ii][jj];
    }
    Real Wconcov[3][3] = {0};
    SPACELOOP3(ii, jj, kk) {
      Wconcov[ii][jj] += Wconcon[ii][jj]*g.gammacov[jj][kk];
    }
    Real D = U[crho];
    Real tau  = U[ceng];
    Real NEWF[5] = {0};
    //F[crho] = D*vtildecon[dir]/g.alpha;
    NEWF[0] = D*vtildecon[dir]/g.alpha;
    //F[ceng] = Scon[dir] - vcon[dir]*D - g.beta[dir]/g.alpha*tau;
    NEWF[4] = Scon[dir] - vcon[dir]*D - g.beta[dir]/g.alpha*tau;

    SPACELOOP(jj) {
      //F[cmom_lo + jj] = Wconcov[dir][jj] - g.beta[dir]/g.alpha*Scov[jj];
      NEWF[jj + 1] = Wconcov[dir][jj] - g.beta[dir]/g.alpha*Scov[jj];
    }//
    */

    auto vc = ValenciaCowling(g.alpha, g.beta, g.gammacov, g.gammacon, g.dgcov,
      g.gradlnalpha, rho, u, P, vcon);
    Real NEWF[5] = {0};
    NEWF[crho] = vc.F[dir][0];
    SPACELOOP(ii) {
      NEWF[cmom_lo + ii] = vc.F[dir][ii + 1];
    }
    NEWF[ceng] = vc.F[dir][4];

    /*if (i == 120 && j == 120) {
      //printf("NEWU: %e %e %e %e %e OLDU: %e %e %e %e %e\n",
      //  vc.U[0], vc.U[1], vc.U[2], vc.U[3], vc.U[4],
      //  U[crho], U[cmom_lo], U[cmom_lo+1], U[cmom_lo+2], U[ceng]);
      printf("F NEW[%i]: %e %e %e %e %e\n  OLD   : %e %e %e %e %e\n", dir, NEWF[0], NEWF[1], NEWF[2], NEWF[3], NEWF[4], F[crho], F[cmom_lo], F[cmom_lo+1], F[cmom_lo+2], F[ceng]);
    }*/

    /*F[crho] = vc.F[dir][0];
    SPACELOOP(ii) {
      F[cmom_lo + ii] = vc.F[dir][ii + 1];
    }
    F[ceng] = vc.F[dir][4];*/

  }

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
  const Geometry::CoordSysMeshBlock geom;
  const Coordinates_t coords;
  fixup::Bounds bounds;
  const int prho, pvel_lo, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, ceng, cb_lo, cb_hi, cye, ncons;
 private:
  //const int prho, pvel_lo, peng, pb_lo, pb_hi, pye, prs, gm1;
  //const int crho, cmom_lo, ceng, cb_lo, cb_hi, cye, ncons;
  static std::vector<std::string> recon_vars, flux_vars;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
    : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
      ql(rc->Get("ql").data),
      qr(rc->Get("qr").data),
      geom(Geometry::GetCoordinateSystem(rc)),
      coords(rc->GetParentPointer().get()->coords),
      bounds(rc->GetParentPointer().get()->packages.Get("fixup").get()->Param<fixup::Bounds>("bounds")),
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
    //auto *pmb = rc->GetParentPointer().get();
    //StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
    //floor_ = fix_pkg->Param<fixup::Floors>("floor");
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
  FaceGeom g(fs.coords, fs.geom, loc, d, k, j, i);
  Real rho_floor, sie_floor;
  fs.bounds.GetFloors(g.X[1], g.X[2], g.X[3], rho_floor, sie_floor);
  Real gam_max, sie_max;
  fs.bounds.GetCeilings(g.X[1], g.X[2], g.X[3], gam_max, sie_max);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl, rho_floor, sie_floor, sie_max, gam_max);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr, rho_floor, sie_floor, sie_max, gam_max);

  const Real cmax = std::max(std::max(-vml,vpl), std::max(-vmr,vpr));

  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = 0.5*((Fl[m] + Fr[m])*g.gdet - cmax*((Ur[m] - Ul[m])*g.gammadet));
    // TODO(BRR) break code
    //fs.v.flux(d,m,k,j,i) = 0.5*(- cmax*((Ur[m] - Ul[m])*g.gammadet));;
    /*if (m == 0 && j == 127 + 4 && i > 180 && i < 183) {
      printf("[%i] F: %e fl: %e fr: %e (%e) ur: %e ul: %e (%e)\n",
        i, fs.v.flux(d,m,k,j,i), Fl[m], Fr[m], 0.5*(Fl[m] + Fr[m])*g.gdet,
        Ur[m], Ul[m], cmax*(Ur[m] - Ul[m])*g.gammadet);
    }*/

    if (m == 0 && d == 2 && i == 133 && j == 84) {
      printf("rhol = %e rhor = %e Ul = %e Ur = %e Fl = %e Fr = %e Flux = %e cmax = %e\n",
        fs.ql(d-1,0,k,j,i), fs.qr(d-1,0,k,j,i), Ul[0], Ur[0], Fl[0], Fr[0], fs.v.flux(d,m,k,j,i), cmax);
    }
    if (isnan(Fl[m]) || isnan(Fr[m]) || isnan(cmax) || isnan(Ur[m]) || isnan(Ul[m])) {
      printf("A nan in a flux! %e %e %e %e %e\n", Fl[m], Fr[m], cmax, Ur[m], Ul[m]);
      PARTHENON_FAIL("a nan in a flux :(");
    }
  }
  return cmax;
}

KOKKOS_INLINE_FUNCTION
Real hll(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  CellLocation loc = DirectionToFaceID(d);
  FaceGeom g(fs.coords, fs.geom, loc, d, k, j, i);
  Real rho_floor, sie_floor;
  fs.bounds.GetFloors(g.X[1], g.X[2], g.X[3], rho_floor, sie_floor);
  Real gam_max, sie_max;
  fs.bounds.GetCeilings(g.X[1], g.X[2], g.X[3], gam_max, sie_max);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl, rho_floor, sie_floor, sie_max, gam_max);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr, rho_floor, sie_floor, sie_max, gam_max);

  const Real cl = std::min(std::min(vml, vmr), 0.0);
  const Real cr = std::max(std::max(vpl, vpr), 0.0);

  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = ((cr*Fl[m] - cl*Fr[m])*g.gdet + cr*cl*(Ur[m] - Ul[m])*g.gammadet)/(cr - cl);
  }
  return std::max(-cl,cr);
}

} // namespace riemann

#endif // FLUID_RIEMANN_HPP_
