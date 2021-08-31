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
    X[3] = (loc == CellLocation::Face3 ? coords.x3f(k, j, i) : coords.x3v(k, j, i));
  }
  const Real alpha;
  const Real gdet;
  const Real gammadet;
  Real gcov[4][4];
  Real beta[3];
  Real gdd;
  Real X[4];
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
    const Real rho = std::max(q(dir,prho,k,j,i),rho_floor);
    Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_lo+2,k,j,i)};
    const Real &vel = vcon[dir];
    Real Bcon[] = {0.0, 0.0, 0.0};
    const Real u = (q(dir,peng,k,j,i)/rho > sie_floor
                      ? (q(dir,peng,k,j,i)/rho > sie_max
                          ? rho*sie_max
                          : q(dir,peng,k,j,i))
                      : rho*sie_floor);
    const Real P = std::max(q(dir,prs,k,j,i), 0.0);
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
  }

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
  const Geometry::CoordSysMeshBlock geom;
  const Coordinates_t coords;
  fixup::Bounds bounds;
 private:
  const int prho, pvel_lo, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, ceng, cb_lo, cb_hi, cye, ncons;
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
