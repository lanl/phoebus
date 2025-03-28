// © 2021-2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

enum class solver { LLF, HLL };

struct FaceGeom {
  KOKKOS_INLINE_FUNCTION
  FaceGeom(const Coordinates_t &coords, const Geometry::CoordSysMeshBlock &g,
           const CellLocation loc, const int d, const int k, const int j, const int i)
      : alpha(g.Lapse(loc, k, j, i)), gdet(g.DetG(loc, k, j, i)),
        gammadet(g.DetGamma(loc, k, j, i)) {
    auto gcon = reinterpret_cast<Real(*)[3]>(&gcov[0][0]);
    g.MetricInverse(loc, k, j, i, gcon);
    gdd = gcon[d - 1][d - 1];
    g.SpacetimeMetric(loc, k, j, i, gcov);
    g.ContravariantShift(loc, k, j, i, beta);
    X[1] = (loc == CellLocation::Face1 ? coords.Xf<1>(k, j, i) : coords.Xc<1>(k, j, i));
    X[2] = (loc == CellLocation::Face2 ? coords.Xf<2>(k, j, i) : coords.Xc<2>(k, j, i));
    X[3] = (loc == CellLocation::Face3 ? coords.Xf<3>(k, j, i) : coords.Xc<3>(k, j, i));
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
  static void ReconVars(const std::string &var) { recon_vars.push_back(var); }
  static void FluxVars(std::vector<std::string> &vars) {
    for (const auto &v : vars) {
      flux_vars.push_back(v);
    }
  }
  static void FluxVars(const std::string &var) { flux_vars.push_back(var); }

  static std::vector<std::string> ReconVars() {
    for (auto &v : recon_vars) {
    }
    return recon_vars;
  }
  static std::vector<std::string> FluxVars() {
    for (auto &v : flux_vars) {
    }
    return flux_vars;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int NumConserved() const { return ncons; }

  KOKKOS_INLINE_FUNCTION
  void prim_to_flux(const int d, const int k, const int j, const int i, const FaceGeom &g,
                    const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F,
                    const Real sie_max, const Real gam_max) const {
    const int dir = d - 1;
    Real rho_floor = q(dir, prho, k, j, i);
    Real sie_floor;
    bounds.GetFloors(g.X[1], g.X[2], g.X[3], rho_floor, sie_floor);
    const Real rho = std::max(q(dir, prho, k, j, i), rho_floor);
    Real vpcon[] = {q(dir, pvel_lo, k, j, i), q(dir, pvel_lo + 1, k, j, i),
                    q(dir, pvel_lo + 2, k, j, i)};
    Real W = phoebus::GetLorentzFactor(vpcon, g.gcov);
    if (W > gam_max) {
      const Real rescale = std::sqrt((gam_max * gam_max - 1.) / (W * W - 1.));
      SPACELOOP(ii) { vpcon[ii] *= rescale; }
    }
    Real vcon[] = {vpcon[0] / W, vpcon[1] / W, vpcon[2] / W};
    const Real &vel = vcon[dir];
    Real Bcon[] = {0.0, 0.0, 0.0};
    const Real u = (q(dir, peng, k, j, i) / rho > sie_floor
                        ? (q(dir, peng, k, j, i) / rho > sie_max ? rho * sie_max
                                                                 : q(dir, peng, k, j, i))
                        : rho * sie_floor);
    const Real P = std::max(q(dir, prs, k, j, i), 0.0);
    const Real gamma1 = q(dir, gm1, k, j, i);

    for (int m = pb_lo; m <= pb_hi; m++) {
      Bcon[m - pb_lo] = q(dir, m, k, j, i);
    }

    Real BdotB = 0.0;
    Real Bdotv = 0.0;
    Real vsq = 0.0;
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        vsq += g.gcov[m + 1][n + 1] * vcon[m] * vcon[n];
        Bdotv += g.gcov[m + 1][n + 1] * Bcon[m] * vcon[n];
        BdotB += g.gcov[m + 1][n + 1] * Bcon[m] * Bcon[n];
      }
    }
    const Real vsq_max = 1.0 - 1.0 / (gam_max * gam_max);
    if (vsq > vsq_max) {
      Real scale = vsq_max / vsq;
      vsq *= scale;
      scale = std::sqrt(scale);
      Bdotv *= scale;
      for (int m = 0; m < 3; m++)
        vcon[m] *= scale;
    }
    W = 1.0 / sqrt(1 - vsq);

    const Real vtil = vel - g.beta[dir] / g.alpha;

    // density
    U[crho] = rho * W;
    F[crho] = U[crho] * vtil;

    // composition
    if (cye > 0) {
      U[cye] = U[crho] * q(dir, pye, k, j, i);
      F[cye] = U[cye] * vtil;
    }

    Real b0 = W * Bdotv; // this is really b0*alpha
    const Real bsq = (BdotB + b0 * b0) / (W * W);

    // conserved momentum
    const Real rhohWsq = (rho + u + P) * W * W;
    const Real bsqWsq = bsq * W * W;

    for (int m = 0; m < 3; m++) {
      Real bcovm = g.gcov[m + 1][0] * b0 / g.alpha;
      Real vcovm = 0.0;
      for (int n = 1; n < 4; n++) {
        bcovm += g.gcov[m + 1][n] *
                 (Bcon[n - 1] / W + b0 * (vcon[n - 1] - g.beta[n - 1] / g.alpha));
        vcovm += g.gcov[m + 1][n] * vcon[n - 1];
      }

      U[cmom_lo + m] = (rhohWsq + bsqWsq) * vcovm - b0 * bcovm;
      F[cmom_lo + m] =
          U[cmom_lo + m] * vtil + (P + 0.5 * bsq) * Delta(dir, m) - bcovm * Bcon[dir] / W;
    }

// energy
#if USE_VALENCIA
    U[ceng] = rhohWsq + bsqWsq - (P + 0.5 * bsq) - b0 * b0 - U[crho];
    F[ceng] = U[ceng] * vtil + (P + 0.5 * bsq) * vel - Bdotv * Bcon[dir];
#else
    // TODO(BRR) share calculated quantities for ucon, ucov, bcon, bcov with above
    Real ucon[4] = {W / g.alpha, vpcon[0] - g.beta[0] * W / g.alpha,
                    vpcon[1] - g.beta[1] * W / g.alpha,
                    vpcon[2] - g.beta[2] * W / g.alpha};
    Real ucov[4] = {0};
    SPACETIMELOOP2(mu, nu) { ucov[mu] += g.gcov[mu][nu] * ucon[nu]; }
    Real bcon[4] = {b0 / g.alpha, 0., 0., 0.};
    Real bcov0 = 0.;
    SPACELOOP(ii) { bcon[ii + 1] = (Bcon[ii] + b0 * ucon[ii + 1]) / W; }
    SPACETIMELOOP(mu) { bcov0 += g.gcov[0][mu] * bcon[mu]; }
    U[ceng] = g.alpha * ((rho + u + P + bsq) * ucon[0] * ucov[0] + P + 0.5 * bsq) -
              bcon[0] * bcov0 + U[crho];
    F[ceng] = (rho + u + P + bsq) * ucon[d] * ucov[0] - bcon[d] * bcov0 + rho * ucon[d];
#endif // USE_VALENCIA

    // magnetic fields
    for (int m = cb_lo; m <= cb_hi; m++) {
      U[m] = Bcon[m - cb_lo];
      F[m] = U[m] * vtil - Bcon[dir] * (vcon[m - cb_lo] - g.beta[m - cb_lo] / g.alpha);
    }

    // TODO(JCD): are all these make_positive/make_bounded calls really necessary?
    const Real vasq = bsqWsq / robust::make_positive(rhohWsq + bsqWsq);
    const Real cssq = robust::make_bounded(
        gamma1 * P * W * W / robust::make_positive(rhohWsq), 0.0, 1.0);
    Real cmsq = robust::make_bounded(cssq + vasq * (1.0 - cssq), 0.0, 1.0);

    const Real vcoff = g.alpha / (1.0 - vsq * cmsq);
    const Real v0 = vel * (1.0 - cmsq);
    const Real vpm = sqrt(robust::make_positive(cmsq * (1.0 - vsq) *
                                                (g.gdd * (1.0 - vsq * cmsq) - vel * v0)));
    vp = vcoff * (v0 + vpm) - g.beta[dir];
    vm = vcoff * (v0 - vpm) - g.beta[dir];
  }

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
  const Geometry::CoordSysMeshBlock geom;
  const Coordinates_t coords;
  fixup::Bounds *pbounds;
  fixup::Bounds bounds;

 private:
  const int prho, pvel_lo, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, ceng, cb_lo, cb_hi, cye, ncons;
  static std::vector<std::string> recon_vars, flux_vars;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
      : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
        ql(rc->Get("ql").data), qr(rc->Get("qr").data),
        geom(Geometry::GetCoordinateSystem(rc)),
        coords(rc->GetParentPointer()->coords), // problem for packs
        pbounds(rc->GetParentPointer()
                    ->packages.Get("fixup")
                    .get()
                    ->MutableParam<fixup::Bounds>("bounds")),
        bounds(*pbounds), prho(imap[fluid_prim::density::name()].first),
        pvel_lo(imap[fluid_prim::velocity::name()].first),
        peng(imap[fluid_prim::energy::name()].first),
        pb_lo(imap[fluid_prim::bfield::name()].first),
        pb_hi(imap[fluid_prim::bfield::name()].second),
        pye(imap[fluid_prim::ye::name()].second),
        prs(imap[fluid_prim::pressure::name()].first),
        gm1(imap[fluid_prim::gamma1::name()].first),
        crho(imap[fluid_prim::density::name()].first),
        cmom_lo(imap[fluid_prim::velocity::name()].first),
        ceng(imap[fluid_prim::energy::name()].first),
        cb_lo(imap[fluid_prim::bfield::name()].first),
        cb_hi(imap[fluid_prim::bfield::name()].second),
        cye(imap[fluid_prim::ye::name()].first),
        ncons(5 + (pb_hi - pb_lo + 1) + (cye > 0)) {
    PARTHENON_REQUIRE_THROWS(
        ncons <= NCONS_MAX,
        "ncons exceeds NCONS_MAX.  Reconfigure to increase NCONS_MAX.");
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int Delta(const int i, const int j) const { return i == j; }
};

KOKKOS_INLINE_FUNCTION
Real llf(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  CellLocation loc = DirectionToFaceID(d);
  FaceGeom g(fs.coords, fs.geom, loc, d, k, j, i);
  Real gam_max, sie_max;
  fs.bounds.GetCeilings(g.X[1], g.X[2], g.X[3], gam_max, sie_max);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl, sie_max, gam_max);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr, sie_max, gam_max);

  const Real cmax = std::max(std::max(-vml, vpl), std::max(-vmr, vpr));

  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d, m, k, j, i) =
        0.5 * ((Fl[m] + Fr[m]) * g.gdet - cmax * ((Ur[m] - Ul[m]) * g.gammadet));
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
  Real gam_max, sie_max;
  fs.bounds.GetCeilings(g.X[1], g.X[2], g.X[3], gam_max, sie_max);
  fs.prim_to_flux(d, k, j, i, g, fs.ql, vml, vpl, Ul, Fl, sie_max, gam_max);
  fs.prim_to_flux(d, k, j, i, g, fs.qr, vmr, vpr, Ur, Fr, sie_max, gam_max);

  const Real cl = std::min(std::min(vml, vmr), 0.0);
  const Real cr = std::max(std::max(vpl, vpr), 0.0);
  const Real crcl = cr * cl;
  const Real inv_cr_cl = robust::ratio(1.0, cr - cl);
  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d, m, k, j, i) =
        ((cr * Fl[m] - cl * Fr[m]) * g.gdet + crcl * (Ur[m] - Ul[m]) * g.gammadet) *
        inv_cr_cl;
  }
  return std::max(-cl, cr);
}

} // namespace riemann

#endif // FLUID_RIEMANN_HPP_
