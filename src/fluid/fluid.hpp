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

#ifndef FLUID_HPP_
#define FLUID_HPP_

#include <memory>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "compile_constants.hpp"
#include <eos/eos.hpp>
#include "con2prim.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"

namespace fluid {

enum class RiemannSolver {LLF, HLL};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc);
template <typename T>
TaskStatus ConservedToPrimitive(T *rc);
TaskStatus CalculateFluidSourceTerms(MeshBlockData<Real> *rc, MeshBlockData<Real> *rc_src);
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

class FluxState {
 public:
  FluxState(MeshBlockData<Real> *rc) : FluxState(rc, PackIndexMap()) {}

  KOKKOS_INLINE_FUNCTION
  void Solve(const int d, const int k, const int j, const int i) const {
    switch(solver){
      case RiemannSolver::LLF:
        llf(d,k,j,i);
        break;
      case RiemannSolver::HLL:
        hll(d, k, j ,i);
        break;
      default:
        PARTHENON_FAIL("Invalid Riemann Solver");
    }
  }

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

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
 private:
  const Geometry::CoordinateSystem geom;
  const RiemannSolver solver;
  const int prho, pvel_lo, pvel_hi, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, cmom_hi, ceng, cb_lo, cb_hi, cye, ncons;
  static std::vector<std::string> recon_vars, flux_vars;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
    : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
      ql(rc->Get("ql").data),
      qr(rc->Get("qr").data),
      geom(Geometry::GetCoordinateSystem(rc)),
      solver(rc->GetBlockPointer()->packages.Get("fluid")->Param<RiemannSolver>("RiemannSolver")),
      prho(imap[primitive_variables::density].first),
      pvel_lo(imap[primitive_variables::velocity].first),
      pvel_hi(imap[primitive_variables::velocity].second),
      peng(imap[primitive_variables::energy].first),
      pb_lo(imap[primitive_variables::bfield].first),
      pb_hi(imap[primitive_variables::bfield].second),
      pye(imap[primitive_variables::ye].second),
      prs(imap[primitive_variables::pressure].first),
      gm1(imap[primitive_variables::gamma1].first),
      crho(imap[conserved_variables::density].first),
      cmom_lo(imap[conserved_variables::momentum].first),
      cmom_hi(imap[conserved_variables::momentum].second),
      ceng(imap[conserved_variables::energy].first),
      cb_lo(imap[conserved_variables::bfield].first),
      cb_hi(imap[conserved_variables::bfield].second),
      cye(imap[conserved_variables::ye].first),
      ncons(5 + (pb_hi-pb_lo+1) + (cye>0)) {
    PARTHENON_REQUIRE_THROWS(ncons <= NCONS_MAX, "ncons exceeds NCONS_MAX.  Reconfigure to increase NCONS_MAX.");
  }



  KOKKOS_FORCEINLINE_FUNCTION
  int Delta(const int i, const int j) const {
    return i==j;
  }

  KOKKOS_INLINE_FUNCTION
  void prim_to_flux(const int d, const int k, const int j, const int i,
                    const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F) const {
    const int dir = d-1;
    const Real &rho = q(dir,prho,k,j,i);
    const Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_hi,k,j,i)};
    const Real &vel = vcon[dir];
    Real Bcon[] = {0.0, 0.0, 0.0};
    const Real &u = q(dir,peng,k,j,i);
    const Real &P = q(dir,prs,k,j,i);
    const Real &gamma1 = q(dir,gm1,k,j,i);

    for (int m = pb_lo; m <= pb_hi; m++) {
      Bcon[m-pb_lo] = q(dir, m, k, j, i);
    }

    CellLocation loc = DirectionToFaceID(d);

    Real gcov[3][3];
    geom.Metric(loc, k, j, i, gcov);

    Real vcov[3];
    Real BdotB = 0.0;
    Real Bdotv = 0.0;
    for (int m = 0; m < 3; m++) {
      vcov[m] = 0.0;
      for (int n = 0; n < 3; n++) {
        vcov[m] += gcov[m][n]*vcon[n];
      }
      Bdotv += Bcon[m]*vcov[m];
      // TODO(JCD): should we just always execute this loop from 0 <= n < 3
      for (int n = pb_lo; n <= pb_hi; n++) {
        BdotB += gcov[m][n-pb_lo] * Bcon[m] * Bcon[n-pb_lo];
      }
    }

    // get Lorentz factor
    Real vsq = 0.0;
    for (int m = 0; m < 3; m++) {
      vsq += vcon[m]*vcov[m];
    }
    const Real W = 1.0/sqrt(1.0 - vsq);

    const Real alpha = geom.Lapse(d, k, j, i);
    const Real vt[3] = {vcon[0] - geom.ContravariantShift(1, loc, k, j, i)/alpha,
                        vcon[1] - geom.ContravariantShift(2, loc, k, j, i)/alpha,
                        vcon[2] - geom.ContravariantShift(3, loc, k, j, i)/alpha};
    const Real &vtil = vt[dir];

    Real b[4] = {W*Bdotv/alpha, 0.0, 0.0, 0.0};
    for (int m = pb_lo; m <= pb_hi; m++) {
      b[m-pb_lo+1] = Bcon[m-pb_lo]/W + alpha*b[0]*vt[m-pb_lo];
    }
    const Real bsq = (BdotB + alpha*alpha*b[0]*b[0])/(W*W);
    Real bcov[] = {0.0, 0.0, 0.0};
    if (pb_hi > 0) {
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 4; n++) {
          bcov[m] += geom.SpacetimeMetric(m+1, n, loc, k, j, i) * b[n];
        }
      }
    }

    // conserved density
    U[crho] = rho*W;
    if (cye>0) U[cye] = U[crho]*q(dir,pye,k,j,i);

    // conserved momentum
    const Real rhohWsq = (rho + u + P + bsq)*W*W;
    for (int m = 0; m < 3; m++) {
      U[cmom_lo+m] = rhohWsq*vcov[m] - alpha*b[0]*bcov[m];
    }

    // conserved energy
    U[ceng] = rhohWsq - (P + 0.5*bsq) - alpha*alpha*b[0]*b[0] - U[crho];

    // magnetic fields
    for (int m = cb_lo; m <= cb_hi; m++) {
      U[m] = Bcon[m-cb_lo];
    }

    // Get fluxes
    // mass flux
    F[crho] = U[crho]*vtil;
    if (cye>0) F[cye] = U[cye]*vtil;

    // momentum flux
    for (int n = 0; n < 3; n++) {
      F[cmom_lo+n] = U[cmom_lo+n]*vtil + (P + 0.5*bsq)*Delta(dir,n) - bcov[n]*Bcon[dir]/W;
    }

    // energy flux
    F[ceng] = U[ceng]*vtil + (P + 0.5*bsq)*vel - alpha*b[0]*Bcon[dir]/W;

    for (int n = cb_lo; n <= cb_hi; n++) {
      F[n] = U[n]*vtil - Bcon[dir]*vt[n-cb_lo];
    }
 
    const Real vasq = bsq*W*W/rhohWsq;
    const Real cssq = gamma1*P/rho;
    Real cmsq = cssq + vasq - cssq*vasq;
    cmsq = (cmsq > 0.0 ? cmsq : 1.e-16); // TODO(JCD): what should this 1.e-16 be?
    cmsq = (cmsq > 1.0 ? 1.0 : cmsq);

    const Real gdd = geom.MetricInverse(d,d,loc,k,j,i);
    const Real vcoff = alpha/(1. - vsq*cmsq);
    const Real v0 = vcon[dir]*(1.0 - cmsq);
    const Real vpm = sqrt(cmsq*(1.0  - vsq)*(gdd*(1.0 - vsq*cmsq) - vel*vel*(1.0 - cmsq)));
    vp = vcoff*(v0 + vpm);
    vm = vcoff*(v0 - vpm);

    //vp = vel + sqrt(cmsq);
    //vm = vel - sqrt(cmsq);
  }

  KOKKOS_INLINE_FUNCTION
  void llf(const int d, const int k, const int j, const int i) const {
    Real Ul[NCONS_MAX], Ur[NCONS_MAX];
    Real Fl[NCONS_MAX], Fr[NCONS_MAX];
    Real vml, vpl, vmr, vpr;

    prim_to_flux(d, k, j, i, ql, vml, vpl, Ul, Fl);
    prim_to_flux(d, k, j, i, qr, vmr, vpr, Ur, Fr);

    const Real cmax = std::max(std::max(-vml,vpl), std::max(-vmr,vpr));

    CellLocation loc = DirectionToFaceID(d);
    const Real gdet = geom.DetGamma(loc, k, j, i);
    for (int m = 0; m < ncons; m++) {
      v.flux(d,m,k,j,i) = 0.5*(Fl[m] + Fr[m] - cmax*(Ur[m] - Ul[m])) * gdet;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void hll(const int d, const int k, const int j, const int i) const {
    Real Ul[NCONS_MAX], Ur[NCONS_MAX];
    Real Fl[NCONS_MAX], Fr[NCONS_MAX];
    Real vml, vpl, vmr, vpr;

    prim_to_flux(d, k, j, i, ql, vml, vpl, Ul, Fl);
    prim_to_flux(d, k, j, i, qr, vmr, vpr, Ur, Fr);

    const Real cl = std::min(std::min(vml, vmr), 0.0);
    const Real cr = std::max(std::max(vpl, vpr), 0.0);

    CellLocation loc = DirectionToFaceID(d);
    const Real gdet = geom.DetGamma(loc, k, j, i);
    for (int m = 0; m < ncons; m++) {
      v.flux(d,m,k,j,i) = (cr*Fl[m] - cl*Fr[m] + cr*cl*(Ur[m] - Ul[m]))/(cr - cl) * gdet;
    }
  }
};

} // namespace fluid

#endif // FLUID_HPP_
