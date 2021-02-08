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
  const int prho, pvel_lo, pvel_hi, peng, pye, prs, gm1, crho, cmom_lo, cmom_hi, ceng, cye, ncons;
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
      pye(imap[primitive_variables::ye].second),
      prs(imap[primitive_variables::pressure].first),
      gm1(imap[primitive_variables::gamma1].first),
      crho(imap[conserved_variables::density].first),
      cmom_lo(imap[conserved_variables::momentum].first),
      cmom_hi(imap[conserved_variables::momentum].second),
      ceng(imap[conserved_variables::energy].first),
      cye(imap[conserved_variables::ye].first),
      ncons(5+(cye>0)) {
    PARTHENON_REQUIRE_THROWS(ncons <= NCONS_MAX, "ncons exceeds NCONS_MAX.  Reconfigure to increase NCONS_MAX.");
  }



  KOKKOS_FORCEINLINE_FUNCTION
  int Delta(const int i, const int j) const {
    return i==j;
  }

  KOKKOS_INLINE_FUNCTION
  void prim_to_flux(const int d, const int k, const int j, const int i,
                    const ParArrayND<Real> &q, Real &vel, Real &cs, Real *U, Real *F) const {
    const int dir = d-1;
    const Real &rho = q(dir,prho,k,j,i);
    const Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_hi,k,j,i)};
    vel = vcon[dir];
    const Real &u = q(dir,peng,k,j,i);
    const Real &P = q(dir,prs,k,j,i);
    const Real &gamma1 = q(dir,gm1,k,j,i);
    cs = sqrt(gamma1*P/rho);

    CellLocation loc = DirectionToFaceID(d);

    Real vcov[3];
    for (int m = 1; m <= 3; m++) {
      vcov[m-1] = 0.0;
      for (int n = 1; n <= 3; n++) {
        vcov[m-1] += geom.Metric(m,n,loc,k,j,i)*vcon[n-1];
      }
    }

    // get Lorentz factor
    Real vsq = 0.0;
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        vsq += vcon[m]*vcov[n];
      }
    }
    const Real W = 1.0/sqrt(1.0 - vsq);

    // conserved density
    U[crho] = rho*W;
    if (cye>0) U[cye] = U[crho]*q(dir,pye,k,j,i);

    // conserved momentum
    const Real rhohWsq = (rho + u + P)*W*W;
    for (int m = 0; m < 3; m++) {
      U[cmom_lo+m] = rhohWsq*vcov[m];
    }

    // conserved energy
    U[ceng] = rhohWsq - P - U[crho];

    // Get fluxes
    const Real alpha = geom.Lapse(d, k, j, i);
    const Real vtil = vcon[dir] - geom.ContravariantShift(d, loc, k, j, i)/alpha;

    // mass flux
    F[crho] = U[crho]*vtil;
    if (cye>0) F[cye] = U[cye]*vtil;

    // momentum flux
    for (int n = 0; n < 3; n++) {
      F[cmom_lo+n] = U[cmom_lo+n]*vtil + P*Delta(dir,n);
    }

    // energy flux
    F[ceng] = U[ceng]*vtil + P*vel;

    return;
  }

  KOKKOS_INLINE_FUNCTION
  void llf(const int d, const int k, const int j, const int i) const {
    Real Ul[NCONS_MAX], Ur[NCONS_MAX];
    Real Fl[NCONS_MAX], Fr[NCONS_MAX];
    Real vl, cl, vr, cr;

    prim_to_flux(d, k, j, i, ql, vl, cl, Ul, Fl);
    prim_to_flux(d, k, j, i, qr, vr, cr, Ur, Fr);

    Real cmax = (cl > cr ? cl : cr);
    cmax += std::max(std::abs(vl), std::abs(vr));

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
    Real vl, cl, vr, cr;

    prim_to_flux(d, k, j, i, ql, vl, cl, Ul, Fl);
    prim_to_flux(d, k, j, i, qr, vr, cr, Ur, Fr);

    cr = std::max(vr + cr,0.0);
    cl = std::min(vl - cl,0.0);

    Real cmax = (cl > cr ? cl : cr);
    cmax += std::max(std::abs(vl), std::abs(vr));

    CellLocation loc = DirectionToFaceID(d);
    const Real gdet = geom.DetGamma(loc, k, j, i);
    for (int m = 0; m < ncons; m++) {
      v.flux(d,m,k,j,i) = (cr*Fl[m] - cl*Fr[m] + cr*cl*(Ur[m] - Ul[m]))/(cr - cl) * gdet;
    }
  }
};

} // namespace fluid

#endif // FLUID_HPP_
