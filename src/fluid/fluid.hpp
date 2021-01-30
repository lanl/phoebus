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
using namespace parthenon::package::prelude;

#include <eos/eos.hpp>
#include "con2prim.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc);
template <typename T>
TaskStatus ConservedToPrimitive(T *rc);
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

class FluxState {
 public:
  FluxState(MeshBlockData<Real> *rc) : FluxState(rc, PackIndexMap()) {}

  KOKKOS_INLINE_FUNCTION
  void Solve(const int d, const int k, const int j, const int i) const {
    llf(d, k, j ,i);
  }

  static std::vector<std::string> ReconVars() {
    return std::vector<std::string>({primitive_variables::density,
                                     primitive_variables::velocity,
                                     primitive_variables::energy,
                                     primitive_variables::pressure,
                                     primitive_variables::gamma1});
  }
  static std::vector<std::string> FluxVars() {
    return std::vector<std::string>({conserved_variables::density,
                                     conserved_variables::momentum,
                                     conserved_variables::energy});
  }

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> &ql;
  const ParArrayND<Real> &qr;
 private:
  const Geometry::CoordinateSystem geom;
  const int prho, pvel_lo, pvel_hi, peng, prs, gm1, crho, cmom_lo, cmom_hi, ceng, ncons;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
    : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
      ql(rc->Get("ql").data),
      qr(rc->Get("qr").data),
      geom(Geometry::GetCoordinateSystem(rc)),
      prho(imap[primitive_variables::density].first),
      pvel_lo(imap[primitive_variables::velocity].first),
      pvel_hi(imap[primitive_variables::velocity].second),
      peng(imap[primitive_variables::energy].first),
      prs(imap[primitive_variables::pressure].first),
      gm1(imap[primitive_variables::gamma1].first),
      crho(prho), cmom_lo(pvel_lo), cmom_hi(pvel_hi), ceng(peng),
      ncons(ceng+1) {}



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
    Real Ul[ncons], Ur[ncons];
    Real Fl[ncons], Fr[ncons];
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
};

} // namespace fluid

#endif // FLUID_HPP_
