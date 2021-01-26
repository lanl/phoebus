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

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/*template <typename T>
TaskStatus PrimitiveToConserved(T *rc);

template <typename T>
TaskStatus ConservedToPrimitive(T *rc);

*/
#define DELTA(i,j) (i==j ? 1 : 0)

KOKKOS_FUNCTION
void prim_to_flux(const int b, const int d, const int k, const int j, const int i,
                  const Geometry::CoordinateSystem &geom, const ParArrayND<Real> &q,
                  Real &cs, Real *U, Real *F) {
  const int dir = d-1;
  const Real &rho = q(b,dir,0,k,j,i);
  const Real vcon[] = {q(b,dir,1,k,j,i), q(b,dir,2,k,j,i), q(b,dir,3,k,j,i)};
  const Real &v = vcon[dir];
  const Real &u = q(b,dir,4,k,j,i);
  const Real &P = q(b,dir,5,k,j,i);
  const Real &gm1 = q(b,dir,6,k,j,i);
  cs = sqrt(gm1*P/rho);

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
  U[0] = rho*W;

  // conserved momentum
  Real H = rho + u + P;
  for (int m = 1; m <= 3; m++) {
    U[m] = H*W*W*vcov[m-1];
  }

  // conserved energy
  U[4] = H*W*W - P - U[0];

  // Get fluxes
  const Real alpha = geom.Lapse(d, k, j, i);
  for (int m = 1; m <= 3; m++) {
    const Real vtil = vcon[m] - geom.ContravariantShift(m, loc, k, j, i)/alpha;

    // mass flux
    F[0] = U[0]*vtil;

    // momentum flux
    for (int n = 1; n <= 3; n++) {
      F[n] = U[n]*vtil + P*DELTA(d,n);
    }

    // energy flux
    F[4] = U[4]*vtil + P*v;
  }

  return;
}

#undef DELTA



template <typename T>
KOKKOS_INLINE_FUNCTION
void llf(const int b, const int d, const int k, const int j, const int i,
         const Geometry::CoordinateSystem &geom, const ParArrayND<Real> &ql,
         const ParArrayND<Real> &qr, T &v) {

  Real Ul[5], Ur[5];
  Real Fl[5], Fr[5];
  Real cl, cr;

  prim_to_flux(b, d, k, j, i, geom, ql, cl, Ul, Fl);
  prim_to_flux(b, d, k, j, i, geom, qr, cr, Ur, Fr);

  const Real cmax = (cl > cr ? cl : cr);

  CellLocation loc = DirectionToFaceID(d);
  const Real gdet = geom.DetGamma(loc, k, j, i);
  for (int m = 0; m < 5; m++) {
    v.flux(d,m,k,j,i) = 0.5*(Fl[m] + Fr[m] - cmax*(Ur[m] - Ul[m])) * gdet;
  }
}



} // namespace fluid

#endif // FLUID_HPP_
