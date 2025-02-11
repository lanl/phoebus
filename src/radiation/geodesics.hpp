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

#ifndef RADIATION_GEODESICS_HPP_
#define RADIATION_GEODESICS_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

#include "geometry/geometry.hpp"

namespace radiation {

using Geometry::NDFULL;

KOKKOS_INLINE_FUNCTION
void GetXSource(Real &Kcon0, Real &Kcon1, Real &Kcon2, Real &Kcon3, Real src[NDFULL]) {
  src[0] = 1.;
  src[1] = Kcon1 / Kcon0;
  src[2] = Kcon2 / Kcon0;
  src[3] = Kcon3 / Kcon0;
}

template <class Geom>
KOKKOS_INLINE_FUNCTION void GetKSource(Real &X0, Real &X1, Real &X2, Real &X3,
                                       Real &Kcov0, Real &Kcov1, Real &Kcov2, Real &Kcov3,
                                       Real &Kcon0, Geom &geom, Real source[4]) {
  // TODO(BLB): include options for more geometries to exploit spacetime symmetries
  if constexpr (std::is_same<PHOEBUS_GEOMETRY,
                             Geometry::Minkowski>::value) { // flat space
    SPACETIMELOOP(mu) { source[mu] = 0.0; }
  } else { // generic
    // TODO(BLB): hook into metric machinery instead of doing it here
    Real Kcov[NDFULL] = {Kcov0, Kcov1, Kcov2, Kcov3};
    Real Xm[NDFULL], Xp[NDFULL];
    Real Gconm[NDFULL][NDFULL], Gconp[NDFULL][NDFULL];
    Real dG;

    constexpr static Real DELTA = 1.0e-6;

    SPACETIMELOOP(mu) {
      source[mu] = 0.;
      Xp[0] = X0;
      Xp[1] = X1;
      Xp[2] = X2;
      Xp[3] = X3;
      SPACETIMELOOP(nu) { Xm[nu] = Xp[nu]; }

      Xm[mu] -= DELTA;
      Xp[mu] += DELTA;

      geom.SpacetimeMetricInverse(Xm[0], Xm[1], Xm[2], Xm[3], Gconm);
      geom.SpacetimeMetricInverse(Xp[0], Xp[1], Xp[2], Xp[3], Gconp);

      SPACETIMELOOP2(nu, kap) {
        dG = (Gconp[nu][kap] - Gconm[nu][kap]) / (Xp[mu] - Xm[mu]);
        source[mu] += Kcov[nu] * Kcov[kap] * dG;
      }

      source[mu] *= -1.0 / (2.0 * Kcon0);
    }
  }
}

template <class Geom>
KOKKOS_INLINE_FUNCTION void PushParticle(Real &X0, Real &X1, Real &X2, Real &X3,
                                         Real &Kcov0, Real &Kcov1, Real &Kcov2,
                                         Real &Kcov3, const Real &dt, Geom &geom) {
  Real c1[NDFULL], c2[NDFULL], d1[NDFULL], d2[NDFULL];
  Real Xtmp[NDFULL], Kcontmp[NDFULL], Kcovtmp[NDFULL];
  Real Kcov[NDFULL] = {Kcov0, Kcov1, Kcov2, Kcov3};
  Real Gcon[NDFULL][NDFULL];

  // First stage
  geom.SpacetimeMetricInverse(X0, X1, X2, X3, Gcon);
  Geometry::Utils::Raise(Kcov, Gcon, Kcontmp);

  GetXSource(Kcontmp[0], Kcontmp[1], Kcontmp[2], Kcontmp[3], c1);
  GetKSource(X0, X1, X2, X3, Kcov0, Kcov1, Kcov2, Kcov3, Kcontmp[0], geom, d1);

  Xtmp[0] = X0 + dt * c1[0];
  Xtmp[1] = X1 + dt * c1[1];
  Xtmp[2] = X2 + dt * c1[2];
  Xtmp[3] = X3 + dt * c1[3];

  Kcovtmp[0] = Kcov0 + dt * d1[0];
  Kcovtmp[1] = Kcov1 + dt * d1[1];
  Kcovtmp[2] = Kcov2 + dt * d1[2];
  Kcovtmp[3] = Kcov3 + dt * d1[3];

  // Second stage
  geom.SpacetimeMetricInverse(Xtmp[0], Xtmp[1], Xtmp[2], Xtmp[3], Gcon);
  Geometry::Utils::Raise(Kcovtmp, Gcon, Kcontmp);

  GetXSource(Kcontmp[0], Kcontmp[1], Kcontmp[2], Kcontmp[3], c2);
  GetKSource(X0, X1, X2, X3, Kcovtmp[0], Kcovtmp[1], Kcovtmp[2], Kcovtmp[3], Kcontmp[0],
             geom, d2);

  X0 += 0.5 * dt * (c1[0] + c2[0]);
  X1 += 0.5 * dt * (c1[1] + c2[1]);
  X2 += 0.5 * dt * (c1[2] + c2[2]);
  X3 += 0.5 * dt * (c1[3] + c2[3]);

  Kcov0 += 0.5 * dt * (d1[0] + d2[0]);
  Kcov1 += 0.5 * dt * (d1[1] + d2[1]);
  Kcov2 += 0.5 * dt * (d1[2] + d2[2]);
  Kcov3 += 0.5 * dt * (d1[3] + d2[3]);
}

} // namespace radiation

#endif // RADIATION_GEODESICS_HPP_
