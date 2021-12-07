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

#include <array>
#include <cmath>

// Parthenon includes
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

// phoebus includes
#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "monopole_gr/monopole_gr_base.hpp"
#include "phoebus_utils/linear_algebra.hpp"

namespace Geometry {

/*
 * Metric from Monopole GR module.
 * Assumes spherical symmetry for the spacetime.
 * This class also assumes spherical coordinates for the fluid sector.

 * Since the metric and fluid radial grids don't necessarily align, we
 * interpolate the metric quantities and use the AnalyticSystem
 * modifier to fill in the missing functionality.
 */
class MonopoleCart;

class MonopoleSph {
 public:
  friend class MonopoleCart;

  MonopoleSph() = default;
  KOKKOS_INLINE_FUNCTION
  MonopoleSph(const MonopoleGR::Hypersurface_t &hypersurface,
              const MonopoleGR::Alpha_t &alpha, const MonopoleGR::Beta_t &beta,
              const MonopoleGR::Gradients_t &gradients, const MonopoleGR::Radius &rgrid)
      : hypersurface_(hypersurface), alpha_(alpha), beta_(beta), gradients_(gradients),
        rgrid_(rgrid) {}

  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    return MonopoleGR::Interpolate(r, alpha_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    const Real r = std::abs(X1);
    LinearAlgebra::SetZero(beta, NDSPACE);
    beta[0] = MonopoleGR::Interpolate(r, beta_, rgrid_);
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real a2 = a * a;

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -alpha * alpha + a2 * beta * beta;
    g[0][1] = g[1][0] = a2 * beta;
    g[1][1] = a2;
    g[2][2] = r2;
    g[3][3] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real ir2 = Utils::ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real ialpha2 = Utils::ratio(1., alpha * alpha);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);

    LinearAlgebra::SetZero(g, NDFULL, NDFULL);
    g[0][0] = -ialpha2;
    g[0][1] = g[1][0] = beta * ialpha2;
    g[1][1] = Utils::ratio(1., a * a) - beta * beta * ialpha2;
    g[2][2] = ir2;
    g[3][3] = ir2 * Utils::ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    const Real r = std::abs(X1);
    const Real r2 = r * r;
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = a * a;
    g[1][1] = r2;
    g[2][2] = r2 * sth * sth;
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    const Real r = std::abs(X1);
    const Real ir2 = Utils::ratio(1., r * r);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    LinearAlgebra::SetZero(g, NDSPACE, NDSPACE);
    g[0][0] = Utils::ratio(1., a * a);
    g[1][1] = ir2;
    g[2][2] = ir2 * Utils::ratio(1., sth * sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    return a * r * r * std::abs(sth);
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    const Real r = std::abs(X1);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    return alpha * DetGamma(X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    const Real r = std::abs(X1);
    const Real th = X2;
    const Real sth = std::sin(th);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real beta = MonopoleGR::Interpolate(r, beta_, rgrid_);
    const Real a =
        MonopoleGR::Interpolate(r, hypersurface_, rgrid_, MonopoleGR::Hypersurface::A);
    const Real dadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DADR);
    const Real dalphadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADR);
    const Real dbetadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DBETADR);
    const Real dadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DADT);
    const Real dalphadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADT);
    const Real dbetadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DBETADT);

    LinearAlgebra::SetZero(dg, NDFULL, NDFULL, NDFULL);
    // d/dt
    dg[0][0][0] =
        2 * a * beta * beta * dadt - 2 * alpha * dalphadt + 2 * a * a * beta * dbetadt;
    dg[1][0][0] = dg[0][1][0] = 2 * a * beta * dadt + a * a * dbetadt;
    dg[1][1][0] = 2 * a * dadt;
    // d/dr
    dg[0][0][1] = -2 * alpha * dalphadr + 2 * a * beta * (beta * dadr + a * dbetadr);
    dg[1][0][1] = dg[0][1][1] = a * (2 * beta * dadr + a * dbetadr);
    dg[1][1][1] = 2 * a * dadr;
    dg[2][2][1] = 2 * r;
    dg[3][3][1] = 2 * r * sth * sth;
    // d/dth
    dg[3][3][2] = r * r * std::sin(2 * th);
    // d/dph = 0
  }

  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    // This is less error prone than a hardcoded version.
    Utils::SetConnectionCoeffByFD(*this, Gamma, X0, X1, X2, X3);
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real alpha = MonopoleGR::Interpolate(r, alpha_, rgrid_);
    const Real dalphadr =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADR);
    const Real dalphadt =
        MonopoleGR::Interpolate(r, gradients_, rgrid_, MonopoleGR::Gradients::DALPHADT);

    LinearAlgebra::SetZero(da, NDFULL);
    da[0] = dalphadt / alpha;
    da[1] = dalphadr / alpha;
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    const Real r = std::abs(X1);
    const Real sth = std::sin(X2);
    const Real cth = std::cos(X2);
    const Real sph = std::sin(X3);
    const Real cph = std::cos(X3);
    C[0] = X0;
    C[1] = r * sth * cph;
    C[2] = r * sth * sph;
    C[3] = r * cth;
  }

 private:
  MonopoleGR::Hypersurface_t hypersurface_;
  MonopoleGR::Alpha_t alpha_;
  MonopoleGR::Beta_t beta_;
  MonopoleGR::Gradients_t gradients_;
  MonopoleGR::Radius rgrid_;
};

class MonopoleCart {
 public:
  MonopoleCart() = default;
  MonopoleCart(const MonopoleGR::Hypersurface_t &hypersurface,
               const MonopoleGR::Alpha_t &alpha, const MonopoleGR::Beta_t &beta,
               const MonopoleGR::Gradients_t &gradients, const MonopoleGR::Radius &rgrid)
      : sph_(hypersurface, alpha, beta, gradients, rgrid) {}
  KOKKOS_INLINE_FUNCTION

  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    return sph_.Lapse(X0, r, th, ph);
  }

  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3, Real beta[NDSPACE]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    const Real cth = std::cos(th);
    const Real sth = std::sin(th);
    const Real cph = std::cos(ph);
    const Real sph = std::sin(ph);
    const Real betar = MonopoleGR::Interpolate(r, sph_.beta_, sph_.rgrid_);

    LinearAlgebra::SetZero(beta, NDSPACE);
    beta[0] = betar * sth * cph;
    beta[1] = betar * sth * sph;
    beta[2] = betar * cth;
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3, Real g[NDFULL][NDFULL]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    Real gsph[NDFULL][NDFULL];
    Real J[NDFULL][NDFULL];
    sph_.SpacetimeMetric(X0, r, th, ph, gsph);
    C2S(X1, X2, X3, r, J);
    SPACETIMELOOP2(mup, nup) {
      g[mup][nup] = 0;
      SPACETIMELOOP2(mu, nu) { g[mup][nup] += J[mu][mup] * J[nu][nup] * gsph[mu][nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    Real gsph[NDFULL][NDFULL];
    Real J[NDFULL][NDFULL];
    sph_.SpacetimeMetricInverse(X0, r, th, ph, gsph);
    S2C(r, th, ph, J);
    SPACETIMELOOP2(mup, nup) {
      g[mup][nup] = 0;
      SPACETIMELOOP2(mu, nu) { g[mup][nup] += J[mup][mu] * J[nup][nu] * gsph[mu][nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    Real gsph[NDSPACE][NDSPACE];
    Real J[NDFULL][NDFULL];
    sph_.Metric(X0, r, th, ph, gsph);
    C2S(X1, X2, X3, r, J);
    SPACELOOP2(ip, jp) {
      g[ip][jp] = 0;
      SPACELOOP2(i, j) { g[ip][jp] += J[i + 1][ip + 1] * J[j + 1][jp + 1] * gsph[i][j]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3, Real g[NDSPACE][NDSPACE]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    Real gsph[NDSPACE][NDSPACE];
    Real J[NDFULL][NDFULL];
    sph_.MetricInverse(X0, r, th, ph, gsph);
    S2C(r, th, ph, J);
    SPACELOOP2(ip, jp) {
      g[ip][jp] = 0;
      SPACELOOP2(i, j) { g[ip][jp] += J[ip + 1][i + 1] * J[jp + 1][j + 1] * gsph[i][j]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    const Real a = MonopoleGR::Interpolate(r, sph_.hypersurface_, sph_.rgrid_,
                                           MonopoleGR::Hypersurface::A);
    return a;
  }

  KOKKOS_INLINE_FUNCTION
  Real DetG(Real X0, Real X1, Real X2, Real X3) const {
    const Real alpha = Lapse(X0, X1, X2, X3);
    return alpha * DetGamma(X0, X1, X2, X3);
  }

  /*
    Hessian formula for derivative of metric in new coordinate system:
    -----
    g_{mu' nu', sigma') = (d^2 x^mu/dx^mu' dx^sigma') (dx^nu/dx^nu') g_{mu nu}
                         + (d^2 x^nu/dx^nu' dx^sigma') (dx^mu/dx^mu') g_{mu nu}
                         + (dx^sigma/dx^sigma') (d/dx^sigma) g_{mu nu}
   */
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);
    Real g[NDFULL][NDFULL];
    sph_.SpacetimeMetric(X0, r, th, ph, g);

    Real J[NDFULL][NDFULL];
    C2S(X1, X2, X3, r, J);
    Real H[NDFULL][NDFULL][NDFULL];
    Hessian(X1, X2, X3, r, H);

    Real dgsph[NDFULL][NDFULL][NDFULL];
    sph_.MetricDerivative(X0, r, th, ph, dgsph);

    SPACETIMELOOP3(mup, nup, sp) {
      dg[mup][nup][sp] = 0;
      SPACETIMELOOP3(mu, nu, sig) {
        dg[mup][nup][sp] += H[mu][mup][sp] * J[nu][nup] * g[mu][nu] +
                            H[nu][nup][sp] * J[mu][mup] * g[mu][nu] +
                            J[sig][sp] * dgsph[mu][nu][sig];
      }
    }
  }

  /*
    Hessian formula for the transformation of a Christoffel symbol:
    -------
    Gamma_{sigma' mu' nu'} = (dx^sigma/dx_sigma') (dx^mu'/dx^mu)(dx^nu'/dx^nu)
    Gamma_{sigma mu nu}
                            + (d^2 x^rho/dx^mu' dx^nu') (dx^sigma)(dx^sigma')
  */
  // Or we can Just compute the combinatorics from the metric
  // derivative.
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Real dg[NDFULL][NDFULL][NDFULL];
    MetricDerivative(X0, X1, X2, X3, dg);
    for (int a = 0; a < NDFULL; ++a) {
      for (int b = 0; b < NDFULL; ++b) {
        for (int c = 0; c < NDFULL; ++c) {
          Gamma[a][b][c] = 0.5 * (dg[b][a][c] + dg[c][a][b] - dg[b][c][a]);
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    Real r, th, ph;
    Cart2Sph(X1, X2, X3, r, th, ph);

    const Real alpha = MonopoleGR::Interpolate(r, sph_.alpha_, sph_.rgrid_);
    const Real dalphadr = MonopoleGR::Interpolate(r, sph_.gradients_, sph_.rgrid_,
                                                  MonopoleGR::Gradients::DALPHADR);
    const Real dalphadt = MonopoleGR::Interpolate(r, sph_.gradients_, sph_.rgrid_,
                                                  MonopoleGR::Gradients::DALPHADT);

    Real J[NDFULL][NDFULL];
    C2S(X1, X2, X3, r, J);

    Real da_sph[NDFULL];
    LinearAlgebra::SetZero(da_sph, NDFULL);
    LinearAlgebra::SetZero(da, NDFULL);
    da[0] = dalphadt / alpha;
    da[1] = dalphadr / alpha;

    SPACETIMELOOP2(mup, mu) {
      SPACELOOP2(i, j) { da[mup] += J[mu][mup] * da[mu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    C[0] = X0;
    C[1] = X1;
    C[2] = X2;
    C[3] = X3;
  }

 private:
  MonopoleSph sph_;

  KOKKOS_INLINE_FUNCTION
  void Cart2Sph(Real X1, Real X2, Real X3, Real &r, Real &th, Real &ph) const {
    r = std::sqrt(X1 * X1 + X2 * X2 + X3 * X3);
    th = std::acos(Utils::ratio(X3, r));
    ph = std::atan2(X2, X1);
  }

  // These are dx^{mu'}/dx^{mu}
  // convention is mu' is first index, mu is second
  // S2C has x^{mu'} = {x,y,z}
  // C2S has x^{mu'} = {r,th,ph}
  KOKKOS_INLINE_FUNCTION
  void S2C(Real r, Real th, Real ph, Real J[NDFULL][NDFULL]) const {
    Real cth = std::cos(th);
    Real sth = std::sin(th);
    Real cph = std::cos(ph);
    Real sph = std::sin(ph);
    LinearAlgebra::SetZero(J, NDFULL, NDFULL);
    J[0][0] = 1;
    J[1][1] = sth * cph;
    J[1][2] = r * cth * cph;
    J[1][3] = -r * sth * sph;
    J[2][1] = sth * sph;
    J[2][2] = r * cth * sph;
    J[2][3] = r * sth * cph;
    J[3][1] = cth;
    J[3][2] = -r * sth;
    J[3][3] = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void C2S(Real x, Real y, Real z, Real r, Real J[NDFULL][NDFULL]) const {
    const Real r2 = r * r;
    const Real rho2 = x * x + y * y;
    const Real rho = std::sqrt(rho2);

    LinearAlgebra::SetZero(J, NDFULL, NDFULL);
    J[0][0] = 1;
    J[1][1] = Utils::ratio(x, r);
    J[1][2] = Utils::ratio(y, r);
    J[1][3] = Utils::ratio(z, r);
    J[2][1] = Utils::ratio(x * z, r2 * rho);
    J[2][2] = Utils::ratio(y * z, r2 * rho2);
    J[2][3] = -Utils::ratio(rho, r2);
    J[3][1] = -Utils::ratio(y, rho2);
    J[3][2] = Utils::ratio(x, rho2);
    J[3][3] = 0;
  }
  // Hessian
  // d^2 x^mu/dx^mu' dx^sigma'
  // where x^mu = {r,th,ph}
  // and x^mu' = {x,y,z}
  KOKKOS_INLINE_FUNCTION
  void Hessian(Real x, Real y, Real z, Real r, Real H[NDFULL][NDFULL][NDFULL]) const {
    const Real x2 = x * x;
    const Real y2 = y * y;
    const Real z2 = z * z;
    const Real x4 = x2 * x2;
    const Real y4 = y2 * y2;
    const Real r2 = r * r;
    const Real r3 = r2 * r;
    const Real r4 = r3 * r;
    const Real rho2 = x2 + y2;
    const Real rho = std::sqrt(rho2);
    const Real rho3 = rho2 * rho;
    const Real rho4 = rho3 * rho;

    const Real irho = Utils::ratio(1., rho);
    const Real irho3 = Utils::ratio(1., rho3);
    const Real irho4 = Utils::ratio(1., rho4);
    const Real ir3 = Utils::ratio(1., r3);
    const Real ir4 = Utils::ratio(1., r4);

    LinearAlgebra::SetZero(H, NDFULL, NDFULL, NDFULL);
    H[1][1][1] = (y2 + z2) * ir3;
    H[1][2][1] = -(x * y) * ir3;
    H[1][3][1] = -(x * z) * ir3;
    H[2][1][1] = (z * (-2 * x4 - x2 * y2 + y4 + y2 * z2)) * irho3 * ir4;
    H[2][2][1] = -(x * y * z * (3 * rho2 + z2)) * irho3 * ir4;
    H[2][3][1] = (x * (x2 + y2 - z2)) * irho * ir4;
    H[3][1][1] = (2 * x * y) * irho4;
    H[3][2][1] = (y2 - x2) * irho4;
    // H[3][3][1] = 0;

    H[1][1][2] = -(x * y) * ir3;
    H[1][2][2] = (x2 + z2) * ir3;
    H[1][3][2] = -(y * z) * ir3;
    H[2][1][2] = -(x * y * z * (3 * rho2 + z2)) * irho3 * ir4;
    H[2][2][2] = (z * (x4 - 2 * y4 + x2 * (z2 - y2))) * irho3 * ir4;
    H[2][3][2] = (y * (x2 + y2 - z2)) * irho * ir4;
    H[3][1][2] = (y2 - x2) * irho4;
    H[3][2][2] = -(2 * x * y) * irho4;
    // H[3][3][2] = 0;

    H[1][1][3] = -(x * z) * irho3;
    H[1][2][3] = -(y * z) * irho3;
    H[1][3][3] = (x2 + y2) * irho3;
    H[2][1][3] = x * (x2 + y2 - z2) * irho * ir4;
    H[2][2][3] = y * (x2 + y2 - z2) * irho * ir4;
    H[2][3][3] = 2 * rho * z * ir4;
    // H[3][*][3] = 0;
  }
};

using MplSphMeshBlock = Analytic<MonopoleSph, IndexerMeshBlock>;
using MplCartMeshBlock = Analytic<MonopoleCart, IndexerMeshBlock>;

using MplSphMesh = Analytic<MonopoleSph, IndexerMesh>;
using MplCartMesh = Analytic<MonopoleCart, IndexerMesh>;

template <>
void Initialize<MplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<MplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

// TODO(JMM): The cached machinery will need to be revisited
using CMplSphMeshBlock = CachedOverMeshBlock<Analytic<MonopoleSph, IndexerMeshBlock>>;
using CMplCartMeshBlock = CachedOverMeshBlock<Analytic<MonopoleCart, IndexerMeshBlock>>;

using CMplSphMesh = CachedOverMesh<Analytic<MonopoleSph, IndexerMesh>>;
using CMplCartMesh = CachedOverMesh<Analytic<MonopoleCart, IndexerMesh>>;

template <>
void Initialize<CMplSphMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);
template <>
void Initialize<CMplCartMeshBlock>(ParameterInput *pin, StateDescriptor *geometry);

} // namespace Geometry
