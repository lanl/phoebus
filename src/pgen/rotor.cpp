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

// From Zanotti et al., 2015
// Solving the relativistic magnetohydrodynamics equations with ADER
// discontinuous Galerkin methods, a posteriori subcell limiting
// and adaptive mesh refinement

#include "pgen/pgen.hpp"

using Geometry::NDFULL;
using Geometry::NDSPACE;

namespace rotor {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const bool is_minkowski = (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski));
  const bool is_snake = (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake));
  PARTHENON_REQUIRE(is_minkowski || is_snake,
                    "Problem \"rotor\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                              fluid_prim::energy::name(), fluid_prim::bfield::name(),
                              fluid_prim::ye::name(), fluid_prim::pressure,
                              fluid_prim::temperature, fluid_prim::gamma1},
                             imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  const Real rho0 = pin->GetOrAddReal("rotor", "rho0", 10.0);
  const Real rho1 = pin->GetOrAddReal("rotor", "rho1", 1.0);
  const Real r0 = pin->GetOrAddReal("rotor", "r0", 0.1);
  const Real press = pin->GetOrAddReal("rotor", "press", 1.0);
  const Real omega = pin->GetOrAddReal("rotor", "omega", 9.95);
  const Real B0 = pin->GetOrAddReal("rotor", "B0", 1.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");
  auto gpkg = pmb->packages.Get("geometry");
  auto geom = Geometry::GetCoordinateSystem(rc.get());
  Real a_snake, k_snake, alpha, betay;
  alpha = 1;
  a_snake = k_snake = betay = 0;
  Real tf = pin->GetReal("parthenon/time", "tlim");
  if (is_snake) {
    a_snake = gpkg->Param<Real>("a");
    k_snake = gpkg->Param<Real>("k");
    alpha = gpkg->Param<Real>("alpha");
    betay = gpkg->Param<Real>("vy");
    PARTHENON_REQUIRE_THROWS(alpha > 0, "lapse must be positive");

    tf /= alpha;
  }
  pin->SetReal("parthenon/time", "tlim", tf);

  pmb->par_for(
      "Phoebus::ProblemGenerator::Rotor", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        if (is_snake) {
          y = y - a_snake * sin(k_snake * x);
        }

        const Real r = std::sqrt(x * x + y * y);
        const Real rho = r < r0 ? rho0 : rho1;
        const Real P = press;
        const Real w = r < r0 ? omega : 0.0;

        const Real vx = -y * w;
        const Real vy = x * w;
        Real Gamma = 1.0 / sqrt(1.0 - vx * vx - vy * vy);

        Real eos_lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = sin(2.0 * M_PI * x);
          eos_lambda[0] = v(iye, k, j, i);
        }

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) =
            phoebus::energy_from_rho_P(eos, rho, P, emin, emax, eos_lambda[0]);
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(ieng, k, j, i) / rho,
            eos_lambda); // this doesn't have to be exact, just a reasonable guess
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                           v(iprs, k, j, i);

        Real u_mink[] = {Gamma, Gamma * vx, Gamma * vy, 0.0};
        Real B_mink[3] = {B0, 0.0, 0.0};

        Real Bdotv = 0.0;
        SPACELOOP(m) Bdotv += B_mink[m] * u_mink[m + 1] / Gamma;
        Real bcon[4] = {Gamma * Bdotv, 0.0, 0.0, 0.0};
        SPACELOOP(m) bcon[m + 1] = (B_mink[m] + bcon[0] * u_mink[m + 1]) / Gamma;

        Real gcov[NDFULL][NDFULL] = {0};
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        Real shift[NDSPACE];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

        Real J[NDFULL][NDFULL] = {0};
        if (is_snake) {
          J[0][0] = 1 / alpha;
          J[2][0] = -betay / alpha;
          J[2][1] = a_snake * k_snake * cos(k_snake * x);
          J[1][1] = J[2][2] = J[3][3] = 1;
        } else if (is_minkowski) {
          J[0][0] = J[1][1] = J[2][2] = J[3][3] = 1.0;
        }

        Real ucon_transformed[NDFULL] = {0, 0, 0, 0};
        SPACETIMELOOP(mu) SPACETIMELOOP(nu) {
          ucon_transformed[mu] += J[mu][nu] * u_mink[nu];
        }
        Real bcon_transformed[NDFULL] = {0, 0, 0, 0};
        SPACETIMELOOP(mu) SPACETIMELOOP(nu) {
          bcon_transformed[mu] += J[mu][nu] * bcon[nu];
        }

        Gamma = alpha * ucon_transformed[0];
        v(ivlo, k, j, i) = ucon_transformed[1] + Gamma * shift[0] / alpha;
        v(ivlo + 1, k, j, i) = ucon_transformed[2] + Gamma * shift[1] / alpha;
        v(ivlo + 2, k, j, i) = ucon_transformed[3] + Gamma * shift[2] / alpha;
        for (int d = ib_lo; d <= ib_hi; d++) {
          v(d, k, j, i) = bcon_transformed[d - ib_lo + 1] * Gamma -
                          alpha * bcon_transformed[0] * ucon_transformed[d - ib_lo + 1];
          // v(d, k, j, i) = B_mink[d-ib_lo];
        }
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace rotor

//} // namespace phoebus
