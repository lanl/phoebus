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
  namespace p = fluid_prim;
  const bool is_minkowski = (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski));
  const bool is_snake = (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake));
  PARTHENON_REQUIRE(is_minkowski || is_snake,
                    "Problem \"rotor\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<p::density, p::velocity, p::energy,
                        p::bfield, p::ye, p::pressure, 
                        p::temperature, p::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

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
        if (v.Contains(0, p::ye())) {
          v(0, p::ye(), k, j, i) = sin(2.0 * M_PI * x);
          eos_lambda[0] = v(0, p::ye(), k, j, i);
        }

        v(0, p::density(), k, j, i) = rho;
        v(0, p::pressure(), k, j, i) = P;
        v(0, p::energy(), k, j, i) =
            phoebus::energy_from_rho_P(eos, rho, P, emin, emax, eos_lambda[0]);
        v(0, p::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(0, p::energy(), k, j, i) / rho,
            eos_lambda); // this doesn't have to be exact, just a reasonable guess
        v(0, p::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, p::density(), k, j, i), v(0, p::temperature(), k, j, i), eos_lambda) /
                           v(0, p::pressure(), k, j, i);

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
        v(0, p::velocity(0), k, j, i) = ucon_transformed[1] + Gamma * shift[0] / alpha;
        v(0, p::velocity(1), k, j, i) = ucon_transformed[2] + Gamma * shift[1] / alpha;
        v(0, p::velocity(2), k, j, i) = ucon_transformed[3] + Gamma * shift[2] / alpha;
        for (int d = 0; d < 3; ++d) {
          v(0, p::bfield(d), k, j, i) = bcon_transformed[d + 1] * Gamma -
                          alpha * bcon_transformed[0] * ucon_transformed[d + 1];
          // v(0, p::bfield(d), k, j, i) = B_mink[d];
        }
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace rotor

//} // namespace phoebus
