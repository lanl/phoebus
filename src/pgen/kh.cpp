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

#include "Kokkos_Random.hpp"
#include "pgen/pgen.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace kelvin_helmholtz {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"kh\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
                        fluid_prim::bfield, fluid_prim::ye, fluid_prim::pressure, 
                        fluid_prim::temperature, fluid_prim::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  const Real rho0 = pin->GetOrAddReal("kelvin_helmholtz", "rho0", 1.0);
  const Real rho1 = pin->GetOrAddReal("kelvin_helmholtz", "rho1", 1.0);
  const Real P0 = pin->GetOrAddReal("kelvin_helmholtz", "P0", 1.0);
  const Real P1 = pin->GetOrAddReal("kelvin_helmholtz", "P1", 1.0);
  const Real v0 = pin->GetOrAddReal("kelvin_helmholtz", "v0", -0.5);
  const Real v1 = pin->GetOrAddReal("kelvin_helmholtz", "v1", 0.5);
  const Real v_pert = pin->GetOrAddReal("kelvin_helmholtz", "v_pert", 0.01);

  const Real Bx0 = pin->GetOrAddReal("kelvin_helmholtz", "Bx0", 0.0);
  const Real By0 = pin->GetOrAddReal("kelvin_helmholtz", "By0", 0.0);
  const Real Bz0 = pin->GetOrAddReal("kelvin_helmholtz", "Bz0", 0.0);
  const Real Bx1 = pin->GetOrAddReal("kelvin_helmholtz", "Bx1", 0.0);
  const Real By1 = pin->GetOrAddReal("kelvin_helmholtz", "By1", 0.0);
  const Real Bz1 = pin->GetOrAddReal("kelvin_helmholtz", "Bz1", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  auto geom = Geometry::GetCoordinateSystem(rc.get());

  RNGPool rng_pool(pin->GetOrAddInteger("kelvin_helmholtz", "seed", 37));

  pmb->par_for(
      "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();
        const Real x = coords.Xc<1>(i);
        const Real y = std::fabs(coords.Xc<2>(j));
        const Real rho = y < 0.25 ? rho1 : rho0;
        const Real P = y < 0.25 ? P1 : P0;
        const Real vel = y < 0.25 ? v1 : v0;

        Real gcov[4][4];
        geom.SpacetimeMetric(0.0, x, y, 0.0, gcov);

        Real eos_lambda[2];
        if (v.Contains(0, fluid_prim::ye())) {
          v(0, fluid_prim::ye(), k, j, i) = sin(2.0 * M_PI * x);
          eos_lambda[0] = v(0, fluid_prim::ye(), k, j, i);
        }

        v(0, fluid_prim::density(), k, j, i) = rho;
        v(0, fluid_prim::pressure(), k, j, i) = P;
        v(0, fluid_prim::energy(), k, j, i) =
            phoebus::energy_from_rho_P(eos, rho, P, emin, emax, eos_lambda[0]);
        v(0, fluid_prim::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(0, fluid_prim::energy(), k, j, i) / rho,
            eos_lambda); // this doesn't have to be exact, just a reasonable guess
        v(0, fluid_prim::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, fluid_prim::density(), k, j, i), v(0, fluid_prim::temperature(), k, j, i), eos_lambda) /
                           v(0, fluid_prim::pressure(), k, j, i);
        for (int d = 0; d < 3; d++)
          v(0, fluid_prim::velocity(d), k, j, i) = v_pert * 2.0 * (rng_gen.drand() - 0.5);
        v(0, fluid_prim::velocity(0), k, j, i) += vel;
        Real vsq = 0.;
        SPACELOOP2(ii, jj) {
          vsq += gcov[ii + 1][jj + 1] * v(0, fluid_prim::velocity(ii), k, j, i) * v(0, fluid_prim::velocity(jj), k, j, i);
        }
        const Real W = 1. / sqrt(1. - vsq);
        SPACELOOP(ii) { v(0, fluid_prim::velocity(ii), k, j, i) *= W; }
        if (v.Contains(0, fluid_prim::bfield(0))) {
          const Real Bx = y < 0.25 ? Bx1 : Bx0;
          const Real By = y < 0.25 ? By1 : By0;
          const Real Bz = y < 0.25 ? Bz1 : Bz0;
          v(0, fluid_prim::bfield(0), k, j, i) = Bx;
          v(0, fluid_prim::bfield(1), k, j, i) = By;
          v(0, fluid_prim::bfield(2), k, j, i) = Bz;
        }
        rng_pool.free_state(rng_gen);
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace kelvin_helmholtz

//} // namespace phoebus
