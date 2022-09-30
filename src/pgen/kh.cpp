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

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density, fluid_prim::velocity, fluid_prim::energy, fluid_prim::bfield,
       fluid_prim::ye, fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1},
      imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

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
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  RNGPool rng_pool(pin->GetOrAddInteger("kelvin_helmholtz", "seed", 37));

  pmb->par_for(
      "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();
        const Real x = coords.x1v(i);
        const Real y = std::fabs(coords.x2v(j));
        const Real rho = y < 0.25 ? rho1 : rho0;
        const Real P = y < 0.25 ? P1 : P0;
        const Real vel = y < 0.25 ? v1 : v0;

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
        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = v_pert * 2.0 * (rng_gen.drand() - 0.5);
        v(ivlo, k, j, i) += vel;
        Real vsq = 0.;
        SPACELOOP2(ii, jj) { vsq += v(ivlo + ii, k, j, i) * v(ivlo + jj, k, j, i); }
        const Real W = 1. / sqrt(1. - vsq);
        SPACELOOP(ii) { v(ivlo + ii, k, j, i) *= W; }
        if (ib_hi > 0) {
          const Real Bx = y < 0.25 ? Bx1 : Bx0;
          const Real By = y < 0.25 ? By1 : By0;
          const Real Bz = y < 0.25 ? Bz1 : Bz0;
          v(ib_lo, k, j, i) = Bx;
          v(ib_lo + 1, k, j, i) = By;
          v(ib_hi, k, j, i) = Bz;
        }
        rng_pool.free_state(rng_gen);
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace kelvin_helmholtz

//} // namespace phoebus
