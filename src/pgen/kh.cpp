#include "pgen/pgen.hpp"
#include "Kokkos_Random.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace kelvin_helmholtz {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({primitive_variables::density,
                              primitive_variables::velocity,
                              primitive_variables::energy,
                              primitive_variables::bfield,
                              primitive_variables::ye,
                              primitive_variables::pressure,
                              primitive_variables::temperature},
                              imap);

  const int irho = imap[primitive_variables::density].first;
  const int ivlo = imap[primitive_variables::velocity].first;
  const int ivhi = imap[primitive_variables::velocity].second;
  const int ieng = imap[primitive_variables::energy].first;
  const int ib_lo = imap[primitive_variables::bfield].first;
  const int ib_hi = imap[primitive_variables::bfield].second;
  const int iye  = imap[primitive_variables::ye].second;
  const int iprs = imap[primitive_variables::pressure].first;
  const int itmp = imap[primitive_variables::temperature].first;

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
      v(irho, k, j, i) = rho;
      v(iprs, k, j, i) = P;
      v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho, P);
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho); // this doesn't have to be exact, just a reasonable guess
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = v_pert*2.0*(rng_gen.drand()-0.5);
      v(ivlo, k, j, i) += vel;
      if (ib_hi > 0) {
        const Real Bx = y < 0.25 ? Bx1 : Bx0;
        const Real By = y < 0.25 ? By1 : By0;
        const Real Bz = y < 0.25 ? Bz1 : Bz0;
        v(ib_lo, k, j, i) = Bx;
        v(ib_lo + 1, k, j, i) = By;
        v(ib_hi, k, j, i) = Bz;
      }
      if (iye > 0) v(iye, k, j, i) = sin(2.0*M_PI*x);
      rng_pool.free_state(rng_gen);
    });

  fluid::PrimitiveToConserved(rc.get());
}

}

//} // namespace phoebus
