#include "pgen/pgen.hpp"

// Single-material blast wave.
// As descriged in the Athena test suite
// https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
// and in
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., & Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000).

//namespace phoebus {

namespace shock_tube {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({"p.density",
                              "p.velocity",
                              "p.energy",
                              "p.ye",
                              "pressure",
                              "temperature",
                              "gamma1",
                              "cs"},
                              imap);

  const int irho = imap["p.density"].first;
  const int ivlo = imap["p.velocity"].first;
  const int ivhi = imap["p.velocity"].second;
  const int ieng = imap["p.energy"].first;
  const int iye  = imap["p.ye"].second;
  const int iprs = imap["pressure"].first;
  const int itmp = imap["temperature"].first;

  const Real rhol = pin->GetOrAddReal("shocktube", "rhol", 1.0);
  const Real Pl = pin->GetOrAddReal("shocktube", "Pl", 1.0);
  const Real vl = pin->GetOrAddReal("shocktube", "vl", 0.0);
  const Real rhor = pin->GetOrAddReal("shocktube", "rhor", 1.0);
  const Real Pr = pin->GetOrAddReal("shocktube", "Pr", 1.0);
  const Real vr = pin->GetOrAddReal("shocktube", "vr", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
    "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      const Real rho = x < 0.5 ? rhol : rhor;
      const Real P = x < 0.5 ? Pl : Pr;
      const Real vel = x < 0.5 ? vl : vr;
      v(irho, k, j, i) = rho;
      v(iprs, k, j, i) = P;
      v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho, P);
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho); // this doesn't have to be exact, just a reasonable guess
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
      v(ivlo, k, j, i) = vel;
      if (iye > 0) v(iye, k, j, i) = sin(2.0*M_PI*x);
      double rho1 = 3.0;
      double T1 = 2.5;
      double P1 = eos.PressureFromDensityTemperature(rho1, T1);
      double ug1 = phoebus::energy_from_rho_P(eos, rho1, P1);
      double sie1 = eos.InternalEnergyFromDensityTemperature(rho1, T1);
      printf("ug1 = %e rho1*sie1 = %e\n", ug1, rho1*sie1);
    });
  exit(-1);

  fluid::PrimitiveToConserved(rc.get());
}

}

//} // namespace phoebus
