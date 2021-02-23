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

  const Real rhol = pin->GetOrAddReal("shocktube", "rhol", 1.0);
  const Real Pl = pin->GetOrAddReal("shocktube", "Pl", 1.0);
  const Real vl = pin->GetOrAddReal("shocktube", "vl", 0.0);
  const Real Bxl = pin->GetOrAddReal("shocktube", "Bxl", 0.0);
  const Real Byl = pin->GetOrAddReal("shocktube", "Byl", 0.0);
  const Real Bzl = pin->GetOrAddReal("shocktube", "Bzl", 0.0);
  const Real rhor = pin->GetOrAddReal("shocktube", "rhor", 1.0);
  const Real Pr = pin->GetOrAddReal("shocktube", "Pr", 1.0);
  const Real vr = pin->GetOrAddReal("shocktube", "vr", 0.0);
  const Real Bxr = pin->GetOrAddReal("shocktube", "Bxr", 0.0);
  const Real Byr = pin->GetOrAddReal("shocktube", "Byr", 0.0);
  const Real Bzr = pin->GetOrAddReal("shocktube", "Bzr", 0.0);

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
      if (ib_hi > 0) {
        const Real Bx = x < 0.5 ? Bxl : Bxr;
        const Real By = x < 0.5 ? Byl : Byr;
        const Real Bz = x < 0.5 ? Bzl : Bzr;
        v(ib_lo, k, j, i) = Bx;
        v(ib_lo + 1, k, j, i) = By;
        v(ib_hi, k, j, i) = Bz;
      }
      if (iye > 0) v(iye, k, j, i) = sin(2.0*M_PI*x);
    });

  fluid::PrimitiveToConserved(rc.get());
}

}

//} // namespace phoebus
