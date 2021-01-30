#include <eos/eos.hpp>
#include <utils/error_checking.hpp>

#include "eos/eos.hpp"

#include "pgen/pgen.hpp"

#include "fluid/fluid.hpp"

// Single-material blast wave.
// As descriged in the Athena test suite
// https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
// and in
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., & Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000). 

namespace phoebus {

KOKKOS_INLINE_FUNCTION
Real energy(const singularity::EOS &eos, const Real rho, const Real P) {
  Real eguessl = P/rho;
  Real Pguessl = eos.PressureFromDensityInternalEnergy(rho, eguessl);
  Real eguessr = eguessl;
  Real Pguessr = Pguessl;
  while (Pguessl > P) {
    eguessl /= 2.0;
    Pguessl = eos.PressureFromDensityInternalEnergy(rho, eguessl);
  }
  while (Pguessr < P) {
    eguessr *= 2.0;
    Pguessr = eos.PressureFromDensityInternalEnergy(rho, eguessr);
  }

  PARTHENON_REQUIRE_THROWS(Pguessr>P && Pguessl<P, "Pressure not bracketed");

  while (Pguessr - Pguessl > 1.e-10*P) {
    Real emid = 0.5*(eguessl + eguessr);
    Real Pmid = eos.PressureFromDensityInternalEnergy(rho, emid);
    if (Pmid < P) {
      eguessl = emid;
      Pguessl = Pmid;
    } else {
      eguessr = emid;
      Pguessr = Pmid;
    }
  }
  std::cout << Pguessl << " " << P << " " << Pguessr << std::endl;
  return 0.5*rho*(eguessl + eguessr);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({"p.density",
                              "p.velocity",
                              "p.energy",
                              "pressure",
                              "temperature",
                              "gamma1",
                              "cs"},
                              imap);

  const int irho = imap["p.density"].first;
  const int ivlo = imap["p.velocity"].first;
  const int ivhi = imap["p.velocity"].second;
  const int ieng = imap["p.energy"].first;
  const int iprs = imap["pressure"].first;
  const int itmp = imap["temperature"].first;

  const Real gam = pin->GetReal("eos", "Gamma");
  const Real cv  = pin->GetReal("eos", "Cv");

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
      v(ieng, k, j, i) = energy(eos, rho, P);
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho); // this doesn't have to be exact, just a reasonable guess
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
      v(ivlo, k, j, i) = vel;
    });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace Riot
