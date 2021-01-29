#include <eos/eos.hpp>
#include <utils/error_checking.hpp>

#include "pgen/pgen.hpp"

#include "fluid/fluid.hpp"

// Single-material blast wave.
// As descriged in the Athena test suite
// https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
// and in
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., & Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000). 

namespace phoebus {

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

  const Real rhol = 10.0;
  const Real Pl = 40.0/3.0;
  const Real rhor = 1.0;
  const Real Pr = 1.e-3;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;

  pmb->par_for(
    "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      const Real rho = x < 0.5 ? rhol : rhor;
      const Real P = x < 0.5 ? Pl : Pr;
      v(irho, k, j, i) = rho;
      v(iprs, k, j, i) = P;
      v(ieng, k, j, i) = P/(gam - 1.0);
      v(itmp, k, j, i) = v(ieng, k, j, i)/rho * cv; // this doesn't have to be exact, just a reasonable guess
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
    });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace Riot
