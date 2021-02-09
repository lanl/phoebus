#include "pgen/pgen.hpp"

// Optically thin cooling.
// As descriged in the bhlight test suite
// Ryan, B. R., Dolence, J. C., & Gammie, C. F. 2015, ApJ, 807, 31. doi:10.1088/0004-637X/807/1/31

namespace thin_cooling {

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

  const Real rho0 = pin->GetOrAddReal("thincooling", "rho0", 1.0);
  const Real T0 = pin->GetOrAddReal("thincooling", "T0", 1.0e6);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
    "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      const Real rho = rho0;
      const Real T = T0;
      const Real vel = 0.;
      v(irho, k, j, i) = rho;
      v(ieng, k, j, i) = eos.InternalEnergyFromDensityTemperature(rho0, T0);
      v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho0, v(ieng, k, j, i));
      v(itmp, k, j, i) = T0;
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
    });

  fluid::PrimitiveToConserved(rc.get());
}

}

//} // namespace phoebus
