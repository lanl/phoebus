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

#include "pgen/pgen.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "radiation/radiation.hpp"
#include "utils/constants.hpp"

// 2D lepton equilibration.
// As described in the nubhlight test suite
// Miller, J. M., Ryan, B. R., & Dolence, J. C. 2019, ApJS, 241, 30
// doi:10.3847/1538-4365/ab09fc

namespace leptoneq {

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {p::density, p::velocity, p::energy, p::ye, p::pressure, p::temperature}, imap);

  const int irho = imap[p::density].first;
  const int ivlo = imap[p::velocity].first;
  const int ivhi = imap[p::velocity].second;
  const int ieng = imap[p::energy].first;
  const int iye  = imap[p::ye].first;
  const int iprs = imap[p::pressure].first;
  const int itmp = imap[p::temperature].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eospkg = pmb->packages.Get("eos");
  auto eos = eospkg->Param<singularity::EOS>("d.EOS");
  auto &unit_conv = eospkg.get()->Param<phoebus::UnitConversions>("unit_conv");

  const Real rho0 = 1.e10 * unit_conv.GetMassDensityCGSToCode();
  const Real T0 = 2.5*1.e6*pc::eV/pc::kb * unit_conv.GetTemperatureCGSToCode();
  printf("T: %e K\n", 2.5*1.e6*pc::eV/pc::kb);
  printf("mp c^2: %e k T: %e\n", pc::mp*pc::c*pc::c, 2.5e6*pc::eV);
  printf("ndens: %e\n", 1.e10/pc::mp);
  printf("rho K T / mu mp: %e\n",
    1.e10*pc::kb*(2.5*1.e6*pc::eV/pc::kb)/pc::mp);
  printf("rho c^2: %e\n", 1.e10*pc::c*pc::c);
  Real U_unit = unit_conv.GetEnergyCGSToCode()/pow(unit_conv.GetLengthCGSToCode(),3);
  printf("U_unit: %e rhoc: %e uc: %e\n", U_unit, 1.e10*pc::c*pc::c*U_unit,
    1.e10*pc::kb*(2.5*1.e6*pc::eV/pc::kb)/pc::mp*U_unit);

  // Cv = du/dT
  // (gam - 1)u = n k T = rho k T / mp
  // u = rho k
  // T = sie / cv = u / ( rho * cv )
  // Cv = u / ( rho * T )
  // u / ( rho * T ) = k / ( (gam - 1) * mp )
  printf("cv: %e\n", pc::kb / ( (5./3. - 1.) * pc::mp));


  pmb->par_for(
      "Phoebus::ProblemGenerator::LeptonEq", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.x1v(i);
        const Real y = coords.x2v(j);
        v(irho, k, j, i) = rho0;
        v(itmp, k, j, i) = T0;

        if (x >= -0.75 && x <= -0.25 && y >= -0.75 && y <= -0.25) {
          v(iye, k, j, i) = 0.1;
        } else if (x >= 0.25 && x <= 0.75 && y >= 0.25 && y <= 0.75) {
          v(iye, k, j, i) = 0.35;
        } else {
          v(iye, k, j, i) = 0.225;
        }

        double lambda[2] = {v(iye, k, j, i), 0.};
        v(ieng, k, j, i) = rho0*eos.InternalEnergyFromDensityTemperature(rho0, T0, lambda);
        v(iprs, k, j, i) = eos.PressureFromDensityTemperature(rho0, T0, lambda);
        printf("rho: %e T: %e u: %e\n", rho0, T0, v(ieng,k,j,i));
        //exit(-1);

        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace leptoneq
