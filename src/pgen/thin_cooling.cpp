//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include "pgen/pgen.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "radiation/radiation.hpp"
#include "utils/constants.hpp"

// Optically thin cooling.
// As described in the bhlight test suite
// Ryan, B. R., Dolence, J. C., & Gammie, C. F. 2015, ApJ, 807, 31.
// doi:10.1088/0004-637X/807/1/31

namespace thin_cooling {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
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

  //const Real ne0 = pin->GetOrAddReal("thincooling", "ne0", 5.858732e+07);
  //const Real rho0 = ne0 * pc.mp * unit_conv.GetMassDensityCGSToCode();
  //const Real T0 =
  //    pin->GetOrAddReal("thincooling", "T0", 1.e8) * unit_conv.GetTemperatureCGSToCode();

  const Real rho0 = 1.e6 * unit_conv.GetMassDensityCGSToCode();
  const Real u0 = 1.e20*unit_conv.GetEnergyCGSToCode()*unit_conv.GetNumberDensityCGSToCode();

  pmb->par_for(
      "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.x1v(i);
        v(irho, k, j, i) = rho0;
        v(ieng, k, j, i) = u0;
        v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho0, u0/rho0);
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho0, u0/rho0);
        //v(iprs, k, j, i) = eos.PressureFromDensityTemperature(rho0, T0);
        //v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho0, v(iprs, k, j, i));
        //v(itmp, k, j, i) = T0;
        v(iye, k, j, i) = 0.5; // TODO(BRR) change depending on species

        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace thin_cooling

//} // namespace phoebus
