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

// Optically thin neutrino cooling.
// As described in the nubhlight test suite
// Miller, J. M., Ryan, B. R., & Dolence, J. C. 2019, ApJS, 241, 30
// doi:10.3847/1538-4365/ab09fc

namespace thin_cooling {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

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
  const int iye  = imap[p::ye].second;
  const int iprs = imap[p::pressure].first;
  const int itmp = imap[p::temperature].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eospkg = pmb->packages.Get("eos");
  auto eos = eospkg->Param<singularity::EOS>("d.EOS");
  auto &unit_conv = eospkg.get()->Param<phoebus::UnitConversions>("unit_conv");

  const Real rho0 = 1.e6 * unit_conv.GetMassDensityCGSToCode();
  const Real u0 = 1.e20*unit_conv.GetEnergyCGSToCode()*unit_conv.GetNumberDensityCGSToCode();

  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const std::string rad_method = pin->GetString("radiation", "method");
  if (x1max > 1.e-7 && rad_method == "cooling_function") {
    PARTHENON_THROW("Set x1max = 1.e-7 for the cooling_function rad method to get small enough timesteps!");
  }

  pmb->par_for(
      "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.x1v(i);
	Real lambda[2] = {0.5, 0.};
	if (iye > 0) {
	  v(iye, k, j, i) = lambda[0];
	}

        v(irho, k, j, i) = rho0;
        v(ieng, k, j, i) = u0;
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho0, u0/rho0, lambda);
        v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho0, u0/rho0, lambda);
        v(iye, k, j, i) = 0.5; // TODO(BRR) change depending on species

        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace thin_cooling
