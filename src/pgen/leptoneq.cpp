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
  auto v = rc->PackVariables({p::density::name(), p::velocity::name(), p::energy::name(),
                              p::ye::name(), p::pressure::name(), p::temperature::name(),
                              p::gamma1},
                             imap);

  const int irho = imap[p::density::name()].first;
  const int ivlo = imap[p::velocity::name()].first;
  const int ivhi = imap[p::velocity::name()].second;
  const int ieng = imap[p::energy::name()].first;
  const int iye = imap[p::ye::name()].first;
  const int iprs = imap[p::pressure::name()].first;
  const int itmp = imap[p::temperature::name()].first;
  const int igm1 = imap[p::gamma1].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eospkg = pmb->packages.Get("eos");
  auto eos = eospkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");

  const Real rho0 = 1.e10 * unit_conv.GetMassDensityCGSToCode();
  const Real T0 = 2.5 * 1.e6 * pc::eV / pc::kb * unit_conv.GetTemperatureCGSToCode();

  pmb->par_for(
      "Phoebus::ProblemGenerator::LeptonEq", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
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
        v(ieng, k, j, i) =
            rho0 * eos.InternalEnergyFromDensityTemperature(rho0, T0, lambda);
        v(iprs, k, j, i) = eos.PressureFromDensityTemperature(rho0, T0, lambda);
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);

        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace leptoneq
