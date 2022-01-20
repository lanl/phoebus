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

namespace rhs_tester {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v =
      rc->PackVariables({fluid_prim::density, fluid_prim::velocity,
                         fluid_prim::energy, fluid_prim::bfield, fluid_prim::ye,
                         fluid_prim::pressure, fluid_prim::temperature},
                        imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::rhs_tester", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = std::abs(coords.x1v(i));
        const Real rho = 1;
        const Real P = x / 2;
        const Real vel = x / 2;

	Real eos_lambda[2];
	if (iye > 0) {
	  v(iye, k, j, i) = 0.5;
	  eos_lambda[0] = v(iye, k, j, i);
	}

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho, P, emin, emax, eos_lambda[0]);

        v(itmp, k, j, i) =
	  eos.TemperatureFromDensityInternalEnergy(rho,
						   v(ieng, k, j, i) / rho,
						   eos_lambda);
        for (int d = 0; d < 3; ++d)
          v(ivlo + d, k, j, i) = 0.0;
        v(ivlo, k, j, i) = vel;
      });
  fluid::PrimitiveToConserved(rc.get());
}

} // namespace rhs_tester
