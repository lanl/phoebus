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

  namespace p = fluid_prim;
  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<p::density, p::velocity, p::energy,
                        p::bfield, p::ye, p::pressure, 
                        p::temperature, p::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::rhs_tester", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = std::abs(coords.Xc<1>(i));
        const Real rho = 1;
        const Real P = x / 2;
        const Real vel = x / 2;

        Real eos_lambda[2];
        if (v.Contains(0, p::ye())) {
          v(0, p::ye(), k, j, i) = 0.5;
          eos_lambda[0] = v(0, p::ye(), k, j, i);
        }

        v(0, p::density(), k, j, i) = rho;
        v(0, p::pressure(), k, j, i) = P;
        v(0, p::energy(), k, j, i) =
            phoebus::energy_from_rho_P(eos, rho, P, emin, emax, eos_lambda[0]);

        v(0, p::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(0, p::energy(), k, j, i) / rho, eos_lambda);
        v(0, p::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, p::density(), k, j, i), v(0, p::temperature(), k, j, i), eos_lambda) /
                           v(0, p::pressure(), k, j, i);
        for (int d = 0; d < 3; ++d)
          v(0, p::velocity(d), k, j, i) = 0.0;
        v(0, p::velocity(0), k, j, i) = vel;
      });
  fluid::PrimitiveToConserved(rc.get());
}

} // namespace rhs_tester
