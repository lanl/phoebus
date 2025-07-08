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

#include <cmath>

#include <utils/error_checking.hpp>

#include "pgen/pgen.hpp"

// Self-similar, relativistic blast wave,
// as constructed by Blandford and McKee, 1976

namespace blandford_mckee {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(
      typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski),
      "Problem \"blandford_mckee\" requires \"SphericalMinkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
                        fluid_prim::bfield, fluid_prim::ye, fluid_prim::pressure, 
                        fluid_prim::temperature, fluid_prim::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  const Real wshock = pin->GetOrAddReal("blandford_mckee", "lorentz_shock", 7);
  Real rescale = std::sqrt(1 - (1 / (wshock * wshock)));

  const Real rho0 = pin->GetOrAddReal("blandford_mckee", "rho0", 1e-2);
  const Real tshock = pin->GetOrAddReal("blandford_mckee", "tshock", 0.25);
  const Real P0 = pin->GetOrAddReal("blandford_mckee", "P0", 1e-4);
  PARTHENON_REQUIRE_THROWS(rescale <= 1, "Rescale must shrink velocities");

  auto &coords = pmb->coords;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::blandford_mckee", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real r = coords.Xc<1>(i);
        Real vel = (std::abs(r) < tshock) ? rescale * r / tshock : 0;
        PARTHENON_REQUIRE(vel < 1, "Velocity subluminal");
        Real W = 1 / std::sqrt(1 - vel * vel);
        Real P = rescale * rescale * P0 * std::pow(W / tshock, 4.0);
        Real rho = rho0 * std::pow(W / tshock, 3.0);

        Real u = phoebus::energy_from_rho_P(eos, rho, P, emin, emax);
        Real eps = u / rho;

        Real eos_lambda[2];
        if (v.Contains(0, fluid_prim::ye())) {
          v(0, fluid_prim::ye(), k, j, i) = 0.5;
          eos_lambda[0] = v(0, fluid_prim::ye(), k, j, i);
        }
        Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, eos_lambda);

        v(0, fluid_prim::density(), k, j, i) = rho;
        v(0, fluid_prim::pressure(), k, j, i) = P;
        v(0, fluid_prim::energy(), k, j, i) = u;
        v(0, fluid_prim::temperature(), k, j, i) = T;
        v(0, fluid_prim::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, fluid_prim::density(), k, j, i), v(0, fluid_prim::temperature(), k, j, i), eos_lambda) /
                           v(0, fluid_prim::pressure(), k, j, i);
        for (int d = 0; d < 3; d++)
          v(0, fluid_prim::velocity(d), k, j, i) = 0.0;
        v(0, fluid_prim::velocity(0), k, j, i) = W * vel;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace blandford_mckee
