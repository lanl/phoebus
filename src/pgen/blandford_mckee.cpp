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

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                              fluid_prim::energy::name(), fluid_prim::bfield::name(),
                              fluid_prim::ye, fluid_prim::pressure,
                              fluid_prim::temperature, fluid_prim::gamma1},
                             imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

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
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }
        Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, eos_lambda);

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = u;
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                           v(iprs, k, j, i);
        for (int d = 0; d < 3; d++)
          v(ivlo + d, k, j, i) = 0.0;
        v(ivlo, k, j, i) = W * vel;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace blandford_mckee
