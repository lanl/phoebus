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

#include "pgen/pgen.hpp"

// Non-relativistic Sedov blast wave.
// As descriged in in the Castro test suite
// https://amrex-astro.github.io/Castro/docs/Verification.html

namespace sedov {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(
      typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski) ||
          typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski),
      "Problem \"sedov\" requires \"Minkowski\" or \"SphericalMinkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                              fluid_prim::energy::name(), fluid_prim::bfield::name(),
                              fluid_prim::ye::name(), fluid_prim::pressure,
                              fluid_prim::temperature, fluid_prim::gamma1},
                             imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  const Real rhoa = pin->GetOrAddReal("sedov", "rho_ambient", 1.0);
  const Real rinner = pin->GetOrAddReal("sedov", "rinner", 0.01);
  const bool spherical = pin->GetOrAddReal("sedov", "spherical_coords", true);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  Real Pa = pin->GetOrAddReal("sedov", "P_ambient", 1e-5);
  Real Eexp = pin->GetOrAddReal("sedov", "explosion_energy", 1);

  const Real v_inner = (4. / 3.) * M_PI * std::pow(rinner, 3.);
  const Real uinner = Eexp / v_inner;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::sedov", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real r;
        if (spherical) {
          r = std::abs(coords.Xc<1>(i));
        } else {
          Real x = coords.Xc<1>(i);
          Real y = ndim > 1 ? coords.Xc<2>(j) : 0;
          Real z = ndim > 2 ? coords.Xc<3>(k) : 0;
          r = std::sqrt(x * x + y * y + z * z);
        }
        const Real rho = rhoa;

        Real lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          lambda[0] = v(iye, k, j, i);
        }

        const Real ua = phoebus::energy_from_rho_P(eos, rho, Pa, emin, emax, lambda[0]);
        const Real u = (r <= rinner) ? uinner : ua;

        const Real eps = u / (rho + 1e-20);
        const Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(rho, eps, lambda);

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = u;
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);
        for (int d = ivlo; d <= ivhi; d++)
          v(d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace sedov
