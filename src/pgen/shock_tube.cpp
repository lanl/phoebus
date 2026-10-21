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

// Single-material blast wave.
// As descriged in the Athena test suite
// https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
// and in
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., &
// Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000).

// namespace phoebus {

namespace shock_tube {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski) ||
                        typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski),
                    "Problem \"shock_tube\" requires \"Minkowski\" or "
                    "\"SphericalMinkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density::name(), fluid_prim::velocity::name(),
       fluid_prim::energy::name(), fluid_prim::ye::name(), fluid_prim::pressure::name(),
       fluid_prim::temperature::name(), fluid_prim::gamma1::name()},
      imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure::name()].first;
  const int itmp = imap[fluid_prim::temperature::name()].first;
  const int igm1 = imap[fluid_prim::gamma1::name()].first;

  const Real rhol = pin->GetOrAddReal("shocktube", "rhol", 1.0);
  const Real Pl = pin->GetOrAddReal("shocktube", "Pl", 1.0);
  const Real vl = pin->GetOrAddReal("shocktube", "vl", 0.0);
  const Real vyl = pin->GetOrAddReal("shocktube", "vyl", 0.0);
  const Real vzl = pin->GetOrAddReal("shocktube", "vzl", 0.0);
  const Real Bxl = pin->GetOrAddReal("shocktube", "Bxl", 0.0);
  const Real Byl = pin->GetOrAddReal("shocktube", "Byl", 0.0);
  const Real Bzl = pin->GetOrAddReal("shocktube", "Bzl", 0.0);
  const Real rhor = pin->GetOrAddReal("shocktube", "rhor", 1.0);
  const Real Pr = pin->GetOrAddReal("shocktube", "Pr", 1.0);
  const Real vr = pin->GetOrAddReal("shocktube", "vr", 0.0);
  const Real vyr = pin->GetOrAddReal("shocktube", "vyr", 0.0);
  const Real vzr = pin->GetOrAddReal("shocktube", "vzr", 0.0);
  const Real Bxr = pin->GetOrAddReal("shocktube", "Bxr", 0.0);
  const Real Byr = pin->GetOrAddReal("shocktube", "Byr", 0.0);
  const Real Bzr = pin->GetOrAddReal("shocktube", "Bzr", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc.get());
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real rho = x < 0.5 ? rhol : rhor;
        const Real P = x < 0.5 ? Pl : Pr;
        const Real vel = x < 0.5 ? vl : vr;
        const Real vely = x < 0.5 ? vyl : vyr;
        const Real velz = x < 0.5 ? vzl : vzr;

        Real lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          lambda[0] = v(iye, k, j, i);
        }

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho, P, emin, emax, lambda[0]);
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(ieng, k, j, i) / rho,
            lambda); // this doesn't have to be exact, just a reasonable guess
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);
        // Convert raw 3-velocity to Phoebus's u^i = W v^i primitive.
        Real gammacov[3][3] = {0};
        Real vcon[3] = {vel, vely, velz};
        geom.Metric(CellLocation::Cent, k, j, i, gammacov);
        Real vsq = 0.0;
        for (int ii = 0; ii < 3; ii++)
          for (int jj = 0; jj < 3; jj++)
            vsq += gammacov[ii][jj] * vcon[ii] * vcon[jj];
        Real Gamma = 1.0 / std::sqrt(1.0 - vsq);
        v(ivlo, k, j, i) = Gamma * vel;
        v(ivlo + 1, k, j, i) = Gamma * vely;
        v(ivlo + 2, k, j, i) = Gamma * velz;
        if (iye > 0) v(iye, k, j, i) = sin(2.0 * M_PI * x);
      });

  if (pmb->packages.Get("fluid")->Param<bool>("mhd")) {
    using TE = parthenon::TopologicalElement;
    auto Bf = rc->PackVariables({fluid_cons::fbfield::name()});

    if (Bxl != 0.0 || Bxr != 0.0) {
      const auto ibf = rc->GetBoundsI(IndexDomain::entire, TE::F1);
      const auto jbf = rc->GetBoundsJ(IndexDomain::entire, TE::F1);
      const auto kbf = rc->GetBoundsK(IndexDomain::entire, TE::F1);
      pmb->par_for(
          "Phoebus::ProblemGenerator::ShockTube::fbfield::F1", kbf.s, kbf.e, jbf.s, jbf.e,
          ibf.s, ibf.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real Bx = coords.Xf<1>(i) < 0.5 ? Bxl : Bxr;
            Bf(TE::F1, 0, k, j, i) = Bx * geom.DetGamma(CellLocation::Face1, k, j, i);
          });
    }
    if (Byl != 0.0 || Byr != 0.0) {
      const auto ibf = rc->GetBoundsI(IndexDomain::entire, TE::F2);
      const auto jbf = rc->GetBoundsJ(IndexDomain::entire, TE::F2);
      const auto kbf = rc->GetBoundsK(IndexDomain::entire, TE::F2);
      pmb->par_for(
          "Phoebus::ProblemGenerator::ShockTube::fbfield::F2", kbf.s, kbf.e, jbf.s, jbf.e,
          ibf.s, ibf.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real By = coords.Xc<1>(i) < 0.5 ? Byl : Byr;
            Bf(TE::F2, 0, k, j, i) = By * geom.DetGamma(CellLocation::Face2, k, j, i);
          });
    }
    if (Bzl != 0.0 || Bzr != 0.0) {
      const auto ibf = rc->GetBoundsI(IndexDomain::entire, TE::F3);
      const auto jbf = rc->GetBoundsJ(IndexDomain::entire, TE::F3);
      const auto kbf = rc->GetBoundsK(IndexDomain::entire, TE::F3);
      pmb->par_for(
          "Phoebus::ProblemGenerator::ShockTube::fbfield::F3", kbf.s, kbf.e, jbf.s, jbf.e,
          ibf.s, ibf.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real Bz = coords.Xc<1>(i) < 0.5 ? Bzl : Bzr;
            Bf(TE::F3, 0, k, j, i) = Bz * geom.DetGamma(CellLocation::Face3, k, j, i);
          });
    }
  }

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace shock_tube
