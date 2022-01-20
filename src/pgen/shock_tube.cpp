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
#include "phoebus_utils/relativity_utils.hpp"

// Single-material blast wave.
// As descriged in the Athena test suite
// https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
// and in
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., & Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000).

//namespace phoebus {

namespace shock_tube {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski) ||
    typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski),
    "Problem \"shock_tube\" requires \"Minkowski\" or \"SphericalMinkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density,
                              fluid_prim::velocity,
                              fluid_prim::energy,
                              fluid_prim::bfield,
                              fluid_prim::ye,
                              fluid_prim::pressure,
                              fluid_prim::temperature},
                              imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iye  = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;

  const Real rhol = pin->GetOrAddReal("shocktube", "rhol", 1.0);
  const Real Pl = pin->GetOrAddReal("shocktube", "Pl", 1.0);
  const Real vl = pin->GetOrAddReal("shocktube", "vl", 0.0);
  const Real Bxl = pin->GetOrAddReal("shocktube", "Bxl", 0.0);
  const Real Byl = pin->GetOrAddReal("shocktube", "Byl", 0.0);
  const Real Bzl = pin->GetOrAddReal("shocktube", "Bzl", 0.0);
  const Real rhor = pin->GetOrAddReal("shocktube", "rhor", 1.0);
  const Real Pr = pin->GetOrAddReal("shocktube", "Pr", 1.0);
  const Real vr = pin->GetOrAddReal("shocktube", "vr", 0.0);
  const Real Bxr = pin->GetOrAddReal("shocktube", "Bxr", 0.0);
  const Real Byr = pin->GetOrAddReal("shocktube", "Byr", 0.0);
  const Real Bzr = pin->GetOrAddReal("shocktube", "Bzr", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc.get());
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  printf("pgen\n"); // debug

  pmb->par_for(
    "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      const Real rho = x < 0.5 ? rhol : rhor;
      const Real P = x < 0.5 ? Pl : Pr;
      const Real vel = x < 0.5 ? vl : vr;

      Real lambda[2];
      if (iye > 0) {
	v(iye, k, j, i) = 0.5;
	lambda[0] = v(iye, k, j, i);
      }

      v(irho, k, j, i) = rho;
      v(iprs, k, j, i) = P;
      printf("get energy from rho P\n");
      v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho, P, emin, emax, lambda[0]);
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho, lambda); // this doesn't have to be exact, just a reasonable guess
      printf("eos calls finished\n");
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
      v(ivlo, k, j, i) = vel;
      Real gammacov[3][3] = {0};
      Real vcon[3] = {v(ivlo, k, j, i), v(ivlo+1, k, j, i), v(ivlo + 2, k, j, i)};
      geom.Metric(CellLocation::Cent, k, j, i, gammacov);
      Real Gamma = phoebus::GetLorentzFactor(vcon, gammacov);
      v(ivlo, k, j, i) *= Gamma;
      if (ib_hi > 0) {
        const Real Bx = x < 0.5 ? Bxl : Bxr;
        const Real By = x < 0.5 ? Byl : Byr;
        const Real Bz = x < 0.5 ? Bzl : Bzr;
        v(ib_lo, k, j, i) = Bx;
        v(ib_lo + 1, k, j, i) = By;
        v(ib_hi, k, j, i) = Bz;
      }
      if (iye > 0) v(iye, k, j, i) = sin(2.0*M_PI*x);
    });

  fluid::PrimitiveToConserved(rc.get());
  printf("pgen finished\n"); // debug
}

} // namespace shock_tube
