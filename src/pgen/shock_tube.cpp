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
// Zachary, Malagoli, A., & Colella,P., SIAM J. Sci. Comp., 15, 263 (1994); Balsara, D., &
// Spicer, D., JCP 149, 270 (1999); Londrillo, P. & Del Zanna, L., ApJ 530, 508 (2000).

// namespace phoebus {

namespace shock_tube {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  namespace p = fluid_prim;
  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski) ||
                        typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski),
                    "Problem \"shock_tube\" requires \"Minkowski\" or "
                    "\"SphericalMinkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<p::density, p::velocity, p::energy,
                        p::bfield, p::ye, p::pressure, 
                        p::temperature, p::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

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

        Real lambda[2];
        if (v.Contains(0, p::ye())) {
          v(0, p::ye(), k, j, i) = 0.5;
          lambda[0] = v(0, p::ye(), k, j, i);
        }

        v(0, p::density(), k, j, i) = rho;
        v(0, p::pressure(), k, j, i) = P;
        v(0, p::energy(), k, j, i) = phoebus::energy_from_rho_P(eos, rho, P, emin, emax, lambda[0]);
        v(0, p::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(0, p::energy(), k, j, i) / rho,
            lambda); // this doesn't have to be exact, just a reasonable guess
        v(0, p::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, p::density(), k, j, i), v(0, p::temperature(), k, j, i), lambda) /
                           v(0, p::pressure(), k, j, i);
        for (int d = 0; d < 3; ++d)
          v(0, p::velocity(d), k, j, i) = 0.0;
        v(0, p::velocity(0), k, j, i) = vel;
        Real gammacov[3][3] = {0};
        Real vcon[3] = {v(0, p::velocity(0), k, j, i), v(0, p::velocity(1), k, j, i), v(0, p::velocity(2), k, j, i)};
        geom.Metric(CellLocation::Cent, k, j, i, gammacov);
        Real Gamma = phoebus::GetLorentzFactor(vcon, gammacov);
        v(0, p::velocity(0), k, j, i) *= Gamma;
        if (v.Contains(0, p::bfield(0))) {
          const Real Bx = x < 0.5 ? Bxl : Bxr;
          const Real By = x < 0.5 ? Byl : Byr;
          const Real Bz = x < 0.5 ? Bzl : Bzr;
          v(0, p::bfield(0), k, j, i) = Bx;
          v(0, p::bfield(1), k, j, i) = By;
          v(0, p::bfield(2), k, j, i) = Bz;
        }
        if (v.Contains(0, p::ye())) {
          v(0, p::ye(), k, j, i) = sin(2.0 * M_PI * x);
        }
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace shock_tube
