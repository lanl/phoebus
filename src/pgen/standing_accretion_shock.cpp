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

#include "geometry/boyer_lindquist.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/error_checking.hpp"

// namespace phoebus {

namespace standing_accretion_shock {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
                    "Problem \"standing_accretion_shock\" requires \"FMKS\" geometry!");

  auto rc = pmb->meshblock_data.Get().get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density, fluid_prim::velocity, fluid_prim::energy, fluid_prim::bfield,
       fluid_prim::ye, fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1},
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
  const int igm1 = imap[fluid_prim::gamma1].first;

  const std::string eos_type = pin->GetString("eos", "type");
  PARTHENON_REQUIRE_THROWS(
      eos_type == "StellarCollapse",
      "Standing Accretion Shock setup only works with StellarCollapse EOS");

  Real Mdot = pin->GetOrAddReal("standing_accretion_shock", "Mdot",
                                0.2); // pin in units of (Msun / sec)
  Real rShock = pin->GetOrAddReal("standing_accretion_shock", "rShock",
                                  200); // pin in units of (km)
  const Real target_mach = pin->GetOrAddReal("standing_accretion_shock", "target_mach",
                                             100); // pin in units of (dimensionless)

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto Tmin = pmb->packages.Get("eos")->Param<Real>("T_min");
  auto Tmax = pmb->packages.Get("eos")->Param<Real>("T_max");

  const Real a = pin->GetReal("geometry", "a");
  auto bl = Geometry::BoyerLindquist(a);

  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");

  // convert to CGS then to code units
  Mdot *= ((solar_mass * unit_conv.GetMassCGSToCode()) / unit_conv.GetTimeCGSToCode());
  rShock *= (1.e5 * unit_conv.GetLengthCGSToCode());

  auto geom = Geometry::GetCoordinateSystem(rc);

  // set up transformation stuff
  auto gpkg = pmb->packages.Get("geometry");
  bool derefine_poles = gpkg->Param<bool>("derefine_poles");
  Real h = gpkg->Param<Real>("h");
  Real xt = gpkg->Param<Real>("xt");
  Real alpha = gpkg->Param<Real>("alpha");
  Real x0 = gpkg->Param<Real>("x0");
  Real smooth = gpkg->Param<Real>("smooth");
  auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

  pmb->par_for(
      "Phoebus::ProblemGenerator::StandingAccrectionShock", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x1 = coords.Xc<1>(k, j, i);
        const Real x2 = coords.Xc<2>(k, j, i);
        const Real x3 = coords.Xc<3>(k, j, i);

        Real r = tr.bl_radius(x1); // r = e^(x1min)
        const Real gamma = 4. / 3.;

        // set Ye everywhere
        Real eos_lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }

        if (r > rShock) {
          // preshock - 0
          Real lapse0 = geom.Lapse(CellLocation::Cent, k, j, rShock);
          Real W0 = 1. / lapse0;
          Real vr0 = -1. * std::sqrt(W0 * W0 - 1.) / W0;
          Real rho0 = Mdot / (4. * M_PI * std::pow(rShock, 2) * W0 * std::abs(vr0));
          // printf("preshock (r>rShock): passing values of: r, rho0, vr0, lapse0, psi  =
          // %g %g %g %g\n",r, rho0, vr0, lapse0);

          Real T = phoebus::temperature_from_rho_mach(eos, rho0, target_mach, Tmin, Tmax,
                                                      vr0, eos_lambda[0]);
          v(irho, k, j, i) = rho0;
          v(itmp, k, j, i) = T;
          v(ieng, k, j, i) =
              rho0 * eos.InternalEnergyFromDensityTemperature(rho0, T, eos_lambda);
          v(iprs, k, j, i) = eos.PressureFromDensityTemperature(
              v(irho, k, j, i), v(itmp, k, j, i), eos_lambda);
          v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                                 v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                             v(iprs, k, j, i);

          Real ucon[4] = {0.0, vr0, 0.0, 0.0};
          const Real lapsed = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real Wd = lapsed * ucon[0];

          // finally compute three-velocity
          for (int d = 0; d < 3; d++) {
            v(ivlo + d, k, j, i) = ucon[d + 1] + Wd * beta[d] / lapsed;
          }

        } else {
          // postshock - 1
          Real lapse0 = geom.Lapse(CellLocation::Cent, k, j, rShock);
          Real W0 = 1. / lapse0;
          Real vr0 = -1. * std::sqrt(W0 * W0 - 1.) / W0;
          Real rho0 = Mdot / (4. * M_PI * std::pow(rShock, 2) * W0 * std::abs(vr0));

          Real alphasq = 1. - (2. / r);
          Real psi = alphasq * ((gamma - 1.) / gamma) * ((W0 - 1.) / W0);
          Real vr1 = (vr0 + std::sqrt(vr0 * vr0 - 4. * psi)) / 2.;
          Real rho1 = rho0 * W0 * (vr0 / vr1);

          const Real epsND = 0.003; // 2.7e16 ergs / g for M = 1.3Mpns

          // printf("postshock r<rShock: passing values of: r, rho1, vr1, lapse0, psi  =
          // %g %g %g %g %g\n",r, rho1, vr1, lapse0, psi);
          v(irho, k, j, i) = rho1;
          v(ieng, k, j, i) = (W0 - 1. + epsND * (gamma - 1.)) / (gamma);
          v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
              rho1, v(ieng, k, j, i) / rho1, eos_lambda);
          v(iprs, k, j, i) = eos.PressureFromDensityTemperature(
              v(irho, k, j, i), v(itmp, k, j, i), eos_lambda);
          v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                                 v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                             v(iprs, k, j, i);

          Real ucon[4] = {0.0, vr1, 0.0, 0.0};
          const Real lapsed = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real Wd = lapsed * ucon[0];

          // finally compute three-velocity
          for (int d = 0; d < 3; d++) {
            v(ivlo + d, k, j, i) = ucon[d + 1] + Wd * beta[d] / lapsed;
          }
        }
      });

  fluid::PrimitiveToConserved(rc);
}

} // namespace standing_accretion_shock
