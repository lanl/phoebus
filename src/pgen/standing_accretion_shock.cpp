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
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/error_checking.hpp"

// namespace phoebus {

namespace standing_accretion_shock {

KOKKOS_INLINE_FUNCTION
void bl_to_ks(const Real r, const Real a, Real *ucon_bl, Real *ucon_ks) {
  using namespace Geometry;
  Real trans[NDFULL][NDFULL];
  LinearAlgebra::SetZero(trans, NDFULL, NDFULL);
  const Real idenom = 1.0 / (r * r - 2.0 * r + a * a);
  trans[0][0] = 1.0;
  trans[0][1] = 2.0 * r * idenom;
  trans[1][1] = 1.0;
  trans[2][2] = 1.0;
  trans[3][1] = a * idenom;
  trans[3][3] = 1.0;
  LinearAlgebra::SetZero(ucon_ks, NDFULL);
  SPACETIMELOOP2(mu, nu) { ucon_ks[mu] += trans[mu][nu] * ucon_bl[nu]; }
}

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

  const Real Mdot = pin->GetOrAddReal("standing_accretion_shock", "Mdot",
                                      0.2); // pin in units of (Msun / sec)
  const Real rShock = pin->GetOrAddReal("standing_accretion_shock", "rShock",
                                        200); // pin in units of (km)
  const Real target_mach = pin->GetOrAddReal("standing_accretion_shock", "target_mach",
                                             100); // pin in units of (dimensionless)

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");

  const Real a = pin->GetReal("geometry", "a");
  auto bl = Geometry::BoyerLindquist(a);

  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");

  // convert to CGS then to code units
  Mdot *= ((solar_mass * unit_conv.GetMassCGSToCode()) / unit_conv.GetTimeCGSToCode());
  rShock *= (1.e5 * unit_conv.GetLengthCGSToCode());
  const Real Tmin =
      11604525006.1598 * unit_conv.GetTemperatureCGSToCode(); // 1 MeV => K => code units
  const Real Tmax = 2901131251539.96 *
                    unit_conv.GetTemperatureCGSToCode(); // 250 MeV => K => code units

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
        Real vel_rad;

        // set Ye everywhere
        Real eos_lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }

        if (r < rShock) {
          const Real lapse0 = geom.Lapse(
              CellLocation::Cent, k, j,
              rShock); // cold, free falling, lapse equal to value at shock radius
          const Real W0 = 1. / lapse0;
          const Real vr0 = abs((std::sqrt(W0 - 1)) / (std::sqrt(W0)));
          vel_rad = const Real rho0 =
              Mdot / (4. * PI * std::pow(rShock, 2) * W0 * std::abs(vr0));
          v(irho, k, j, i) = rho0;
          Real T0 = temperature_from_rho_mach(
              &eos, const Real rho0, const Real target_mach, const Real Tmin,
              const Real Tmax, const Real Ye = 0.5, const Real vr0);
          v(itmp, k, j, i) = T0;
          v(ieng, k, j, i) =
              rho0 * eos.InternalEnergyFromDensityTemperature(rho0, T0, eos_lambda);
          v(iprs, k, j, i) = eos.PressureFromDensityTemperature(
              v(irho, k, j, i), v(itmp, k, j, i), eos_lambda);
          v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                                 v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                             v(iprs, k, j, i);

        } else {
          const Real lapse1 = geom.Lapse(CellLocation::Cent, k, j, r);
          const Real W0 = 1. / lapse0;
          const Real vr0 = abs((std::sqrt(W0 - 1)) / (std::sqrt(W0)));
          const Real rho0 = Mdot / (4. * PI * std::pow(rShock, 2) * W0 * std::abs(vr0));
          const Real gamma = 1.4;
          const Real psi = std::pow(2, lapse1) *
                           ((gamma - 1) / gamma * (W0 - 1) /
                            W0); // here we assume epsilonND = 0 and accounted for in EoS
           const Real vr1 = (vr0 + std::sqrt(vr0 * vr0 - 4.*psi)/2;
           const Real rho1 = rho0*W0*(vr0/vr1);
	   v(irho, k, j, i) = rho1;
           Real T1 = temperature_from_rho_mach(&eos, const Real rho1, const Real target_mach, const Real Tmin, const Real Tmax, const Real Ye = 0.5, const Real vr1);
           v(itmp, k, j, i) = T1;
           v(ieng, k, j, i) = rho1 * eos.InternalEnergyFromDensityTemperature(rho1, T1, eos_lambda);
           v(iprs, k, j, i) = eos.PressureFromDensityTemperature(v(irho, k, j, i), v(itmp, k, j, i), eos_lambda);
           v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) / v(iprs, k, j, i);
        }

        Real ucon_bl[] = {0.0, 0.0, 0.0, 0.0};
        Real gcov[4][4];
        const Real th = tr.bl_theta(x1, x2);
        bl.SpacetimeMetric(0.0, r, th, x3, gcov);
        Real AA = gcov[0][0];
        Real BB = 2. * (gcov[0][1] * ucon_bl[1] + gcov[0][2] * ucon_bl[2] +
                        gcov[0][3] * ucon_bl[3]);
        Real CC = 1. + gcov[1][1] * ucon_bl[1] * ucon_bl[1] +
                  gcov[2][2] * ucon_bl[2] * ucon_bl[2] +
                  gcov[3][3] * ucon_bl[3] * ucon_bl[3] +
                  2. * (gcov[1][2] * ucon_bl[1] * ucon_bl[2] +
                        gcov[1][3] * ucon_bl[1] * ucon_bl[3] +
                        gcov[2][3] * ucon_bl[2] * ucon_bl[3]);
        Real discr = BB * BB - 4. * AA * CC;
        ucon_bl[0] = (-BB - std::sqrt(discr)) / (2. * AA);
        const Real W_bl = ucon_bl[0] * bl.Lapse(0.0, r, th, x3);

        Real ucon[4];
        tr.bl_to_fmks(x1, x2, x3, a, ucon_bl, ucon);

        // ucon won't be properly normalized here if x1 is not consistent with i
        // so renormalize
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        AA = gcov[0][0];
        BB = 2. * (gcov[0][1] * ucon[1] + gcov[0][2] * ucon[2] + gcov[0][3] * ucon[3]);
        CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + gcov[2][2] * ucon[2] * ucon[2] +
             gcov[3][3] * ucon[3] * ucon[3] +
             2. * (gcov[1][2] * ucon[1] * ucon[2] + gcov[1][3] * ucon[1] * ucon[3] +
                   gcov[2][3] * ucon[2] * ucon[3]);
        discr = BB * BB - 4. * AA * CC;
        PARTHENON_REQUIRE(discr > 0, "discr < 0");
        ucon[0] = (-BB - std::sqrt(discr)) / (2. * AA);

        // now get three velocity
        const Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
        Real beta[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
        Real W = lapse * ucon[0];
        for (int d = 0; d < 3; d++) {
          v(ivlo + d, k, j, i) = ucon[d + 1] + W * beta[d] / lapse;
        }
      });

  fluid::PrimitiveToConserved(rc);
}

} // namespace standing_accretion_shock
