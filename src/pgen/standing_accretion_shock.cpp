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

// Parthenon
#include <globals.hpp>

#include "pgen/pgen.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/error_checking.hpp"
#include <cmath>

namespace standing_accretion_shock {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
using Microphysics::EOS::EOS;

class MachResidual {
 public:
  KOKKOS_FUNCTION
  MachResidual(const EOS &eos, const Real rho, const Real vr0, const Real target_mach,
               const Real Ye)
      : eos_(eos), rho_(rho), vr0_(vr0), target_mach_(target_mach) {
    lambda_[0] = Ye;
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real eps) {
    const Real gamma = 4. / 3.;
    Real cs = std::sqrt(gamma * (gamma - 1.) * eps);
    Real mach = vr0_ / cs;
    return mach - target_mach_;
  }

 private:
  const EOS &eos_;
  Real rho_, vr0_, target_mach_;
  Real lambda_[2];
};

KOKKOS_FUNCTION
Real eps_from_rho_mach(const Microphysics::EOS::EOS &eos, const Real rho,
                       const Real target_mach, const Real epsmin, const Real epsmax,
                       const Real vr0, const Real Ye);
KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]);

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalKerrSchild),
                    "Problem \"standing_accretion_shock\" requires \"SphericalKerrSchild\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

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
      eos_type == "IdealGas" || eos_type == "StellarCollapse",
      "Standing Accretion Shock setup only works with Ideal Gas or Stellar Collapse EOS");

  Real Mdot = pin->GetOrAddReal("standing_accretion_shock", "Mdot", 0.2);
  Real rShock = pin->GetOrAddReal("standing_accretion_shock", "rShock", 200);
  Real target_mach = pin->GetOrAddReal("standing_accretion_shock", "target_mach", -100);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  const Real a = pin->GetReal("geometry", "a");
  auto bl = Geometry::BoyerLindquist(a);
  auto epsmin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto epsmax = pmb->packages.Get("eos")->Param<Real>("sie_max");
  //auto Cv = pmb->packages.Get("eos")->Param<Real>("Cv");
  auto Tmin = pmb->packages.Get("eos")->Param<Real>("T_min");
  auto Tmax = pmb->packages.Get("eos")->Param<Real>("T_max");

  printf("Tmin, Tmax, epsmin, epsmax = %g %g %g %g\n", Tmin,Tmax, epsmin, epsmax);

  Mdot *= ((solar_mass * unit_conv.GetMassCGSToCode()) / unit_conv.GetTimeCGSToCode());
  rShock *= (1.e5 * unit_conv.GetLengthCGSToCode());
  Real MPNS = 1.3 * solar_mass;
  Real rs = (2. * pc.g_newt * MPNS / (std::pow(pc.c, 2))) * unit_conv.GetLengthCGSToCode();

  auto geom = Geometry::GetCoordinateSystem(rc.get());
  printf("Rs, rmin (code units) = %g %g \n", rs,  std::abs(coords.Xc<1>(1, 1, 1)));
  pmb->par_for(
      "Phoebus::ProblemGenerator::StandingAccrectionShock", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x1 = coords.Xc<1>(k, j, i);
        const Real x2 = coords.Xc<2>(k, j, i);
        const Real x3 = coords.Xc<3>(k, j, i);

        Real r = std::abs(x1);
        const Real gamma = 4. / 3.;
        const Real alpha0 = std::sqrt(1. - (rs / rShock));
        const Real W0 = 1. / alpha0;
        const Real vr0 = -1. * std::sqrt(std::pow(W0, 2) - 1.) / W0;
        const Real epsND = 0.003; // eps_bar * (W0-1), eps_bar=0.3
        const Real eta = 0.95;

	Real eos_lambda[2] = {0.};
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }

        // postshock - 1
        if (r < rShock) {

          const Real rho_0 = Mdot / (4. * M_PI * std::pow(r, 2) * W0 * std::abs(vr0));
          const Real rho_0_Shock = Mdot / (4. * M_PI * std::pow(rShock, 2) * W0 * std::abs(vr0));
          const Real alphasq = 1. - (rs / r);
          const Real psi = alphasq * ((gamma - 1.) / gamma) * ((W0 - 1. - epsND) / W0);
          const Real vr1 = (vr0 + std::sqrt(vr0 * vr0 - 4. * psi)) / 2.;
          const Real rho1 = (rho_0 * W0 * vr0) / (vr1);
          Real eps1 = (W0 - 1. + (gamma - 1) * epsND) / gamma;

          if (eps1 <= epsND) {
            eps1 = epsmin;
          } else if (eps1 > epsND && eps1 <= epsND * (eta + 1.) / eta) {
            eps1 = eps1 - (eta * (eps1 - epsND));
          } else {
            eps1 = eps1 - epsND;
          }

          Real T1 = eos.TemperatureFromDensityInternalEnergy(rho1, eps1, eos_lambda);

          v(irho, k, j, i) = rho1;
          v(itmp, k, j, i) = T1;
          v(ieng, k, j, i) = rho1 * eps1;
          v(iprs, k, j, i) = eos.PressureFromDensityTemperature(rho1, v(itmp, k, j, i), eos_lambda);
          v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) / v(iprs, k, j, i);

          Real ucon[] = {0.0, vr1, 0.0, 0.0};
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          ucon[0] = ucon_norm(ucon, gcov);

          const Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real W = lapse * ucon[0];

          for (int d = 0; d < 3; d++) {
            v(ivlo + d, k, j, i) = ucon[d + 1] + W * beta[d] / lapse;
          }

          // preshock - 0
        } else {

          const Real rho_0 = Mdot / (4. * M_PI * std::pow(rShock, 2) * W0 * std::abs(vr0));
          const Real eps0 = eps_from_rho_mach(eos, rho_0, target_mach, epsmin, epsmax, vr0, eos_lambda[0]);
	  Real T0 = eos.TemperatureFromDensityInternalEnergy(rho_0,eps0, eos_lambda);

          v(irho, k, j, i) = rho_0;
          v(itmp, k, j, i) = T0;
          v(ieng, k, j, i) = rho_0 * eps0;
          v(iprs, k, j, i) = eos.PressureFromDensityTemperature(
              v(irho, k, j, i), v(itmp, k, j, i), eos_lambda);
          v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                                 v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                             v(iprs, k, j, i);

          Real ucon[] = {0.0, vr0, 0.0, 0.0};
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          ucon[0] = ucon_norm(ucon, gcov);

          const Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real W = lapse * ucon[0];

          for (int d = 0; d < 3; d++) {
            v(ivlo + d, k, j, i) = ucon[d + 1] + W * beta[d] / lapse;
          }
        }
	printf("r, rho, T, eps, W0, eos_lambda[0], eos_lambda[1] = %g %g %g %g %g %g %g \n", r, v(irho, k, j, i), v(itmp, k, j, i), v(ieng, k, j, i) / v(irho, k, j, i), W0, eos_lambda[0], eos_lambda[1]);
      });
  fluid::PrimitiveToConserved(rc.get());
}

KOKKOS_FUNCTION
Real eps_from_rho_mach(const EOS &eos, const Real rho, const Real target_mach,
                       const Real epsmin, const Real epsmax, const Real vr0,
                       const Real Ye) {
  MachResidual res(eos, rho, vr0, target_mach, Ye);
  root_find::RootFind root;
  Real epsroot =
      root.regula_falsi(res, epsmin, epsmax, 1.e-6 * target_mach, std::max(epsmin,1e-10));
  return epsroot;
}

KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]) {
  Real AA = gcov[0][0];
  Real BB = 2. * (gcov[0][1] * ucon[1] + gcov[0][2] * ucon[2] + gcov[0][3] * ucon[3]);
  Real CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + gcov[2][2] * ucon[2] * ucon[2] +
            gcov[3][3] * ucon[3] * ucon[3] +
            2. * (gcov[1][2] * ucon[1] * ucon[2] + gcov[1][3] * ucon[1] * ucon[3] +
                  gcov[2][3] * ucon[2] * ucon[3]);
  Real discr = BB * BB - 4. * AA * CC;
  if (discr < 0) printf("discr = %g   %g %g %g\n", discr, AA, BB, CC);
  PARTHENON_REQUIRE(discr >= 0, "discr < 0");
  return (-BB - std::sqrt(discr)) / (2. * AA);
}

} // namespace standing_accretion_shock
