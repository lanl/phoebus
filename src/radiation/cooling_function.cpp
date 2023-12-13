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

#include "geometry/geometry.hpp"
#include "light_bulb_constants.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation.hpp"
#include <algorithm>

namespace radiation {

using Microphysics::Opacities;
using Microphysics::RadiationType;

TaskStatus LightBulbCalcTau(MeshBlockData<Real> *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  Mesh *pmesh = rc->GetMeshPointer();

  std::vector<std::string> vars({p::density::name(), iv::tau::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int prho = imap[p::density::name()].first;
  const int ptau = imap[iv::tau::name()].first;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto &unit_conv =
      pmesh->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  const Real density_conversion_factor = unit_conv.GetMassDensityCodeToCGS();

  pmesh->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalcTau", DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real rho = v(prho, k, j, i) * density_conversion_factor; // Density in CGS
        const Real lRho = std::log10(rho);
        // Calculate tau
        constexpr Real xl1 = LightBulb::HeatAndCool::XL1;
        constexpr Real xl2 = LightBulb::HeatAndCool::XL2;
        constexpr Real xl3 = LightBulb::HeatAndCool::XL3;
        constexpr Real xl4 = LightBulb::HeatAndCool::XL4;
        constexpr Real yl1 = LightBulb::HeatAndCool::YL1;
        constexpr Real yl2 = LightBulb::HeatAndCool::YL2;
        constexpr Real yl3 = LightBulb::HeatAndCool::YL3;
        constexpr Real yl4 = LightBulb::HeatAndCool::YL4;
        Real tau;
        if (lRho < xl2) {
          tau = std::pow(10, (yl2 - yl1) / (xl2 - xl1) * (lRho - xl1) + yl1);
        } else if (lRho > xl3) {
          tau = std::pow(10, (yl4 - yl3) / (xl4 - xl3) * (lRho - xl3) + yl3);
        } else {
          tau = std::pow(10, (yl3 - yl2) / (xl3 - xl2) * (lRho - xl2) + yl2);
        }
        v(ptau, k, j, i) = tau;
      });
  return TaskStatus::complete;
}

TaskStatus CheckDoGain(MeshBlockData<Real> *rc, bool *do_gain_global) {
  if (*do_gain_global) {
    return TaskStatus::complete;
  }
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  Mesh *pmesh = rc->GetMeshPointer();

  std::vector<std::string> vars({iv::tau::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int ptau = imap[iv::tau::name()].first;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto &unit_conv =
      pmesh->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  auto rad = pmesh->packages.Get("radiation").get();
  auto opac = pmesh->packages.Get("opacity").get();

  int do_gain_local = 0;
  bool do_gain;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "calc_do_gain", DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &do_gain) {
        do_gain = do_gain + (v(ptau, k, j, i) > 1.e2);
      },
      Kokkos::Sum<int>(do_gain_local));
  do_gain = do_gain_local;
  *do_gain_global = std::max(do_gain, *do_gain_global);
  return TaskStatus::complete;
}

TaskStatus CoolingFunctionCalculateFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({c::density::name(), p::density::name(),
                                 p::velocity::name(), p::temperature::name(),
                                 p::ye::name(), c::energy::name(), iv::Gcov::name(),
                                 iv::GcovHeat::name(), iv::GcovCool::name(),
                                 iv::Gye::name(), iv::tau::name(), p::energy::name()});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int crho = imap[c::density::name()].first;
  const int prho = imap[p::density::name()].first;
  const int pvlo = imap[p::velocity::name()].first;
  const int pvhi = imap[p::velocity::name()].second;
  const int ptemp = imap[p::temperature::name()].first;
  const int pye = imap[p::ye::name()].first;
  const int penergy = imap[p::energy::name()].first;
  const int Gcov_lo = imap[iv::Gcov::name()].first;
  const int Gcov_hi = imap[iv::Gcov::name()].second;
  const int Gye = imap[iv::Gye::name()].first;
  const int ptau = imap[iv::tau::name()].first;
  const int GcovHeat = imap[iv::GcovHeat::name()].first;
  const int GcovCool = imap[iv::GcovCool::name()].first;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  auto rad = pmb->packages.Get("radiation").get();
  auto opac = pmb->packages.Get("opacity").get();

  auto &phoebus_pkg = pmb->packages.Get("phoebus");
  auto &code_constants = phoebus_pkg->Param<phoebus::CodeConstants>("code_constants");
  const Real mp_code = code_constants.mp;

  const auto d_opacity = opac->Param<Opacities>("opacities");

  auto geom = Geometry::GetCoordinateSystem(rc);

  bool do_species[3] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti"),
                        rad->Param<bool>("do_nu_heavy")};

  // Code to CGS
  const Real density_conversion_factor = unit_conv.GetMassDensityCodeToCGS();
  const Real temperature_conversion_factor = unit_conv.GetTemperatureCodeToCGS();
  const Real length_conversion_factor = unit_conv.GetLengthCodeToCGS();

  // CGS to code
  const Real energy_conversion_factor = unit_conv.GetEnergyCGSToCode();
  const Real mass_conversion_factor = unit_conv.GetMassCGSToCode();
  const Real time_conversion_factor = unit_conv.GetTimeCGSToCode();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Initialize five-force to zero
        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
        v(Gye, k, j, i) = 0.;
      });

  // Light Bulb with Liebendorfer model
  auto &coords = pmb->coords;
  const bool do_liebendorfer = rad->Param<bool>("do_liebendorfer");
  const bool do_lightbulb = rad->Param<bool>("do_lightbulb");
  if (do_lightbulb) {
#ifdef SPINER_USE_HDF
    const Real lum = rad->Param<Real>("lum");
    auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
    singularity::StellarCollapse eos_sc =
        eos.GetUnmodifiedObject().get<singularity::StellarCollapse>();
    const parthenon::AllReduce<bool> *pdo_gain_reducer =
        rad->MutableParam<parthenon::AllReduce<bool>>("do_gain_reducer");
    const bool do_gain = pdo_gain_reducer->val;

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(), kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real r = std::abs(coords.Xc<1>(k, j, i)); // TODO(MG) coord transform game
          const Real rho = v(prho, k, j, i) * density_conversion_factor; // Density in CGS
          const Real cdensity = v(crho, k, j, i); // conserved density
          Real Gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
          Real Ucon[4];
          Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
          GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
          Geometry::Tetrads Tetrads(Ucon, Gcov);
          Real Jye = 0.0;
          Real J;
          const Real lRho = std::log10(rho);
          const Real lRho2 = lRho * lRho;
          const Real lRho3 = lRho2 * lRho;
          const Real lRho4 = lRho2 * lRho2;
          const Real lRho5 = lRho4 * lRho;
          const Real lRho6 = lRho3 * lRho3;
          constexpr Real lRhoMin = LightBulb::Liebendorfer::LRHOMIN;
          constexpr Real lRhoMax = LightBulb::Liebendorfer::LRHOMAX;
          bool do_densityregion = (lRhoMin <= lRho && lRho <= lRhoMax); // better name?
          constexpr Real rnorm = LightBulb::HeatAndCool::RNORM;
          constexpr Real MeVToCGS = 1.16040892301e10;
          constexpr Real Tnorm = 2.0 * MeVToCGS;

          Real Ye = v(pye, k, j, i);

          if (do_liebendorfer) {
            constexpr Real Ye_beta = 0.27;
            constexpr Real Ye_floor = 0.05;
            constexpr Real a0 = LightBulb::Liebendorfer::A0;
            constexpr Real a1 = LightBulb::Liebendorfer::A1;
            constexpr Real a2 = LightBulb::Liebendorfer::A2;
            constexpr Real a3 = LightBulb::Liebendorfer::A3;
            constexpr Real a4 = LightBulb::Liebendorfer::A4;
            constexpr Real a5 = LightBulb::Liebendorfer::A5;
            constexpr Real a6 = LightBulb::Liebendorfer::A6;

            if (do_densityregion) {
              const Real Ye_fit = (a0 + a1 * lRho + a2 * lRho2 + a3 * lRho3 + a4 * lRho4 +
                                   a5 * lRho5 + a6 * lRho6);
              Real dYe = std::max(-0.05 * Ye, std::min(0.0, Ye_fit - Ye));
              if (rho < 3.e8) { // impose plateau Ye for low densities
                dYe = dYe * (rho - 1.e8) / 2.e8;
              }
              if (Ye < Ye_beta) {
                dYe = 0;
              }
              Jye = dYe / dt * cdensity;
            } else {
              Jye = 0.0;
            }
          }
          Real heat;
          Real cool;
          const Real tau = v(ptau, k, j, i);
          const Real hfac = LightBulb::HeatAndCool::HFAC * lum;
          const Real cfac = LightBulb::HeatAndCool::CFAC;
          Real Xa, Xh, Xn, Xp, Abar, Zbar;
          Real lambda[2];
          lambda[0] = Ye;
          eos_sc.MassFractionsFromDensityTemperature(
              rho, v(ptemp, k, j, i) * temperature_conversion_factor, Xa, Xh, Xn, Xp,
              Abar, Zbar, lambda);
          heat = do_gain * (Xn + Xp) * hfac * std::exp(-tau) *
                 pow((rnorm / (r * length_conversion_factor)), 2);
          cool = (Xn + Xp) * cfac * std::exp(-tau) *
                 pow((v(ptemp, k, j, i) * temperature_conversion_factor / Tnorm), 6);

          Real CGSToCodeFact =
              energy_conversion_factor / mass_conversion_factor / time_conversion_factor;

          Real tempr = 1 / 30.76 / 9e20;
          Real H = heat * CGSToCodeFact;
          Real C = cool * CGSToCodeFact;
          J = cdensity * (H - C);                // looks like Cufe
          Real Gcov_tetrad[4] = {J, 0., 0., 0.}; // minus sign included above
          Real Gcov_coord[4];
          Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            // detg included above
            Kokkos::atomic_add(&(v(mu, k, j, i)), -Gcov_coord[mu - Gcov_lo]);
          }
          v(GcovHeat, k, j, i) = v(prho, k, j, i) * density_conversion_factor * heat;
          v(GcovCool, k, j, i) = v(prho, k, j, i) * density_conversion_factor * cool;
          Kokkos::atomic_add(&(v(Gye, k, j, i)), Jye);
        });
#else
    PARTHENON_THROW("Lighbulb only supported with HDF5");
#endif // SPINER_USE_HDF
  } else {
    for (int sidx = 0; sidx < 3; sidx++) {
      // Apply cooling for each neutrino species separately
      if (do_species[sidx]) {
        auto s = species[sidx];

        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(),
            kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
              Real Gcov[4][4];
              geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
              Real Ucon[4];
              Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i),
                             v(pvlo + 2, k, j, i)};
              GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
              Geometry::Tetrads Tetrads(Ucon, Gcov);

              const Real Ye = v(pye, k, j, i);

              double J = d_opacity.Emissivity(v(prho, k, j, i), v(ptemp, k, j, i), Ye, s);
              double Jye = mp_code * d_opacity.NumberEmissivity(v(prho, k, j, i),
                                                                v(ptemp, k, j, i), Ye, s);

              Real Gcov_tetrad[4] = {-J, 0., 0., 0.};
              Real Gcov_coord[4];
              Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
              Real detG = geom.DetG(CellLocation::Cent, k, j, i);

              for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
                Kokkos::atomic_add(&(v(mu, k, j, i)), -detG * Gcov_coord[mu - Gcov_lo]);
              }
              Kokkos::atomic_add(&(v(Gye, k, j, i)), -LeptonSign(s) * detG * Jye);
            });
      }
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
