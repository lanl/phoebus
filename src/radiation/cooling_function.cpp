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

TaskStatus CoolingFunctionCalculateFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars(
      {p::density, p::velocity, p::temperature, p::ye, c::energy, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
  const int ptemp = imap[p::temperature].first;
  const int pye = imap[p::ye].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

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
  const Real lum = rad->Param<Real>("lum");
  if (do_lightbulb) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(), kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real r = std::abs(coords.Xc<1>(k, j, i)); // TODO(MG) coord transform game
          const Real rho =
              v(prho, k, j, i) * unit_conv.GetMassDensityCodeToCGS(); // Density in CGS
          const Real crho = v(crho, k, j, i);                         // conserved density
          Real Gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
          Real Ucon[4];
          Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
          GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
          Geometry::Tetrads Tetrads(Ucon, Gcov);
          Real Jye;
          Real J;
          const Real lRho = std::log10(rho);
          const Real lRho2 = lRho * lRho;
          const Real lRho3 = lRho2 * lRho;
          const Real lRho4 = lRho2 * lRho2;
          const Real lRho5 = lRho4 * lRho;
          const Real lRho6 = lRho3 * lRho3;

          if (do_liebendorfer) {
            constexpr Real Ye_floor = 0.27;
            constexpr Real a0 = LightBulb::Liebendorfer::A0;
            constexpr Real a1 = LightBulb::Liebendorfer::A1;
            constexpr Real a2 = LightBulb::Liebendorfer::A2;
            constexpr Real a3 = LightBulb::Liebendorfer::A3;
            constexpr Real a4 = LightBulb::Liebendorfer::A4;
            constexpr Real a5 = LightBulb::Liebendorfer::A5;
            constexpr Real a6 = LightBulb::Liebendorfer::A6;

            const Real Ye_fit = a0 + a1 * lRho + a2 * lRho2 + a3 * lRho3 + a4 * lRho4 +
                                a5 * lRho5 + a6 * lRho6;
            const Real Ye = v(pye, k, j, i);
            Real dYe = std::max(-0.05 * Ye, std::min(0.0, Ye_fit - Ye));
            if (rho < 3.e8) { // impose plateau Ye for low densities
              dYe = dYe * (rho - 1.e8) / 2.e8;
            }
            if (Ye < Ye_floor) {
              dYe = 0;
            }
            Jye = dYe / dt * crho;
          }

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
          Real heat;
          Real cool;
          if (lRho < xl2) {
            tau = std::pow(10, (yl2 - yl1) / (xl2 - xl1) * (lRho - xl1) +
                                   yl1); // maybe *tnue42?
          } else if (lRho > xl3) {
            tau = std::pow(10, (yl4 - yl3) / (xl4 - xl3) * (lRho - xl3) +
                                   yl3); // maybe *tnue42?
          } else {
            tau = std::pow(10, (yl3 - yl2) / (xl3 - xl2) * (lRho - xl2) +
                                   yl2); // maybe *tnue42?
          }

          const Real hfac = LightBulb::HeatAndCool::HFAC * lum;
          const Real cfac = LightBulb::HeatAndCool::CFAC;

          constexpr Real lRhoMin = 8;
          constexpr Real lRhoMax = 13;
          constexpr Real rnorm = 1.e7;
          constexpr Real MeVToCGS = 1.6021773e6;
          constexpr Real Tnorm = 2.0 * MeVToCGS;
          bool do_heatcool = (lRhoMin <= lRho && lRho <= lRhoMax);
          heat = (tau > 1.e2) * do_heatcool * hfac * std::exp(-tau) *
                 pow((rnorm / r), 2); // maybe *compweight?
          cool = do_heatcool * cfac * std::exp(-tau) *
                 pow((ptemp / Tnorm), 6); // Maybe *compweight?
          Real CGSToCodeFact = unit_conv.GetEnergyCGSToCode() /
                               unit_conv.GetMassCGSToCode() /
                               unit_conv.GetTimeCGSToCode();
          Real H = heat * CGSToCodeFact;
          Real C = cool * CGSToCodeFact;
          // convert cool and heat from erg/g/s to code units

          J = crho * (cool - heat);
          Real Gcov_tetrad[4] = {-J, 0., 0., 0.};
          Real Gcov_coord[4];
          Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
          Real detG = geom.DetG(CellLocation::Cent, k, j, i); // can skip detg
          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            Kokkos::atomic_add(&(v(mu, k, j, i)), -detG * Gcov_coord[mu - Gcov_lo]);
          }
          Kokkos::atomic_add(&(v(Gye, k, j, i)), Jye);
        });
  }

  for (int sidx = 0; sidx < 3; sidx++) {
    // Apply cooling for each neutrino species separately
    if (do_species[sidx]) {
      auto s = species[sidx];

      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(), kb.s,
          kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Real Gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
            Real Ucon[4];
            Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
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
            Kokkos::atomic_add(&(v(Gye, k, j, i)), -detG * Jye);
          });
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
