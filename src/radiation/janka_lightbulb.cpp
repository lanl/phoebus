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
#include "phoebus_utils/variables.hpp"
#include "radiation.hpp"
#include <algorithm>
#include "geometry/boyer_lindquist.hpp"

namespace radiation {

using Microphysics::Opacities;
using Microphysics::RadiationType;

TaskStatus JankaLightbulbCalculateFourForce(MeshBlockData<Real> *rc, const double dt) {
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

  auto &unit_conv = pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  auto rad = pmb->packages.Get("radiation").get();
  auto &phoebus_pkg = pmb->packages.Get("phoebus");
  auto geom = Geometry::GetCoordinateSystem(rc);
  const Real Tnue = rad->Param<Real>("Tnue"); // MeV
  const Real Lnue = rad->Param<Real>("Lnue"); // Bethe
  auto &coords = pmb->coords;
  bool do_species[2] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti")};

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
                    "Problem \"torus\" requires FMKS geometry!");

  auto gpkg = pmb->packages.Get("geometry");
  bool derefine_poles = gpkg->Param<bool>("derefine_poles");
  Real h = gpkg->Param<Real>("h");
  Real xt = gpkg->Param<Real>("xt");
  Real alpha = gpkg->Param<Real>("alpha");
  Real x0 = gpkg->Param<Real>("x0");
  Real smooth = gpkg->Param<Real>("smooth");
  auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JankaLightbulbCalculateFourForce", DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
        v(Gye, k, j, i) = 0.;
      });

  parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "JankaLightbulbCalculateFourForce", DevExecSpace(), kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real x1 = coords.Xc<1>(k, j, i);
          Real r = tr.bl_radius(x1);
	  Real rho = v(prho, k, j, i) * unit_conv.GetMassDensityCodeToCGS();
          Real crho = v(crho, k, j, i);                    
          Real Gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
          Real Ucon[4];
          Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
          GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
          Geometry::Tetrads Tetrads(Ucon, Gcov);
          Real Lnue_cgs = Lnue * 1e51; // B = erg / s
	  Real r_km = (r * unit_conv.GetLengthCodeToCGS())  / 1.e5;
          Real H_cgs = 1.544e20 * ( Lnue_cgs / 1.e52 ) * std::pow(Tnue/4.,2) * std::pow(100./r_km,2) * std::exp(-1.*taunue);  
          Real H = H_cgs * (unit_conv.GetEnergyCGSToCode() / unit_conv.GetTimeCGSToCode() * unit_conv.GetMassCGSToCode());


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
          DEFAULT_LOOP_PATTERN, "JankLightbulbCalculateFourForce", DevExecSpace(), kb.s,
          kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Real Gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
            Real Ucon[4];
            Real vel[3] = {v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
            GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
            Geometry::Tetrads Tetrads(Ucon, Gcov);


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

  return TaskStatus::complete;
}

} // namespace radiation
