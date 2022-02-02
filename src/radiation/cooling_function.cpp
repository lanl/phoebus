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

namespace radiation {

using namespace singularity::neutrinos;
using singularity::RadiationType;

TaskStatus CoolingFunctionCalculateFourForce(MeshBlockData<Real> *rc,
                                             const double dt) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({p::density, p::velocity, p::temperature, p::ye,
                                 c::energy, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
  const int ptemp = imap[p::temperature].first;
  const int pye = imap[p::ye].first;
  const int ceng = imap[c::energy].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  auto rad = pmb->packages.Get("radiation").get();
  auto opac = pmb->packages.Get("opacity").get();

  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMPERATURE = unit_conv.GetTemperatureCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  // TODO(BRR) output these dimensions to parameters dump file
  /*const Real T_unit = unit_conv.GetTimeCodeToCGS();
  const Real L_unit = unit_conv.GetLengthCodeToCGS();
  const Real U_unit = unit_conv.GetEnergyCodeToCGS()/pow(L_unit,3);
  printf("T_unit: %e U_unit: %e\n", T_unit, U_unit);
  exit(-1);*/

  auto geom = Geometry::GetCoordinateSystem(rc);

  bool do_species[3] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti"),
                        rad->Param<bool>("do_nu_heavy")};

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Initialize five-force to zero
        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
        v(Gye, k, j, i) = 0.;
      });

  for (int sidx = 0; sidx < 3; sidx++) {
    printf("do_species[%i] = %i\n", sidx, do_species[sidx]);
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
          Real vel[3] = {v(pvlo, k, j, i),
                         v(pvlo + 1, k, j, i),
                         v(pvlo + 2, k, j, i)};
          GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
          Geometry::Tetrads Tetrads(Ucon, Gcov);

          const Real rho_cgs = v(prho, k, j, i) * RHO;
          const Real T_cgs = v(ptemp, k, j, i) * TEMPERATURE;
          const Real Ye = v(pye, k, j, i);

          //double J = d_opacity.Emissivity(rho_cgs, T_cgs, Ye, s);
          double J = d_opacity.Emissivity(v(prho,k,j,i), v(ptemp,k,j,i), Ye, s);
          //double Jye =
          //    pc::mp * d_opacity.NumberEmissivity(rho_cgs, T_cgs, Ye, s);
          double Jye =
              pc::mp * d_opacity.NumberEmissivity(v(prho,k,j,i), v(ptemp,k,j,i), Ye, s);

          // Is this a singularity-opac or my script issue?
          J /= 4.*M_PI;
          Jye /= 4.*M_PI;

          //Real Gcov_tetrad[4] = {-J * CPOWERDENS, 0., 0., 0.};
          Real Gcov_tetrad[4] = {-J, 0., 0., 0.};
          Real Gcov_coord[4];
          Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
          Real detG = geom.DetG(CellLocation::Cent, k, j, i);

          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            Kokkos::atomic_add(&(v(mu, k, j, i)),
                               detG * Gcov_coord[mu - Gcov_lo]);
          }
          //Kokkos::atomic_add(&(v(Gye, k, j, i)),
          //                   -detG * Jye * CDENSITY / CTIME);
          Kokkos::atomic_add(&(v(Gye, k, j, i)),
                             -detG * Jye);
        });
      }
  }

  return TaskStatus::complete;
}

} // namespace radiation
