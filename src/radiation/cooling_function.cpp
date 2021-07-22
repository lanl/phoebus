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

  auto geom = Geometry::GetCoordinateSystem(rc);

  bool do_species[3] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti"),
                        rad->Param<bool>("do_nu_heavy")};

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(),
      0, 2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i) {
        // Initialize five-force to zero
        if (sidx == 0) {
          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            v(mu, k, j, i) = 0.;
          }
          v(Gye, k, j, i) = 0.;
        }

        // Apply cooling for each neutrino species separately
        if (do_species[sidx]) {
          auto s = species[sidx];
          Real Gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
          Real Ucon[4];
          Real vel[4] = {0, v(pvlo, k, j, i), v(pvlo + 1, k, j, i),
                         v(pvlo + 2, k, j, i)};
          GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
          Geometry::Tetrads Tetrads(Ucon, Gcov);

          const Real rho_cgs = v(prho, k, j, i) * RHO;
          const Real T_cgs = v(ptemp, k, j, i) * TEMPERATURE;
          const Real Ye = v(pye, k, j, i);

          double J = d_opacity.Emissivity(s, rho_cgs, T_cgs, Ye);
          double Jye =
              pc::mp * d_opacity.NumberEmissivity(s, rho_cgs, T_cgs, Ye);
          Real Gcov_tetrad[4] = {-J * CPOWERDENS, 0., 0., 0.};
          Real Gcov_coord[4];
          Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
          Real detG = geom.DetG(CellLocation::Cent, k, j, i);

          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            v(mu, k, j, i) += detG * Gcov_coord[mu - Gcov_lo];
          }
          v(Gye, k, j, i) -= detG * Jye * CDENSITY / CTIME;
        }
      });

  return TaskStatus::complete;
}

} // namespace radiation
