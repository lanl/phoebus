#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation.hpp"

#include "opacity.hpp"

namespace radiation {

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
  const int ceng = imap[c::energy].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  // TODO(BRR) Temporary cooling problem parameters
  NeutrinoSpecies s = NeutrinoSpecies::Electron;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  auto rad = pmb->packages.Get("radiation").get();

  const auto d_opacity = rad->Param<Opacity*>("d_opacity");
  printf("now d_opacity: %p\n", d_opacity);
  printf("? %e\n", d_opacity->GetJ(0,0,0,NeutrinoSpecies::Electron));

  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMPERATURE = unit_conv.GetTemperatureCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CoolingFunctionCalculateFourForce", DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real Gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
        Real Ucon[4];
        Real vel[4] = {0, v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
        GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
        Geometry::Tetrads Tetrads(Ucon, Gcov);

        const Real rho_cgs = v(prho, k, j, i) * RHO;
        const Real T_cgs = v(ptemp, k, j, i)*TEMPERATURE;
        const Real Ye = v(pye, k, j, i);

         //double J = GetJ(v(prho, k, j, i) * RHO, v(pye, k, j, i), s);
        double J = GetJ(v(pye, k, j, i), s);
        printf("ye: %e s: %i\n", v(pye,k,j,i), static_cast<int>(s));
        //printf("about to get opacity\n");
        //printf("rho t ye s: %e %e %e %i\n",
        //  v(prho, k, j, i) * RHO,
        //  v(ptemp, k, j, i)*TEMPERATURE,
        //  v(pye, k, j, i),
        //  static_cast<int>(s));
        //double J = d_opacity->GetJ(rho_cgs, T_cgs, Ye, s);
        double Jye = GetJye(v(prho, k, j, i)*RHO, v(pye, k, j, i), s);
        //double Jye = d_opacity->GetJye(rho_cgs, T_cgs, Ye, s);
        Real Gcov_tetrad[4] = {-J * CPOWERDENS, 0., 0., 0.};
        Real Gcov_coord[4];
        Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);

        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = detG * Gcov_coord[mu - Gcov_lo];
        }
        v(Gye, k, j, i) = -detG * Jye * CDENSITY / CTIME;
        printf("J: %e Jcode: %e\n", J, J*CPOWERDENS);
        printf("ye: %e rho: %e\n", v(pye, k, j, i), v(prho, k, j, i)*RHO);
        printf("G: %e %e %e %e (%e)\n", detG * Gcov_coord[0],
          detG * Gcov_coord[1],
          detG * Gcov_coord[2],
          detG * Gcov_coord[3],
          detG*Jye*CDENSITY/CTIME);

//        printf("ceng/G0: %e ceng: %e G0: %e J: %e\n", v(ceng, k, j, i)/Gcov_coord[0],
//          v(ceng, k, j, i), Gcov_coord[0], J);
//        exit(-1);
      });

  return TaskStatus::complete;
}

} // namespace radiation
