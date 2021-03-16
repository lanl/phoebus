//#include <parthenon/package.hpp>
//#include <utils/error_checking.hpp>
//using namespace parthenon::package::prelude;

#include "radiation.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

#include "opacity.hpp"

//#include "compile_constants.hpp"
//#include "phoebus_utils/unit_conversions.hpp"
//#include "utils/constants.hpp"

namespace radiation {

TaskStatus CalculateCoolingFunctionFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars(
      {p::density, p::velocity, p::temperature, p::ye, c::energy, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
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

  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  auto geom = Geometry::GetCoordinateSystem(rc);

  // Temporary cooling problem parameters
  //Real C = 1.;
  //Real numax = 1.e17;
  //Real numin = 1.e7;
  NeutrinoSpecies s = NeutrinoSpecies::Electron;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateRadiationForce", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real Gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
        Real Ucon[4];
        Real vel[4] = {0, v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
        GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
        Geometry::Tetrads Tetrads(Ucon, Gcov);

        //Real Ac = pc.mp / (pc.h * v(prho, k, j, i) * RHO) * C * log(numax / numin);
        //Real Bc = C * (numax - numin);

        //Real Gcov_tetrad[4] = {-Bc * Getyf(v(pye, k, j, i), s) * CPOWERDENS, 0, 0, 0};
        double J = GetJ(v(prho, k, j, i) * RHO, v(pye, k, j, i), s);
        double Jye = GetJye(v(prho, k, j, i), v(pye, k, j, i), s);
        Real Gcov_tetrad[4] = {-J*CPOWERDENS, 0., 0., 0.};
        Real Gcov_coord[4];
        Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);

        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = detG * Gcov_coord[mu - Gcov_lo];
        }
        v(Gye, k, j, i) = -detG * Jye * CDENSITY / CTIME;
//            -detG * v(prho, k, j, i) * (Ac / CTIME) * Getyf(v(pye, k, j, i), s);
      });

  return TaskStatus::complete;
}

} // namespace radiation
