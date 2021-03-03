//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include "radiation.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

namespace radiation {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("radiation");

  Params &params = physics->AllParams();

  const bool active = pin->GetBoolean("physics", "rad");
  params.Add("active", active);

  if (!active) {
    return physics;
  }

  std::vector<int> four_vec(1,4);
  Metadata mfourforce = Metadata({Metadata::Cell, Metadata::OneCopy}, four_vec);
  physics->AddField("Gcov", mfourforce);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  return physics;
}

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({c::energy, c::momentum, "Gcov"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int ceng    = imap[c::energy].first;
  const int cmom_lo = imap[c::density].first;
  const int cmom_hi = imap[c::density].second;
  const int Gcov_lo = imap["Gcov"].first;
  const int Gcov_hi = imap["Gcov"].second;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyRadiationFourForce", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(ceng, k, j, i) -= v(Gcov_lo, k, j, i)*dt;
        v(cmom_lo, k, j, i) += v(Gcov_lo + 1, k, j, i);
        v(cmom_lo + 1, k, j, i) += v(Gcov_lo + 2, k, j, i);
        v(cmom_lo + 2, k, j, i) += v(Gcov_lo + 3, k, j, i);
      });

  return TaskStatus::complete;
}

TaskStatus CalculateRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({p::density, p::temperature, c::energy, "Gcov"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int ptemp = imap[p::temperature].first;
  const int ceng = imap[c::energy].first;
  const int Gcov_lo = imap["Gcov"].first;
  const int Gcov_hi = imap["Gcov"].second;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");

  constexpr double N = 5.4e-39; // cgs

  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMP = unit_conv.GetTemperatureCodeToCGS();

  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateRadiationForce", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {

        Real Gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
        Real Ucon[4] = {-1, 0, 0, 0};
        Real Trial[4] = {0, 1, 0, 0};
        Geometry::Tetrad tetrad(Ucon, Trial, Gcov);
        exit(-1);

        double T_cgs = v(ptemp, k, j, i) * TEMP;
        double ne_cgs = GetNumberDensity(v(prho, k, j, i) * RHO);
        double Lambda_cgs =
            4. * M_PI * pc.kb * pow(ne_cgs, 2) * N / pc.h * pow(T_cgs, 1. / 2.);
        double Lambda_code = Lambda_cgs * CPOWERDENS;

        v(Gcov_lo, k, j, i) = Lambda_code;
        for (int mu = Gcov_lo + 1; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
      });

  return TaskStatus::complete;
}

} // namespace radiation
