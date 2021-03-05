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

// TODO(BRR) Utilities that should be moved
#define SMALL (1.e-200)
KOKKOS_INLINE_FUNCTION Real GetLorentzFactor(Real v[4],
                                             const Geometry::CoordinateSystem &system,
                                             CellLocation loc, const int k, const int j,
                                             const int i) {
  Real W = 1;
  Real gamma[Geometry::NDSPACE][Geometry::NDSPACE];
  system.Metric(loc, k, j, i, gamma);
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    for (int m = 1; m < Geometry::NDFULL; ++m) {
      W -= v[l] * v[m] * gamma[l - 1][m - 1];
    }
  }
  W = 1. / std::sqrt(std::abs(W) + SMALL);
  return W;
}

KOKKOS_INLINE_FUNCTION void GetFourVelocity(Real v[4],
                                            const Geometry::CoordinateSystem &system,
                                            CellLocation loc, const int k, const int j,
                                            const int i, Real u[Geometry::NDFULL]) {
  Real beta[Geometry::NDSPACE];
  Real W = GetLorentzFactor(v, system, loc, k, j, i);
  Real alpha = system.Lapse(loc, k, j, i);
  system.ContravariantShift(loc, k, j, i, beta);
  u[0] = W / (std::abs(alpha) + SMALL);
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    u[l] = W * v[l - 1] - u[0] * beta[l - 1];
  }
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("radiation");

  Params &params = physics->AllParams();

  const bool active = pin->GetBoolean("physics", "rad");
  params.Add("active", active);

  if (!active) {
    return physics;
  }

  std::vector<int> four_vec(1, 4);
  Metadata mfourforce = Metadata({Metadata::Cell, Metadata::OneCopy}, four_vec);
  physics->AddField("Gcov", mfourforce);

  Metadata mscalar = Metadata({Metadata::Cell, Metadata::OneCopy});
  physics->AddField("Gye", mscalar);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  return physics;
}

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({c::energy, c::momentum, c::ye, "Gcov", "Gye"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int ceng = imap[c::energy].first;
  const int cmom_lo = imap[c::density].first;
  const int cmom_hi = imap[c::density].second;
  const int cye = imap[c::ye].first;
  const int Gcov_lo = imap["Gcov"].first;
  const int Gcov_hi = imap["Gcov"].second;
  const int Gye = imap["Gye"].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyRadiationFourForce", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        printf("Applying four force %e %e %e %e\n", v(Gcov_lo, k, j, i),
               v(Gcov_lo + 1, k, j, i), v(Gcov_lo + 2, k, j, i), v(Gcov_lo + 3, k, j, i));
        exit(-1);
        v(ceng, k, j, i) -= v(Gcov_lo, k, j, i) * dt;
        v(cmom_lo, k, j, i) += v(Gcov_lo + 1, k, j, i) * dt;
        v(cmom_lo + 1, k, j, i) += v(Gcov_lo + 2, k, j, i) * dt;
        v(cmom_lo + 2, k, j, i) += v(Gcov_lo + 3, k, j, i) * dt;
        v(cye, k, j, i) += v(Gye, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

// Temporary cooling problem functions etc.
  enum class NeutrinoSpecies { Electron, ElectronAnti, Heavy };
  KOKKOS_INLINE_FUNCTION Real Getyf(Real Ye, NeutrinoSpecies s) {
    if (s == NeutrinoSpecies::Electron) {
      return 2.*Ye;
    } else if (s == NeutrinoSpecies::ElectronAnti) {
      return 1. - 2.*Ye;
    } else {
      return 0.;
    }
  }

TaskStatus CalculateRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars(
      {p::density, p::velocity, p::temperature, p::ye, c::energy, "Gcov", "Gye"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
  const int ptemp = imap[p::temperature].first;
  const int pye = imap[p::ye].first;
  const int ceng = imap[c::energy].first;
  const int Gcov_lo = imap["Gcov"].first;
  const int Gcov_hi = imap["Gcov"].second;
  const int Gye = imap["Gye"].first;

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

  // Temporary cooling problem parameters
  Real C = 1.;
  Real numax = 1.e17;
  Real numin = 1.e7;
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

        Real T_cgs = v(ptemp, k, j, i) * TEMP;
        Real ne_cgs = GetNumberDensity(v(prho, k, j, i) * RHO);
        Real Lambda_cgs =
            4. * M_PI * pc.kb * pow(ne_cgs, 2) * N / pc.h * pow(T_cgs, 1. / 2.);
            printf("4.*M_PI*pc.kb*pow*ne_cgs, 2) *N/pc.h = %e\n",
              4.*M_PI*pc.kb*pow(ne_cgs, 2) *N/pc.h);
        Real Lambda_code = Lambda_cgs * CPOWERDENS;

        Real Ac = pc.mp/(pc.h*v(prho, k, j, i) * RHO)*C*log(numax/numin);
        printf("rho = %e\n", v(prho, k, j, i) * RHO);
        Real Bc = C*(numax - numin);
        printf("Ac: %e Bc: %e\n", Ac, Bc);

        printf("Lambda_code = %e\n", Lambda_code);
        Real J = C*Getyf(v(pye, k, j, i), s)*(numax - numin);
        printf("C: %e yf: %e dnu: %e\n", C, Getyf(v(pye, k, j, i), s), numax-numin);
        printf("J = %e\n", J);
        exit(-1);

        Real Gcov_tetrad[4] = {Lambda_code, 0, 0, 0};
        Real Gcov_coord[4];
        Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);

        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = detG * Gcov_coord[mu - Gcov_lo];
        }
        v(Gye, k, j, i) = detG * 0. / dt;
      });

  return TaskStatus::complete;
}

} // namespace radiation
