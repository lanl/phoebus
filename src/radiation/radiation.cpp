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
  namespace iv = internal_variables;
  auto physics = std::make_shared<StateDescriptor>("radiation");

  Params &params = physics->AllParams();

  const bool active = pin->GetBoolean("physics", "rad");
  params.Add("active", active);

  if (!active) {
    return physics;
  }

  std::vector<int> four_vec(1, 4);
  Metadata mfourforce = Metadata({Metadata::Cell, Metadata::OneCopy}, four_vec);
  physics->AddField(iv::Gcov, mfourforce);

  Metadata mscalar = Metadata({Metadata::Cell, Metadata::OneCopy});
  physics->AddField(iv::Gye, mscalar);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  if (method == "monte_carlo" || method == "mocmc") {
    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("Particles", "rng_seed", 238947);
    physics->AddParam<>("rng_seed", rng_seed);
    RNGPool rng_pool(rng_seed);
    physics->AddParam<>("rng_pool", rng_pool);

    std::string swarm_name = "neutrinos";
    Metadata swarm_metadata;
    physics->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});
    physics->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);
  }

  return physics;
}

// TaskCollection radiation::Transport
// TODO(BRR) nvm this needs to be a PhoebusDriver member function
//TaskCollection Transport() {
//  TaskCollection tc;
//  TaskID none(0);
//  const double t0 = tm.time;
//}

TaskStatus TransportMCParticles(MeshBlockData <Real> *rc, const double t0, const double dt) {
  auto *pmb = rc->GetParentPointer().get();
  auto swarm = pmb->swarm_data.Get()->Get("neutrinos");
  int max_active_index = swarm->GetMaxActiveIndex();

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  const auto &vx = swarm->Get<Real>("vx").Get();
  const auto &vy = swarm->Get<Real>("vy").Get();
  const auto &vz = swarm->Get<Real>("vz").Get();

  return TaskStatus::complete;
}

// TaskStatus ApplyRadiationFourForce

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  namespace c = conserved_variables;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({c::energy, c::momentum, c::ye, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int ceng = imap[c::energy].first;
  const int cmom_lo = imap[c::density].first;
  const int cmom_hi = imap[c::density].second;
  const int cye = imap[c::ye].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyRadiationFourForce", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(ceng, k, j, i) += v(Gcov_lo, k, j, i) * dt;
        v(cmom_lo, k, j, i) += v(Gcov_lo + 1, k, j, i) * dt;
        v(cmom_lo + 1, k, j, i) += v(Gcov_lo + 2, k, j, i) * dt;
        v(cmom_lo + 2, k, j, i) += v(Gcov_lo + 3, k, j, i) * dt;
        v(cye, k, j, i) += v(Gye, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

// TODO(BRR) move this somewhere else
// Temporary cooling problem functions etc.
enum class NeutrinoSpecies { Electron, ElectronAnti, Heavy };
KOKKOS_INLINE_FUNCTION Real Getyf(Real Ye, NeutrinoSpecies s) {
  if (s == NeutrinoSpecies::Electron) {
    return 2. * Ye;
  } else if (s == NeutrinoSpecies::ElectronAnti) {
    return 1. - 2. * Ye;
  } else {
    return 0.;
  }
}

TaskStatus CalculateRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
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

        Real Ac = pc.mp / (pc.h * v(prho, k, j, i) * RHO) * C * log(numax / numin);
        Real Bc = C * (numax - numin);

        Real Gcov_tetrad[4] = {-Bc * Getyf(v(pye, k, j, i), s) * CPOWERDENS, 0, 0, 0};
        Real Gcov_coord[4];
        Tetrads.TetradToCoordCov(Gcov_tetrad, Gcov_coord);
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);

        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = detG * Gcov_coord[mu - Gcov_lo];
        }
        v(Gye, k, j, i) =
            -detG * v(prho, k, j, i) * (Ac / CTIME) * Getyf(v(pye, k, j, i), s);
      });

  return TaskStatus::complete;
}

} // namespace radiation
