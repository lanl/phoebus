//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

  Real nu_min = pin->GetReal("radiation", "nu_min");
  params.Add("nu_min", nu_min);
  Real nu_max = pin->GetReal("radiation", "nu_max");
  params.Add("nu_max", nu_max);
  int nu_bins = pin->GetInteger("radiation", "nu_bins");
  params.Add("nu_bins", nu_bins);
  Real dlnu = (log(nu_max) - log(nu_min)) / nu_bins;
  params.Add("dlnu", dlnu);
  ParArray1D<Real> nusamp("Frequency grid", nu_bins + 1);
  auto nusamp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), nusamp);
  for (int n = 0; n < nu_bins + 1; n++) {
    nusamp_h(n) = exp(log(nu_min) + n * dlnu);
  }
  Kokkos::deep_copy(nusamp, nusamp_h);
  params.Add("nusamp", nusamp);

  int num_species = pin->GetOrAddInteger("radiation", "num_species", 1);
  params.Add("num_species", num_species);

  std::vector<std::string> known_methods = {"cooling_function", "moment", "monte_carlo",
                                            "mocmc"};
  if (std::find(known_methods.begin(), known_methods.end(), method) ==
      known_methods.end()) {
    std::stringstream msg;
    msg << "Radiation method \"" << method << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  if (method == "monte_carlo") {
    std::string swarm_name = "monte_carlo";
    Metadata swarm_metadata;
    physics->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});
    physics->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k0", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k1", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k2", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k3", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);
    Metadata int_swarmvalue_metadata({Metadata::Integer});

    physics->AddField("dNdlnu_max", mscalar);
    physics->AddField("dN", mscalar);

    std::vector<int> dNdlnu_size(1, nu_bins + 1);
    Metadata mdNdlnu = Metadata({Metadata::Cell, Metadata::OneCopy}, dNdlnu_size);
    physics->AddField("dNdlnu", mdNdlnu);

    Real tune_emiss = pin->GetOrAddReal("radiation", "tune_emiss", 1.);
    params.Add("tune_emiss", tune_emiss);
  }

  if (method == "monte_carlo" || method == "mocmc") {
    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("Particles", "rng_seed", 238947);
    physics->AddParam<>("rng_seed", rng_seed);
    RNGPool rng_pool(rng_seed);
    physics->AddParam<>("rng_pool", rng_pool);
  }

  return physics;
}

TaskStatus TransportMCParticles(MeshBlockData<Real> *rc, const double t0,
                                const double dt) {
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
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
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
        printf("G0:  %e\n", v(Gcov_lo, k, j, i));
        printf("Gye: %e\n", v(Gye, k, j, i));
        exit(-1);
      });

  return TaskStatus::complete;
}

} // namespace radiation
