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

#include "radiation.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

namespace radiation {

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

  std::vector<std::string> known_methods = {"cooling_function", "moment",
                                            "monte_carlo", "mocmc"};
  if (std::find(known_methods.begin(), known_methods.end(), method) ==
      known_methods.end()) {
    std::stringstream msg;
    msg << "Radiation method \"" << method << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  // Set which neutrino species to include in simulation
  bool do_nu_electron =
      pin->GetOrAddBoolean("radiation", "do_nu_electron", true);
  params.Add("do_nu_electron", do_nu_electron);
  bool do_nu_electron_anti =
      pin->GetOrAddBoolean("radiation", "do_nu_electron_anti", true);
  params.Add("do_nu_electron_anti", do_nu_electron_anti);
  bool do_nu_heavy = pin->GetOrAddBoolean("radiation", "do_nu_heavy", true);
  params.Add("do_nu_heavy", do_nu_heavy);

  // Initialize frequency discretization
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

  bool absorption = pin->GetOrAddBoolean("radiation", "absorption", true);
  params.Add("absorption", absorption);

  if (method == "monte_carlo") {
    std::string swarm_name = "monte_carlo";
    Metadata swarm_metadata({Metadata::Provides});
    physics->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});
    physics->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k0", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k1", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k2", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("k3", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);
    Metadata int_swarmvalue_metadata({Metadata::Integer});
    physics->AddSwarmValue("species", swarm_name, int_swarmvalue_metadata);

    Metadata mspecies_scalar = Metadata({Metadata::Cell, Metadata::OneCopy},
                                        std::vector<int>(1, NumRadiationTypes));
    physics->AddField("dNdlnu_max", mspecies_scalar);
    physics->AddField("dN", mspecies_scalar);
    physics->AddField("Ns", mspecies_scalar);

    std::vector<int> dNdlnu_size{NumRadiationTypes, nu_bins + 1};
    Metadata mdNdlnu =
        Metadata({Metadata::Cell, Metadata::OneCopy}, dNdlnu_size);
    physics->AddField("dNdlnu", mdNdlnu);

    Real tune_emiss = pin->GetOrAddReal("radiation", "tune_emiss", 1.);
    params.Add("tune_emiss", tune_emiss);

    int num_particles = pin->GetOrAddInteger("radiation", "num_particles", 100);
    params.Add("num_particles", num_particles);

    bool remove_emitted_particles =
        pin->GetOrAddBoolean("monte_carlo", "remove_emitted_particles", false);
    params.Add("remove_emitted_particles", remove_emitted_particles);
  }

  if (method == "monte_carlo" || method == "mocmc") {
    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("radiation", "rng_seed", 238947);
    physics->AddParam<>("rng_seed", rng_seed);
    RNGPool rng_pool(rng_seed);
    physics->AddParam<>("rng_pool", rng_pool);
  }

  if (method != "cooling_function") {
    physics->EstimateTimestepBlock = EstimateTimestepBlock;
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
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars(
      {c::energy, c::momentum, c::ye, iv::Gcov, iv::Gye});
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
      DEFAULT_LOOP_PATTERN, "ApplyRadiationFourForce", DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(ceng, k, j, i) += v(Gcov_lo, k, j, i) * dt;
        v(cmom_lo, k, j, i) += v(Gcov_lo + 1, k, j, i) * dt;
        v(cmom_lo + 1, k, j, i) += v(Gcov_lo + 2, k, j, i) * dt;
        v(cmom_lo + 2, k, j, i) += v(Gcov_lo + 3, k, j, i) * dt;
        v(cye, k, j, i) += v(Gye, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  // TODO(BRR) Can't just use dx^i/dx^0 = 1 for speed of light
  // TODO(BRR) add a cfl-like fudge factor to radiation
  auto &pars = pmb->packages.Get("fluid")->AllParams();
  Real min_dt;
  pmb->par_reduce(
      "Radiation::EstimateTimestep::1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        Real ldt = 0.0;
        Real csig = 1.0;
        for (int d = 0; d < ndim; d++) {
          lmin_dt =
              std::min(lmin_dt, 1.0 / (csig / coords.Dx(X1DIR + d, k, j, i)));
        }
      },
      Kokkos::Min<Real>(min_dt));
  const auto &cfl = pars.Get<Real>("cfl");
  return cfl * min_dt;
}

} // namespace radiation
