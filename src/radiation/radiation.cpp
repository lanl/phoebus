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
    const bool moments_active = false;
    params.Add("moments_active", moments_active);
    return physics;
  }

  std::vector<int> four_vec(1, 4);
  Metadata mfourforce = Metadata({Metadata::Cell, Metadata::OneCopy}, four_vec);
  physics->AddField(iv::Gcov, mfourforce);

  Metadata mscalar = Metadata({Metadata::Cell, Metadata::OneCopy});
  physics->AddField(iv::Gye, mscalar);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  std::vector<std::string> known_methods = {"cooling_function", "moment_m1", "moment_eddington",
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

  // Boundary conditions
  const std::string bc_vars = pin->GetOrAddString("phoebus/mesh", "bc_vars", "conserved");
  params.Add("bc_vars", bc_vars);

  // Set radiation cfl factor
  Real cfl = pin->GetOrAddReal("radiation", "cfl", 0.8);
  params.Add("cfl", cfl);

  // Get fake value for integrated BB for testing
  Real B_fake = pin->GetOrAddReal("radiation", "B_fake", 1.0);
  params.Add("B_fake", B_fake);
  bool use_B_fake = pin->GetOrAddBoolean("radiation", "use_B_fake", false);
  params.Add("use_B_fake", use_B_fake);
  Real scattering_fraction = pin->GetOrAddReal("radiation", "scattering_fraction", 0.0);
  params.Add("scattering_fraction", scattering_fraction);

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

    // Parameters controlling automatic sampling resolution.
    // This system targets 1 scattering per light crossing time.
    // This explicit Monte Carlo method is not accurate once optical depths per
    // zone become >~ 1.
    int num_particles = pin->GetOrAddInteger("radiation", "num_particles", 100);
    params.Add("num_particles", num_particles);

    Real num_total = 0.;
    params.Add("num_total", num_total, true);

    Real num_emitted = 0.;
    params.Add("num_emitted", num_emitted, true);

    Real num_absorbed = 0.;
    params.Add("num_absorbed", num_absorbed, true);

    std::string tuning = pin->GetOrAddString("radiation", "tuning", "static");
    params.Add("tuning", tuning);

    Real dt_tune_emission = pin->GetOrAddReal("radiation", "dt_tune_emission", 1.);
    params.Add("dt_tune_emission", dt_tune_emission);

    // Time of first tuning. Make > dt_tune_emission to avoid tuning to a transient
    Real t_tune_emission = pin->GetOrAddReal("radiation", "t_tune_emission", dt_tune_emission);
    PARTHENON_REQUIRE(t_tune_emission >= dt_tune_emission, "Must wait at least dt_tune_emission before tuning!");
    params.Add("t_tune_emission", t_tune_emission, true);

    // Tuning parameter for particle emission
    Real tune_emission = pin->GetOrAddReal("radiation", "tune_emission", 1.);
    PARTHENON_REQUIRE(tune_emission > 0.0, "tune_emission must be > 0 for resolution controls!");
    params.Add("tune_emission", tune_emission, true);

    Real num_scattered = 0.;
    params.Add("num_scattered", num_scattered, true);

    // Approximately the light crossing time for the simulation volume.
    Real dt_tune_scattering = pin->GetOrAddReal("radiation", "dt_tune_scattering", 100.);
    params.Add("dt_tune_scattering", dt_tune_scattering);

    // Time of first tuning. Make > dt_tune_scattering to avoid tuning to a transient
    Real t_tune_scattering = pin->GetOrAddReal("radiation", "t_tune_scattering", dt_tune_scattering);
    PARTHENON_REQUIRE(t_tune_scattering >= dt_tune_scattering, "Must wait at least dt_tune_scattering before tuning!");
    params.Add("t_tune_scattering", t_tune_scattering, true);

    // Tuning parameter for biased particle scattering
    Real tune_scattering = pin->GetOrAddReal("radiation", "tune_scattering", 1.0);
    PARTHENON_REQUIRE(tune_scattering > 0.0, "tune_scattering must be > 0 for resolution controls!");
    params.Add("tune_scattering", tune_scattering, true);

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

//  physics->AddField(c::E, mspecies_scalar_cons);
//  physics->AddField(c::F, mspecies_three_vector_cons);

//  physics->AddField(p::J, mspecies_scalar);
//  physics->AddField(p::H, mspecies_three_vector);

  bool moments_active = false;
  if ((method == "mocmc") || (method == "moment_m1") || (method == "moment_eddington")) {
    moments_active = true;

    namespace p = radmoment_prim;
    namespace c = radmoment_cons;
    namespace i = radmoment_internal;

    int ndim = 3;
    //if (pin->GetInteger("parthenon/mesh", "nx3") > 1) ndim = 3;
    //else if (pin->GetInteger("parthenon/mesh", "nx2") > 1) ndim = 2;

    Metadata mspecies_three_vector = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Derived,
                                              Metadata::Intensive, Metadata::FillGhost, Metadata::Vector},
                                              std::vector<int>{NumRadiationTypes, 3});
    Metadata mspecies_scalar = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Derived,
                                              Metadata::Intensive, Metadata::FillGhost},
                                              std::vector<int>{NumRadiationTypes});

    Metadata mspecies_three_vector_cons = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                                                   Metadata::Intensive, Metadata::WithFluxes, Metadata::FillGhost, Metadata::Vector},
                                                   std::vector<int>{NumRadiationTypes, 3});
    Metadata mspecies_scalar_cons = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                                              Metadata::Intensive, Metadata::WithFluxes, Metadata::FillGhost},
                                              std::vector<int>{NumRadiationTypes});

    physics->AddField(c::E, mspecies_scalar_cons);
    physics->AddField(c::F, mspecies_three_vector_cons);

    physics->AddField(p::J, mspecies_scalar);
    physics->AddField(p::H, mspecies_three_vector);

    // Fields for saving guesses for NR iteration in the radiation Con2Prim type solve
    physics->AddField(i::xi, mspecies_scalar);
    physics->AddField(i::phi, mspecies_scalar);

    // Fields for cell edge reconstruction
    /// TODO: (LFR) The amount of storage can likely be reduced, but maybe at the expense of more dependency
    Metadata mrecon = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
			     std::vector<int>{NumRadiationTypes, 4, ndim});
    Metadata mrecon_v = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
			     std::vector<int>{3, ndim});
    physics->AddField(i::ql, mrecon);
    physics->AddField(i::qr, mrecon);
    physics->AddField(i::ql_v, mrecon_v);
    physics->AddField(i::qr_v, mrecon_v);

    // Add variable for calculating gradients of rest frame energy density
    Metadata mdJ = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
			     std::vector<int>{NumRadiationTypes, ndim, ndim});
    physics->AddField(i::dJ, mdJ);

    // Add variables for source functions
    Metadata mSourceVar = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
			     std::vector<int>{NumRadiationTypes});
    physics->AddField(i::kappaJ, mSourceVar);
    physics->AddField(i::kappaH, mSourceVar);
    physics->AddField(i::JBB, mSourceVar);

    physics->FillDerivedBlock = MomentCon2Prim<MeshBlockData<Real>>;
  }

  params.Add("moments_active", moments_active);

  if (method != "cooling_function") {
    physics->EstimateTimestepBlock = EstimateTimestepBlock;
  }

  return physics;
}

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
  auto &pars = pmb->packages.Get("radiation")->AllParams();
  Real min_dt;
  pmb->par_reduce(
      "Radiation::EstimateTimestep::1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
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
