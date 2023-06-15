// © 2021-2022. Triad National Security, LLC. All rights reserved
// This program was produced under U.S. Government contract
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
#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/local_three_geometry.hpp"
#include "reconstruction.hpp"

#include "closure.hpp"

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

  auto unit_conv = phoebus::UnitConversions(pin);

  std::vector<int> four_vec(1, 4);
  Metadata mfourforce = Metadata({Metadata::Cell, Metadata::OneCopy}, four_vec);
  physics->AddField(iv::Gcov, mfourforce);

  Metadata mscalar = Metadata({Metadata::Cell, Metadata::OneCopy});
  physics->AddField(iv::Gye, mscalar);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  std::set<std::string> known_methods = {"cooling_function", "moment_m1",
                                         "moment_eddington", "monte_carlo", "mocmc"};
  if (!known_methods.count(method)) {
    std::stringstream msg;
    msg << "Radiation method \"" << method << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }

  bool is_frequency_dependent = (method == "monte_carlo" || method == "mocmc");

  // Set which neutrino species to include in simulation
  bool do_nu_electron = pin->GetOrAddBoolean("radiation", "do_nu_electron", true);
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

  // Initialize frequency discretization
  Real nu_min;
  Real nu_max;
  int nu_bins;
  Real dlnu;
  if (is_frequency_dependent) {
    nu_min = pin->GetReal("radiation", "nu_min") / unit_conv.GetTimeCGSToCode();
    ;
    params.Add("nu_min", nu_min);
    nu_max = pin->GetReal("radiation", "nu_max") / unit_conv.GetTimeCGSToCode();
    ;
    params.Add("nu_max", nu_max);
    nu_bins = pin->GetInteger("radiation", "nu_bins");
    params.Add("nu_bins", nu_bins);
    dlnu = (log(nu_max) - log(nu_min)) / nu_bins;
    params.Add("dlnu", dlnu);
  }

  std::vector<RadiationType> species;
  if (do_nu_electron) {
    species.push_back(RadiationType::NU_ELECTRON);
  }
  if (do_nu_electron_anti) {
    species.push_back(RadiationType::NU_ELECTRON_ANTI);
  }
  if (do_nu_heavy) {
    species.push_back(RadiationType::NU_HEAVY);
  }
  const int num_species = species.size();
  params.Add("num_species", num_species);
  params.Add("species", species);

  bool absorption = pin->GetOrAddBoolean("radiation", "absorption", true);
  params.Add("absorption", absorption);

  if (method == "cooling_function") {
    const bool do_liebendorfer =
        pin->GetOrAddBoolean("radiation", "do_liebendorfer", false);
    const bool do_lightbulb = pin->GetOrAddBoolean("radiation", "do_lightbulb", false);
    const Real lum = pin->GetOrAddReal("radiation", "lum", 4.0);
    params.Add("do_liebendorfer", do_liebendorfer);
    params.Add("do_lightbulb", do_lightbulb);
    params.Add("lum", lum);
    if (do_lightbulb) {
      physics->AddField(iv::GcovHeat, mscalar);
      physics->AddField(iv::GcovCool, mscalar);
      physics->AddField(iv::compweight, mscalar);
      std::string eos_type = pin->GetString("eos", "type");
      if (eos_type != singularity::StellarCollapse::EosType()) {
        PARTHENON_THROW("Lightbulb only supported with stellar collapse EOS");
      }
      Metadata m({Metadata::Cell, Metadata::OneCopy});
      physics->AddField(iv::tau, m);
      parthenon::AllReduce<bool> do_gain_reducer;
      bool always_gain = pin->GetOrAddBoolean("radiation", "always_gain", false);
      do_gain_reducer.val = always_gain;
      params.Add("do_gain_reducer", do_gain_reducer, true);
      params.Add("always_gain", always_gain);
    }
  }

  if (method == "mocmc") {
    std::string swarm_name = "mocmc";
    Metadata swarm_metadata({Metadata::Provides});
    physics->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});
    physics->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("mu_lo", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("mu_hi", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("phi_lo", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("phi_hi", swarm_name, real_swarmvalue_metadata);
    Metadata fourv_swarmvalue_metadata({Metadata::Real}, std::vector<int>{4});
    physics->AddSwarmValue("ncov", swarm_name, fourv_swarmvalue_metadata);
    Metadata Inu_swarmvalue_metadata({Metadata::Real},
                                     std::vector<int>{num_species, nu_bins});
    physics->AddSwarmValue("Inuinv", swarm_name, Inu_swarmvalue_metadata);

    // Boundary temperatures for outflow sample boundary conditions
    const std::string ix1_bc = pin->GetOrAddString("phoebus", "ix1_bc", "None");
    if (ix1_bc == "outflow") {
      params.Add("ix1_bc", MOCMCBoundaries::outflow);
    } else if (ix1_bc == "fixed_temp") {
      params.Add("ix1_bc", MOCMCBoundaries::fixed_temp);
    } else {
      params.Add("ix1_bc", MOCMCBoundaries::periodic);
    }
    if (ix1_bc == "fixed_temp") {
      const Real ix1_temp = pin->GetOrAddReal("phoebus", "ix1_temp", 0.) *
                            unit_conv.GetTemperatureCGSToCode();
      params.Add("ix1_temp", ix1_temp);
    }

    const std::string ox1_bc = pin->GetOrAddString("phoebus", "ox1_bc", "None");
    if (ox1_bc == "outflow") {
      params.Add("ox1_bc", MOCMCBoundaries::outflow);
    } else if (ox1_bc == "fixed_temp") {
      params.Add("ox1_bc", MOCMCBoundaries::fixed_temp);
    } else {
      params.Add("ox1_bc", MOCMCBoundaries::periodic);
    }
    if (ox1_bc == "fixed_temp") {
      const Real ox1_temp = pin->GetOrAddReal("phoebus", "ox1_temp", 0.) *
                            unit_conv.GetTemperatureCGSToCode();
      params.Add("ox1_temp", ox1_temp);
    }

    const int nsamp_per_zone =
        pin->GetOrAddInteger("radiation/mocmc", "nsamp_per_zone", 32);
    params.Add("nsamp_per_zone", nsamp_per_zone);

    ParArray1D<Real> nusamp("Frequency grid", nu_bins);
    auto nusamp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), nusamp);
    for (int n = 0; n < nu_bins; n++) {
      nusamp_h(n) = exp(log(nu_min) + (n + 0.5) * dlnu);
    }
    Kokkos::deep_copy(nusamp, nusamp_h);
    params.Add("nusamp", nusamp);

    std::string mocmc_recon_str =
        pin->GetOrAddString("radiation/mocmc", "recon", "kdgrid");
    MOCMCRecon mocmc_recon;
    if (mocmc_recon_str == "kdgrid") {
      mocmc_recon = MOCMCRecon::kdgrid;
    } else {
      std::stringstream msg;
      msg << "MOCMC reconstruction method \"" << mocmc_recon_str << "\" not recognized!";
      PARTHENON_FAIL(msg);
    }
    params.Add("mocmc_recon", mocmc_recon);

    std::vector<int> Inu_size{num_species, nu_bins};
    Metadata mInu = Metadata({Metadata::Cell, Metadata::OneCopy}, Inu_size);
    physics->AddField(mocmc_internal::Inu0, mInu);
    physics->AddField(mocmc_internal::Inu1, mInu);
    physics->AddField(mocmc_internal::jinvs, mInu);
    physics->AddField(iv::Gye, mscalar);

    Real num_total = 0.;
    params.Add("num_total", num_total, true);
  }

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

    Metadata mspecies_scalar =
        Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>{num_species});
    physics->AddField("dNdlnu_max", mspecies_scalar);
    physics->AddField("dN", mspecies_scalar);
    physics->AddField("Ns", mspecies_scalar);

    std::vector<int> dNdlnu_size{num_species, params.Get<int>("nu_bins") + 1};
    Metadata mdNdlnu = Metadata({Metadata::Cell, Metadata::OneCopy}, dNdlnu_size);
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
    Real t_tune_emission =
        pin->GetOrAddReal("radiation", "t_tune_emission", dt_tune_emission);
    PARTHENON_REQUIRE(t_tune_emission >= dt_tune_emission,
                      "Must wait at least dt_tune_emission before tuning!");
    params.Add("t_tune_emission", t_tune_emission, true);

    // Tuning parameter for particle emission
    Real tune_emission = pin->GetOrAddReal("radiation", "tune_emission", 1.);
    PARTHENON_REQUIRE(tune_emission > 0.0,
                      "tune_emission must be > 0 for resolution controls!");
    params.Add("tune_emission", tune_emission, true);

    Real num_scattered = 0.;
    params.Add("num_scattered", num_scattered, true);

    // Approximately the light crossing time for the simulation volume.
    Real dt_tune_scattering = pin->GetOrAddReal("radiation", "dt_tune_scattering", 100.);
    params.Add("dt_tune_scattering", dt_tune_scattering);

    // Time of first tuning. Make > dt_tune_scattering to avoid tuning to a transient
    Real t_tune_scattering =
        pin->GetOrAddReal("radiation", "t_tune_scattering", dt_tune_scattering);
    PARTHENON_REQUIRE(t_tune_scattering >= dt_tune_scattering,
                      "Must wait at least dt_tune_scattering before tuning!");
    params.Add("t_tune_scattering", t_tune_scattering, true);

    // Tuning parameter for biased particle scattering
    Real tune_scattering = pin->GetOrAddReal("radiation", "tune_scattering", 1.0);
    PARTHENON_REQUIRE(tune_scattering > 0.0,
                      "tune_scattering must be > 0 for resolution controls!");
    params.Add("tune_scattering", tune_scattering, true);

    bool remove_emitted_particles =
        pin->GetOrAddBoolean("monte_carlo", "remove_emitted_particles", false);
    params.Add("remove_emitted_particles", remove_emitted_particles);

    ParArray1D<Real> nusamp("Frequency grid", nu_bins + 1);
    auto nusamp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), nusamp);
    for (int n = 0; n < nu_bins + 1; n++) {
      nusamp_h(n) = exp(log(nu_min) + n * dlnu);
    }
    Kokkos::deep_copy(nusamp, nusamp_h);
    params.Add("nusamp", nusamp);
  }

  if (method == "monte_carlo" || method == "mocmc") {
    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("radiation", "rng_seed", 238947);
    physics->AddParam<>("rng_seed", rng_seed);
    RNGPool rng_pool(rng_seed);
    physics->AddParam<>("rng_pool", rng_pool);
  }

  bool moments_active = false;
  if ((method == "mocmc") || (method == "moment_m1") || (method == "moment_eddington")) {
    moments_active = true;

    namespace p = radmoment_prim;
    namespace c = radmoment_cons;
    namespace i = radmoment_internal;

    std::string closure_strategy_str =
        pin->GetOrAddString("radiation", "closure_c2p_strategy", "robust");
    ClosureCon2PrimStrategy closure_strategy;
    if (closure_strategy_str == "robust") {
      closure_strategy = ClosureCon2PrimStrategy::robust;
    } else if (closure_strategy_str == "frail") {
      closure_strategy = ClosureCon2PrimStrategy::frail;
    } else {
      PARTHENON_THROW("Invalid closure_c2p_strategy option. Choose from [robust,frail]");
    }

    ClosureRuntimeSettings closure_runtime_params{closure_strategy};
    params.Add("closure_runtime_params", closure_runtime_params);

    std::string recon = pin->GetOrAddString("radiation", "recon", "linear");
    PhoebusReconstruction::ReconType rt = PhoebusReconstruction::ReconType::linear;
    if (recon == "weno5" || recon == "weno5z") {
      PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                               "weno5 requires 4+ ghost cells");
      rt = PhoebusReconstruction::ReconType::weno5z;
    } else if (recon == "mp5") {
      PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                               "mp5 requires 4+ ghost cells");
      if (cfl > 0.4) {
        PARTHENON_WARN("mp5 often requires smaller cfl numbers for stability");
      }
      rt = PhoebusReconstruction::ReconType::mp5;
    } else if (recon == "linear") {
      rt = PhoebusReconstruction::ReconType::linear;
    } else if (recon == "constant") {
      rt = PhoebusReconstruction::ReconType::constant;
    } else {
      PARTHENON_THROW("Invalid Reconstruction option.  Choose from "
                      "[constant,linear,mp5,weno5,weno5z]");
    }
    params.Add("Recon", rt);

    const std::string recon_fixup_strategy_str =
        pin->GetOrAddString("radiation", "recon_fixup_strategy", "bounds");
    ReconFixupStrategy recon_fixup_strategy;
    if (recon_fixup_strategy_str == "none") {
      recon_fixup_strategy = ReconFixupStrategy::none;
    } else if (recon_fixup_strategy_str == "bounds") {
      recon_fixup_strategy = ReconFixupStrategy::bounds;
    } else {
      PARTHENON_FAIL("\"radiation/recon_fixup_strategy\" option unrecognized!");
    }
    params.Add("recon_fixup_strategy", recon_fixup_strategy);

    const std::string src_solver_name =
        pin->GetOrAddString("radiation", "src_solver", "oned");
    SourceSolver src_solver;
    if (src_solver_name == "zerod") {
      src_solver = SourceSolver::zerod;
    } else if (src_solver_name == "oned") {
      src_solver = SourceSolver::oned;
    } else if (src_solver_name == "fourd") {
      src_solver = SourceSolver::fourd;
    } else {
      PARTHENON_FAIL("\"radiation/src_solver\" option unrecognized!");
    }
    params.Add("src_solver", src_solver);

    const bool src_use_oned_backup =
        pin->GetOrAddBoolean("radiation", "src_use_oned_backup", false);
    params.Add("src_use_oned_backup", src_use_oned_backup);

    Real src_rootfind_eps = pin->GetOrAddReal("radiation", "src_rootfind_eps", 1.e-8);
    params.Add("src_rootfind_eps", src_rootfind_eps);

    Real src_rootfind_tol = pin->GetOrAddReal("radiation", "src_rootfind_tol", 1.e-12);
    params.Add("src_rootfind_tol", src_rootfind_tol);

    int src_rootfind_maxiter =
        pin->GetOrAddInteger("radiation", "src_rootfind_maxiter", 100);
    params.Add("src_rootfind_maxiter", src_rootfind_maxiter);

    std::string oned_fixup_strategy_str =
        pin->GetOrAddString("radiation", "oned_fixup_strategy", "none");
    OneDFixupStrategy oned_fixup_strategy;
    if (oned_fixup_strategy_str == "none") {
      oned_fixup_strategy = OneDFixupStrategy::none;
    } else if (oned_fixup_strategy_str == "ignore_dJ") {
      oned_fixup_strategy = OneDFixupStrategy::ignore_dJ;
    } else if (oned_fixup_strategy_str == "ignore_all") {
      oned_fixup_strategy = OneDFixupStrategy::ignore_all;
    } else {
      PARTHENON_FAIL("radiation/oned_fixup_strategy has an invalid entry!");
    }
    params.Add("oned_fixup_strategy", oned_fixup_strategy);

    // Required to be 3D in dJ calculation in radiation::ReconstructEdgeStates
    const int ndim = 3;

    Metadata mspecies_three_vector =
        Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Derived,
                  Metadata::Intensive, Metadata::FillGhost, Metadata::Vector},
                 std::vector<int>{num_species, 3});
    Metadata mspecies_scalar =
        Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Derived,
                  Metadata::Intensive, Metadata::FillGhost},
                 std::vector<int>{num_species});

    Metadata mspecies_three_vector_cons = Metadata(
        {Metadata::Cell, Metadata::Independent, Metadata::Conserved, Metadata::Intensive,
         Metadata::WithFluxes, Metadata::FillGhost, Metadata::Vector},
        std::vector<int>{num_species, 3});
    Metadata mspecies_scalar_cons =
        Metadata({Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                  Metadata::Intensive, Metadata::WithFluxes, Metadata::FillGhost},
                 std::vector<int>{num_species});

    physics->AddField(c::E, mspecies_scalar_cons);
    physics->AddField(c::F, mspecies_three_vector_cons);

    physics->AddField(p::J, mspecies_scalar);
    physics->AddField(p::H, mspecies_three_vector);

    // Fields for saving guesses for NR iteration in the radiation Con2Prim type solve
    physics->AddField(i::xi, mspecies_scalar);
    physics->AddField(i::phi, mspecies_scalar);

    // Fields for cell edge reconstruction
    /// TODO: (LFR) The amount of storage can likely be reduced, but maybe at the expense
    /// of more dependency
    int nrecon = 4;
    if (method == "mocmc") {
      nrecon += 9; // Reconstruct conTilPi // TODO(BRR) Use 6 elements by symmetry
    }
    Metadata mrecon = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                               std::vector<int>{num_species, nrecon, ndim});
    Metadata mrecon_v = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                 std::vector<int>{3, ndim});
    physics->AddField(i::ql, mrecon);
    physics->AddField(i::qr, mrecon);
    physics->AddField(i::ql_v, mrecon_v);
    physics->AddField(i::qr_v, mrecon_v);

    // Add variable for calculating gradients of rest frame energy density
    Metadata mdJ = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                            std::vector<int>{num_species, ndim, ndim});
    physics->AddField(i::dJ, mdJ);

    // Add variables for source functions
    Metadata mSourceVar = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                                   std::vector<int>{num_species});
    physics->AddField(i::kappaJ, mSourceVar);
    physics->AddField(i::kappaH, mSourceVar);
    physics->AddField(i::JBB, mSourceVar);

    // this fail flag should really be an enum or something
    // but parthenon doesn't yet support that kind of thing
    Metadata m_scalar = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Derived,
                                  Metadata::Intensive, Metadata::FillGhost});
    physics->AddField(i::c2pfail, m_scalar);
    physics->AddField(i::srcfail, m_scalar);

    // Make Eddington tensor an independent quantity for MOCMC to supply
    if (method == "mocmc") {
      Metadata mspecies_three_tensor = Metadata(
          {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost},
          std::vector<int>{num_species, 3, 3});

      physics->AddField(i::tilPi, mspecies_three_tensor);
      physics->AddField(mocmc_internal::dnsamp, mscalar);
    }

#if SET_FLUX_SRC_DIAGS
    // DIAGNOSTIC STUFF FOR DEBUGGING
    std::vector<int> spec_four_vec{num_species, 4};
    Metadata mdiv = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                              Metadata::Derived, Metadata::OneCopy},
                             std::vector<int>{num_species, 4});
    physics->AddField(diagnostic_variables::r_divf, mdiv);
    Metadata mdiag = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                               Metadata::Derived, Metadata::OneCopy},
                              std::vector<int>{num_species, 4});
    physics->AddField(diagnostic_variables::r_src_terms, mdiag);
    printf("name: %s\n", diagnostic_variables::r_src_terms);
    printf("Added fields!\n");
#endif

    // Use PostFillDerivedBlock to guarantee that fluid, if active, was already inverted
    // (we need updated fluid primitive velocities to calculate radiation primitives)
    physics->PostFillDerivedBlock = MomentCon2Prim<MeshBlockData<Real>>;
  }

  params.Add("moments_active", moments_active);

  if (method != "cooling_function") {
    physics->EstimateTimestepBlock = EstimateTimestepBlock;
  }
  return physics;
}

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt) {
  PARTHENON_REQUIRE(USE_VALENCIA, "Covariant GRMHD formulation not supported!");

  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars(
      {c::density, c::energy, c::momentum, c::ye, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int crho = imap[c::density].first;
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
        v(ceng, k, j, i) -= v(Gcov_lo, k, j, i) * dt;
        v(cmom_lo, k, j, i) += v(Gcov_lo + 1, k, j, i) * dt;
        v(cmom_lo + 1, k, j, i) += v(Gcov_lo + 2, k, j, i) * dt;
        v(cmom_lo + 2, k, j, i) += v(Gcov_lo + 3, k, j, i) * dt;
        v(cye, k, j, i) += v(Gye, k, j, i) * dt;
      });
  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  namespace ir = radmoment_internal;
  namespace p = fluid_prim;

  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  auto geom = Geometry::GetCoordinateSystem(rc);

  PackIndexMap imap;
  std::vector<std::string> vars{ir::kappaH, p::velocity};
  auto v = rc->PackVariables(vars, imap);
  auto idx_v = imap.GetFlatIdx(p::velocity);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH, false);

  auto num_species = rad->Param<int>("num_species");

  auto &pars = pmb->packages.Get("radiation")->AllParams();
  Real min_dt;
  pmb->par_reduce(
      "Radiation::EstimateTimestep::1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        Vec con_beta;
        geom.ContravariantShift(CellLocation::Cent, k, j, i, con_beta.data);
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, k, j, i, cov_gamma.data);
        Tens2 con_gamma;
        geom.MetricInverse(CellLocation::Cent, k, j, i, con_gamma.data);
        const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);

        for (int ispec = 0; ispec < num_species; ispec++) {

          const Real kappaH = idx_kappaH.IsValid() ? v(idx_kappaH(ispec), k, j, i) : 0.;

          for (int d = 0; d < ndim; d++) {
            // Signal speeds (assume (i.e. somewhat overestimate, esp. for large opt.
            // depth) cs_rad = 1)
            const Real sigp = alpha * std::sqrt(con_gamma(d, d)) - con_beta(d);
            const Real sigm = -alpha * std::sqrt(con_gamma(d, d)) - con_beta(d);
            const Real asym_sigl = alpha * v(idx_v(d), k, j, i) - con_beta(d);
            const Real rad_speed = std::max<Real>(std::fabs(sigm), std::fabs(sigp));
            const Real asym_speed = std::fabs(asym_sigl);

            const Real dx =
                coords.CellWidthFA(X1DIR + d, k, j, i) * sqrt(cov_gamma(d, d));
            const Real a = tanh(ratio(1.0, std::pow(std::abs(kappaH * dx), 1)));
            const Real csig = a * rad_speed + (1. - a) * asym_speed;

            lmin_dt =
                std::min(lmin_dt, 1.0 / (csig / coords.CellWidthFA(X1DIR + d, k, j, i)));
          }
        }
      },
      Kokkos::Min<Real>(min_dt));
  const auto &cfl = pars.Get<Real>("cfl");
  return cfl * min_dt;
}

} // namespace radiation
