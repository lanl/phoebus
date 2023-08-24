//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

// TODO(JCD): this should be exported by parthenon
#include <amr_criteria/refinement_package.hpp>
#include <globals.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <prolong_restrict/prolong_restrict.hpp>
#include <utils/error_checking.hpp>

// Local Includes
#include "compile_constants.hpp"
#include "fixup/fixup.hpp"
#include "fluid/fluid.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "microphysics/opac_phoebus/opac_phoebus.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"
#include "phoebus_driver.hpp"
#include "phoebus_package.hpp"
#include "phoebus_utils/robust.hpp"
#include "progenitor/progenitordata.hpp"
#include "radiation/radiation.hpp"
#include "tov/tov.hpp"
#include "tracers/tracers.hpp"

using namespace parthenon::driver::prelude;
using parthenon::AllReduce;

namespace phoebus {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
PhoebusDriver::PhoebusDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm,
                             const bool is_restart)
    : EvolutionDriver(pin, app_in, pm),
      integrator(std::make_unique<LowStorageIntegrator>(pin)), is_restart_(is_restart) {

  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");

  dt_init = pin->GetOrAddReal("parthenon/time", "dt_init", 1.e300);
  dt_init_fact = pin->GetOrAddReal("parthenon/time", "dt_init_fact", 1.0);
}

TaskListStatus PhoebusDriver::Step() {
  static bool first_call = true;
  TaskListStatus status;
  Real dt_trial = tm.dt;
  if (first_call) dt_trial = std::min(dt_trial, dt_init);
  if (!is_restart_) {
    dt_trial *= dt_init_fact;
    dt_init_fact = 1.0;
  }
  if (tm.time + dt_trial > tm.tlim) dt_trial = tm.tlim - tm.time;
  tm.dt = dt_trial;
  integrator->dt = dt_trial;

  for (int stage = 1; stage <= integrator->nstages; stage++) {
    TaskCollection tc = RungeKuttaStage(stage);
    status = tc.Execute();
    if (status != TaskListStatus::complete) break;
  }

  status = RadiationPostStep();

  return status;
}

// TODO(BRR) This is required for periodic BCs, unless the issue is radiation not being
// included in ConvertBoundaryConditions
void PhoebusDriver::PostInitializationCommunication() {
  auto phoebus_package = pmesh->packages.Get("phoebus");
  auto do_post_init_comms = phoebus_package->Param<bool>("do_post_init_comms");
  if (!do_post_init_comms) return;

  TaskCollection tc;
  TaskID none(0);
  BlockList_t &blocks = pmesh->block_list;

  auto rad = pmesh->packages.Get("radiation");
  auto fluid = pmesh->packages.Get("fluid");
  const bool rad_active = rad->Param<bool>("active");
  const bool fluid_active = fluid->Param<bool>("active");

  const int num_partitions = pmesh->DefaultNumPartitions();

  TaskRegion &async_region_1 = tc.AddRegion(blocks.size());
  for (int ib = 0; ib < blocks.size(); ib++) {
    auto pmb = blocks[ib].get();
    auto &tl = async_region_1[ib];
    auto &sc = pmb->meshblock_data.Get();
    auto apply_floors =
        tl.AddTask(none, fixup::ApplyFloors<MeshBlockData<Real>>, sc.get());
  }

  TaskRegion &sync_region_1 = tc.AddRegion(num_partitions);
  for (int ib = 0; ib < num_partitions; ib++) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", ib);
    auto &tl = sync_region_1[ib];

    const auto any = parthenon::BoundaryType::any;
    const auto local = parthenon::BoundaryType::local;
    const auto nonlocal = parthenon::BoundaryType::nonlocal;

    auto boundary_tasks =
        parthenon::AddBoundaryExchangeTasks(none, tl, md, pmesh->multilevel);
  }

  TaskRegion &async_region_2 = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_2[i];
    auto &sc = pmb->meshblock_data.Get();

    auto set_bc = none;
    // called in boundary exchange tasks
    // auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc);

    auto convert_bc = tl.AddTask(set_bc, Boundaries::ConvertBoundaryConditions, sc);

    auto fill_derived = tl.AddTask(
        convert_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc.get());
  }

  tc.Execute();
}

TaskCollection PhoebusDriver::RungeKuttaStage(const int stage) {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  BlockList_t &blocks = pmesh->block_list;

  const Real beta = integrator->beta[stage - 1];
  const Real t = tm.time;
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  auto rad = pmesh->packages.Get("radiation");
  auto fluid = pmesh->packages.Get("fluid");
  auto tracers = pmesh->packages.Get("tracers");
  auto monopole = pmesh->packages.Get("monopole_gr");
  const auto rad_active = rad->Param<bool>("active");
  const auto rad_moments_active = rad->Param<bool>("moments_active");
  const auto fluid_active = fluid->Param<bool>("active");
  const auto tracers_active = tracers->Param<bool>("active");
  bool rad_mocmc_active = false;
  if (rad_active) {
    rad_mocmc_active = (rad->Param<std::string>("method") == "mocmc");
  }
  const auto monopole_enabled = monopole->Param<bool>("enable_monopole_gr");
  // Force static here means monopole only called at initialization.
  // and source terms are disabled
  const auto monopole_force_static =
      (monopole_enabled && monopole->Param<bool>("force_static"));
  // nth call means only run monopole solver the first run_n_times
  // subcycles. Then stop.
  int monopole_nth_call = 0;
  int monopole_run_n_times = 0;
  if (monopole_enabled) {
    monopole_run_n_times = monopole->Param<int>("run_n_times");
    monopole_nth_call = monopole->Param<int>("nth_call");
  }
  const auto monopole_gr_active =
      (monopole_enabled &&
       ((monopole_run_n_times < 0) || (monopole_nth_call <= monopole_run_n_times)) &&
       !monopole_force_static);
  if (monopole_gr_active) {
    monopole->AllParams().Update("nth_call", monopole_nth_call + 1);
  }

  std::vector<std::string> src_names;
  std::vector<std::string> src_w_diag;
  if (fluid_active) {
    src_names.push_back(fluid_cons::momentum);
    src_names.push_back(fluid_cons::energy);
  }
  if (rad_moments_active) {
    src_names.push_back(radmoment_cons::E);
    src_names.push_back(radmoment_cons::F);
  }
  src_w_diag = src_names;
#if SET_FLUX_SRC_DIAGS
  if (fluid_active) src_w_diag.push_back(diagnostic_variables::src_terms);
#endif

  using MonoMatRed_t = AllReduce<MonopoleGR::Matter_host_t>;
  using MonoVolRed_t = AllReduce<MonopoleGR::Volumes_host_t>;
  MonoMatRed_t *pmono_mat_red;
  MonoVolRed_t *pmono_vol_red;
  if (monopole_gr_active) {
    pmono_mat_red = monopole->MutableParam<MonoMatRed_t>("matter_reducer");
    pmono_vol_red = monopole->MutableParam<MonoVolRed_t>("volumes_reducer");
  }

  // Hack to update time for time-dependent geometries
  auto geometry = pmesh->packages.Get("geometry");
  auto &geom_params = geometry->AllParams();
  // tm = SimTime field in EvolutionDriver. See parthenon/src/driver/driver.hpp
  if (geom_params.hasKey("time")) {
    const Real tstage = tm.time + (stage > 1 ? (integrator->beta[stage - 2]) : 0) * tm.dt;
    geom_params.Update("time", tstage);
  }
  bool time_dependent_geom = geom_params.Get("time_dependent", false);

  const int num_partitions = pmesh->DefaultNumPartitions();

  const auto any = parthenon::BoundaryType::any;
  const auto local = parthenon::BoundaryType::local;
  const auto nonlocal = parthenon::BoundaryType::nonlocal;

  if (stage == 1) {
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      pmb->meshblock_data.Add("geometric source terms", base, src_w_diag);
      for (int s = 1; s < integrator->nstages; s++) {
        pmb->meshblock_data.Add(stage_name[s], base);
      }
    }
  }

  const auto num_independent_task_lists = blocks.size();

  // be ready for flux corrections and boundary comms
  TaskRegion &sync_region_1 = tc.AddRegion(num_partitions);
  for (int ib = 0; ib < num_partitions; ++ib) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", ib);
    auto &sc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], ib);
    auto &sc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], ib);
    auto &tl = sync_region_1[ib];

    tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, sc1);
    if (pmesh->multilevel) {
      tl.AddTask(none, parthenon::StartReceiveFluxCorrections, sc0);
    }
  }

  TaskRegion &async_region_1 = tc.AddRegion(num_independent_task_lists);
  for (int ib = 0; ib < num_independent_task_lists; ib++) {
    auto pmb = blocks[ib].get();
    auto &tl = async_region_1[ib];

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    // pull out a container for the geometric source terms
    auto &gsrc = pmb->meshblock_data.Get("geometric source terms");

    TaskID geom_src(0);
    TaskID sndrcv_flux_depend(0);

    if (fluid_active) {
      auto hydro_flux = tl.AddTask(none, fluid::CalculateFluxes, sc0.get());
      auto hydro_flux_ct = tl.AddTask(hydro_flux, fluid::FluxCT, sc0.get());
      auto hydro_geom_src =
          tl.AddTask(none, fluid::CalculateFluidSourceTerms, sc0.get(), gsrc.get());
      sndrcv_flux_depend = sndrcv_flux_depend | hydro_flux_ct;
      geom_src = geom_src | hydro_geom_src;
    }

    if (rad_moments_active) {
      using MDT = std::remove_pointer<decltype(sc0.get())>::type;
      auto moment_recon =
          tl.AddTask(none, radiation::ReconstructEdgeStates<MDT>, sc0.get());
      // TODO(BRR) Remove from task list if not using due to MOCMC
      auto get_opacities =
          tl.AddTask(moment_recon, radiation::MomentCalculateOpacities<MDT>, sc0.get());
      auto moment_flux =
          tl.AddTask(get_opacities, radiation::CalculateFluxes<MDT>, sc0.get());
      auto moment_geom_src = tl.AddTask(none, radiation::CalculateGeometricSource<MDT>,
                                        sc0.get(), gsrc.get());
      sndrcv_flux_depend = sndrcv_flux_depend | moment_flux;
      geom_src = geom_src | moment_geom_src;
    }

    if (rad_moments_active || fluid_active) {
      auto fix_flux = tl.AddTask(sndrcv_flux_depend, fixup::FixFluxes, sc0.get());
      sndrcv_flux_depend = sndrcv_flux_depend | fix_flux;
    }

    if (rad_mocmc_active) {
      using MDT = std::remove_pointer<decltype(sc0.get())>::type;
      // TODO(BRR) stage_name[stage - 1]?
      auto &sd0 = pmb->swarm_data.Get(stage_name[integrator->nstages]);
      auto samples_transport =
          tl.AddTask(none, radiation::MOCMCTransport<MDT>, sc0.get(), dt);
      auto send = tl.AddTask(samples_transport, &SwarmContainer::Send, sd0.get(),
                             BoundaryCommSubset::all);
      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sd0.get(), BoundaryCommSubset::all);
      auto sample_bounds =
          tl.AddTask(receive, radiation::MOCMCSampleBoundaries<MDT>, sc0.get());
      auto sample_recon =
          tl.AddTask(sample_bounds, radiation::MOCMCReconstruction<MDT>, sc0.get());
      auto eddington =
          tl.AddTask(sample_recon, radiation::MOCMCEddington<MDT>, sc0.get());

      geom_src = geom_src | eddington;
    }

    // BLB: is this right?
    if (tracers_active) {
      auto &sd0 = pmb->swarm_data.Get(stage_name[integrator->nstages]);
      auto advect_tracers = tl.AddTask(none, tracers::AdvectTracers, sc0.get(), dt);
      // BLB: Just copying above.. needs a good check.
      auto send = tl.AddTask(advect_tracers, &SwarmContainer::Send, sd0.get(),
                             BoundaryCommSubset::all);
      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sd0.get(), BoundaryCommSubset::all);
    }
  }

  // Extra per-step user work
  TaskRegion &sync_region_5 = tc.AddRegion(num_partitions);
  int sync_region_5_dep_id;
  net_field_totals.val.resize(2); // Mdot, Phi
  net_field_totals_2.val.resize(2);
  for (int i = 0; i < 2; i++) {
    net_field_totals.val[i] = 0.;
    net_field_totals_2.val[i] = 0.;
  }
  for (int i = 0; i < num_partitions; i++) {
    sync_region_5_dep_id = 0;
    auto &tl = sync_region_5[i];

    auto &md = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);

    // Begin tuning region that only evaluates occasionally and on first stage
    // TODO(BRR) Setting scale factor of B field need only be done once per simulation

    // Evaluate current Mdot, Phi
    auto sum_mdot_1 = tl.AddTask(none, fixup::SumMdotPhiForNetFieldScaling, md.get(), t,
                                 stage, &(net_field_totals.val));

    sync_region_5.AddRegionalDependencies(sync_region_5_dep_id, i, sum_mdot_1);
    sync_region_5_dep_id++;

    TaskID start_reduce_1 = (i == 0 ? tl.AddTask(sum_mdot_1, fixup::NetFieldStartReduce,
                                                 md.get(), t, stage, &net_field_totals)
                                    : none);
    // Test the reduction until it completes
    TaskID finish_reduce_1 = tl.AddTask(start_reduce_1, fixup::NetFieldCheckReduce,
                                        md.get(), t, stage, &net_field_totals);
    sync_region_5.AddRegionalDependencies(sync_region_5_dep_id, i, finish_reduce_1);
    sync_region_5_dep_id++;

    auto mod_net = tl.AddTask(finish_reduce_1, fixup::ModifyNetField, md.get(), t,
                              beta * dt, stage, true, 1.);

    // Evaluate Mdot, Phi (only Phi changes) after modifying B field
    auto sum_mdot_2 = tl.AddTask(mod_net, fixup::SumMdotPhiForNetFieldScaling, md.get(),
                                 t, stage, &(net_field_totals_2.val));
    sync_region_5.AddRegionalDependencies(sync_region_5_dep_id, i, sum_mdot_2);
    sync_region_5_dep_id++;

    TaskID start_reduce_2 = (i == 0 ? tl.AddTask(sum_mdot_2, fixup::NetFieldStartReduce,
                                                 md.get(), t, stage, &net_field_totals_2)
                                    : none);
    // Test the reduction until it completes
    TaskID finish_reduce_2 = tl.AddTask(start_reduce_2, fixup::NetFieldCheckReduce,
                                        md.get(), t, stage, &net_field_totals_2);
    sync_region_5.AddRegionalDependencies(sync_region_5_dep_id, i, finish_reduce_2);
    sync_region_5_dep_id++;

    // Remove artificial contribution to net field
    auto mod_net_2 = tl.AddTask(finish_reduce_2, fixup::ModifyNetField, md.get(), t,
                                beta * dt, stage, true, -1.);

    auto set_scale =
        tl.AddTask(mod_net_2, fixup::UpdateNetFieldScaleControls, md.get(), t, dt, stage,
                   &(net_field_totals.val), &(net_field_totals_2.val));

    // End tuning region that only evaluates occasionally and on first stage

    // Update net field every stage of every timestep given current tuning parameters
    auto update_netfield = tl.AddTask(set_scale, fixup::ModifyNetField, md.get(), t,
                                      beta * dt, stage, false, 0.0);
  }
  // This is a bad pattern. Having a per-mesh data p2c would help.
  TaskRegion &async_region_4 = tc.AddRegion(num_independent_task_lists);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_4[i];
    auto &sc = pmb->meshblock_data.Get(stage_name[stage]);

    StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
    const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

    if (enable_phi_enforcement) {
      auto p2c = tl.AddTask(none, fluid::PrimitiveToConserved, sc.get());
    }
  }

  TaskRegion &sync_region_2 = tc.AddRegion(num_partitions);
  for (int ib = 0; ib < num_partitions; ib++) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", ib);
    auto &sc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], ib);
    auto &sc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], ib);
    auto &dudt = pmesh->mesh_data.GetOrAdd("dUdt", ib);
    auto &tl = sync_region_2[ib];
    auto &gsrc = pmesh->mesh_data.GetOrAdd("geometric source terms", ib);
    int reg_dep_id = 0;

    using MDT = std::remove_pointer<decltype(sc0.get())>::type;

    // TODO(JMM): Not sure what the optimal inter-weaving is from an
    // accuracy point of view. The way I have it now, the metric
    // updates before con2prim and boundaries, which is I think where
    // it's first needed.
    TaskID interp_to_monopole = none;
    if (monopole_gr_active) {
      auto interp_to_monopole =
          tl.AddTask(none, MonopoleGR::InterpolateMatterTo1D<MDT>, sc0.get());
      sync_region_2.AddRegionalDependencies(reg_dep_id++, ib, interp_to_monopole);

      // TODO(JMM): Is this the right place for this?
      // TODO(JMM): Should this stuff be in the synchronous region?
      auto matter_to_host =
          (ib == 0 ? tl.AddTask(interp_to_monopole, MonopoleGR::MatterToHost,
                                monopole.get(), true)
                   : none);
      sync_region_2.AddRegionalDependencies(reg_dep_id++, ib, matter_to_host);
      auto start_reduce_matter =
          (ib == 0 ? tl.AddTask(matter_to_host, &MonoMatRed_t::StartReduce, pmono_mat_red,
                                MPI_SUM)
                   : none);
      auto start_reduce_vols =
          (ib == 0 ? tl.AddTask(matter_to_host, &MonoVolRed_t::StartReduce, pmono_vol_red,
                                MPI_SUM)
                   : none);
      auto finish_reduce_matter =
          tl.AddTask(start_reduce_matter, &MonoMatRed_t::CheckReduce, pmono_mat_red);
      sync_region_2.AddRegionalDependencies(reg_dep_id++, ib, finish_reduce_matter);
      auto finish_reduce_vols =
          tl.AddTask(start_reduce_vols, &MonoVolRed_t::CheckReduce, pmono_vol_red);
      sync_region_2.AddRegionalDependencies(reg_dep_id++, ib, finish_reduce_vols);
      auto finish_mono_reds = finish_reduce_matter | finish_reduce_vols;
      auto divide_vols =
          (ib == 0 ? tl.AddTask(finish_mono_reds, MonopoleGR::DivideVols, monopole.get())
                   : none);
      auto integrate_hypersurface =
          (ib == 0 ? tl.AddTask(divide_vols, MonopoleGR::IntegrateHypersurface,
                                monopole.get())
                   : none);
      auto lin_solve_for_lapse =
          (ib == 0 ? tl.AddTask(integrate_hypersurface, MonopoleGR::LinearSolveForAlpha,
                                monopole.get())
                   : none);
      auto spacetime_to_device =
          (ib == 0 ? tl.AddTask(lin_solve_for_lapse, MonopoleGR::SpacetimeToDevice,
                                monopole.get())
                   : none);
      auto check_monopole_dt =
          (ib == 0 ? tl.AddTask(spacetime_to_device, MonopoleGR::CheckRateOfChange,
                                monopole.get(), tm.dt / 2)
                   : none);
    }
  }

  // Communicate flux corrections and update independent data with fluxes and geometric
  // sources
  TaskRegion &sync_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = sync_region_3[i];
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);
    auto &mgsrc = pmesh->mesh_data.GetOrAdd("geometric source terms", i);

    auto send_flux = tl.AddTask(none, parthenon::LoadAndSendFluxCorrections, mc0);
    auto recv_flux = tl.AddTask(none, parthenon::ReceiveFluxCorrections, mc0);
    auto set_flux = tl.AddTask(recv_flux, parthenon::SetFluxCorrections, mc0);

    auto flux_div =
        tl.AddTask(set_flux, parthenon::Update::FluxDivergence<MeshData<Real>>, mc0.get(),
                   mdudt.get());

#if SET_FLUX_SRC_DIAGS
    auto copy_flux_div = tl.AddTask(
        flux_div /*| geom_src*/, fluid::CopyFluxDivergence<MeshData<Real>>, mdudt.get());
#endif

    auto add_rhs = tl.AddTask(flux_div, SumData<std::vector<std::string>, MeshData<Real>>,
                              src_names, mdudt.get(), mgsrc.get(), mdudt.get());

    auto avg_data = tl.AddTask(flux_div, AverageIndependentData<MeshData<Real>>,
                               mc0.get(), mbase.get(), beta);

    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshData<Real>>, mc0.get(),
                             mdudt.get(), beta * dt, mc1.get());
  }

  // Fix up flux failures
  TaskRegion &async_region_2 = tc.AddRegion(num_independent_task_lists);
  for (int ib = 0; ib < num_independent_task_lists; ib++) {
    auto pmb = blocks[ib].get();
    auto &tl = async_region_2[ib];

    // first make other useful containers
    auto &base = pmb->meshblock_data.Get();

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    using MDT = std::remove_pointer<decltype(sc1.get())>::type;

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());

    auto fixup = tl.AddTask(
        fill_derived, fixup::ConservedToPrimitiveFixup<MeshBlockData<Real>>, sc1.get());

    auto radfixup = tl.AddTask(
        fixup, fixup::RadConservedToPrimitiveFixup<MeshBlockData<Real>>, sc1.get());

    auto floors =
        tl.AddTask(radfixup, fixup::ApplyFloors<MeshBlockData<Real>>, sc1.get());

    TaskID gas_rad_int(0);
    if (rad_mocmc_active) {
      auto impl_update =
          tl.AddTask(floors, radiation::MOCMCFluidSource<MeshBlockData<Real>>, sc1.get(),
                     beta * dt, fluid_active);
      auto impl_edd = tl.AddTask(
          impl_update, radiation::MOCMCEddington<MeshBlockData<Real>>, sc1.get());
      gas_rad_int = gas_rad_int | impl_edd;
    } else if (rad_moments_active) {
      auto impl_update =
          tl.AddTask(floors, radiation::MomentFluidSource<MeshBlockData<Real>>, sc1.get(),
                     beta * dt, fluid_active);
      gas_rad_int = gas_rad_int | impl_update;
    }

    if (rad_moments_active) {
      // Only apply floors because MomentFluidSource already ensured that a sensible state
      // was returned
      auto floors =
          tl.AddTask(gas_rad_int, fixup::ApplyFloors<MeshBlockData<Real>>, sc1.get());
    }
  }

  // Communicate (after applying stencil-based fixup)
  TaskRegion &sync_region_4 = tc.AddRegion(num_partitions);
  for (int ip = 0; ip < num_partitions; ip++) {
    auto &tl = sync_region_4[ip];
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], ip);

    auto boundary_tasks =
        parthenon::AddBoundaryExchangeTasks(none, tl, mc1, pmesh->multilevel);
  }
  // TODO(BRR) loop this in thoughtfully!
  TaskRegion &async_region_3 = tc.AddRegion(num_independent_task_lists);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_3[i];
    auto &sc = pmb->meshblock_data.Get(stage_name[stage]);

    // auto prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, sc);

    auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc);

    auto convert_bc = tl.AddTask(set_bc, Boundaries::ConvertBoundaryConditions, sc);

    auto fill_derived = tl.AddTask(
        convert_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc.get());

    auto fixup = tl.AddTask(
        fill_derived, fixup::ConservedToPrimitiveFixup<MeshBlockData<Real>>, sc.get());

    auto radfixup = tl.AddTask(
        fixup, fixup::RadConservedToPrimitiveFixup<MeshBlockData<Real>>, sc.get());

    auto floors = tl.AddTask(radfixup, fixup::ApplyFloors<MeshBlockData<Real>>, sc.get());

    if (rad_mocmc_active && stage == integrator->nstages) {
      particle_resolution.val.resize(1); // total
      // for (int i = 0; i < particle_resolution.val.size(); i++) {
      particle_resolution.val[i] = 0.;
      //}
      int reg_dep_id = 0;
      // auto &tl = async_region_3[0];

      auto update_count = tl.AddTask(none, radiation::MOCMCUpdateParticleCount, pmesh,
                                     &particle_resolution.val);

      async_region_3.AddRegionalDependencies(reg_dep_id, 0, update_count);
      reg_dep_id++;

      auto start_count_reduce =
          tl.AddTask(update_count, &AllReduce<std::vector<Real>>::StartReduce,
                     &particle_resolution, MPI_SUM);

      auto finish_count_reduce =
          tl.AddTask(start_count_reduce, &AllReduce<std::vector<Real>>::CheckReduce,
                     &particle_resolution);
      async_region_3.AddRegionalDependencies(reg_dep_id, 0, finish_count_reduce);
      reg_dep_id++;

      // Report particle count
      auto report_count =
          (Globals::my_rank == 0 ? tl.AddTask(
                                       finish_count_reduce,
                                       [](std::vector<Real> *res) {
                                         std::cout << "MOCMC total particles = "
                                                   << static_cast<long>((*res)[0])
                                                   << std::endl;
                                         return TaskStatus::complete;
                                       },
                                       &particle_resolution.val)
                                 : none);
    }

    if (time_dependent_geom) {
      auto update_geom =
          tl.AddTask(floors, Geometry::UpdateGeometry<MeshBlockData<Real>>, sc.get());
    }

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          floors, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc.get());

      if (fluid_active) {
        auto divb = tl.AddTask(floors, fluid::CalculateDivB, sc.get());
      }

      // Update refinement
      if (pmesh->adaptive) {
        // using tag_type = TaskStatus(std::shared_ptr<MeshBlockData<Real>> &);
        auto tag_refine =
            tl.AddTask(floors, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc.get());
      }
    }
  }

  return tc;
} // namespace phoebus

TaskStatus DefaultTask() { return TaskStatus::complete; }

TaskListStatus PhoebusDriver::RadiationPostStep() {
  TaskCollection tc;
  TaskID none(0);

  BlockList_t &blocks = pmesh->block_list;

  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  auto rad = pmesh->packages.Get("radiation");
  const auto rad_active = rad->Param<bool>("active");
  if (!rad_active) {
    return TaskListStatus::complete;
  }
  auto fluid = pmesh->packages.Get("fluid");
  const auto fluid_active = fluid->Param<bool>("active");

  auto num_independent_task_lists = blocks.size();

  const auto rad_method = rad->Param<std::string>("method");
  if (rad_method == "cooling_function") {
    parthenon::AllReduce<bool> *pdo_gain_reducer;
    bool do_lightbulb = rad->Param<bool>("do_lightbulb");
    if (do_lightbulb) {
      pdo_gain_reducer = rad->MutableParam<parthenon::AllReduce<bool>>("do_gain_reducer");
    }
    TaskRegion &async_region = tc.AddRegion(num_independent_task_lists);
    for (int ib = 0; ib < num_independent_task_lists; ib++) {
      auto pmb = blocks[ib].get();
      auto &tl = async_region[ib];
      auto &sc0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto finish_gain_reducer = none;
      if (do_lightbulb) {
        auto calc_tau = tl.AddTask(none, radiation::LightBulbCalcTau, sc0.get());
        auto check_do_gain_local = tl.AddTask(calc_tau, radiation::CheckDoGain, sc0.get(),
                                              &(pdo_gain_reducer->val));
        auto start_gain_reducer =
            (ib == 0 ? tl.AddTask(check_do_gain_local,
                                  &parthenon::AllReduce<bool>::StartReduce,
                                  pdo_gain_reducer, MPI_LOR)
                     : none);
        finish_gain_reducer =
            tl.AddTask(start_gain_reducer, &parthenon::AllReduce<bool>::CheckReduce,
                       pdo_gain_reducer);
        int reg_dep_id = 0;
        async_region.AddRegionalDependencies(reg_dep_id++, ib, finish_gain_reducer);
      }

      auto calculate_four_force =
          tl.AddTask(finish_gain_reducer, radiation::CoolingFunctionCalculateFourForce,
                     sc0.get(), dt);
      auto apply_four_force = tl.AddTask(
          calculate_four_force, radiation::ApplyRadiationFourForce, sc0.get(), dt);
    }
  } else if (rad_method == "monte_carlo") {
    return MonteCarloStep();
  }

  return tc.Execute();
}

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;

  packages.Add(phoebus::Initialize(pin.get()));
  packages.Add(Microphysics::EOS::Initialize(pin.get()));
  packages.Add(Microphysics::Opacity::Initialize(pin.get()));
  packages.Add(Geometry::Initialize(pin.get()));
  packages.Add(fluid::Initialize(pin.get()));
  packages.Add(radiation::Initialize(pin.get()));
  packages.Add(fixup::Initialize(pin.get()));
  packages.Add(MonopoleGR::Initialize(pin.get())); // Does nothing if not enabled
  packages.Add(TOV::Initialize(pin.get()));        // Does nothing if not enabled.
  packages.Add(tracers::Initialize(pin.get()));
  packages.Add(Progenitor::Initialize(pin.get()));

  // TODO(JMM): I need to do this before problem generators get
  // called. For now I'm hacking this in here. But in the long term,
  // it may require a shift in how parthenon does things.
  auto tov_pkg = packages.Get("tov");
  auto progenitor_pkg = packages.Get("progenitor");
  auto monopole_pkg = packages.Get("monopole_gr");
  auto eos_pkg = packages.Get("eos");
  const auto enable_tov = tov_pkg->Param<bool>("enabled");
  const auto enable_monopole = monopole_pkg->Param<bool>("enable_monopole_gr");
  const bool is_monopole_cart =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleCart));
  const bool is_monopole_sph =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph));
  if (enable_tov && !enable_monopole) {
    PARTHENON_THROW("MonopoleGR required for TOV initialization");
  }
  const auto enable_progenitor = progenitor_pkg->Param<bool>("enabled");
  if (enable_monopole && !(enable_tov || enable_progenitor)) {
    PARTHENON_THROW("Currently monopole GR only enabled with TOV or Progenitor packages");
  }
  if ((enable_monopole && !(is_monopole_cart || is_monopole_sph)) ||
      (is_monopole_cart || is_monopole_sph) && !enable_monopole) {
    PARTHENON_THROW("MonopoleGR must be coupled with monopole metric");
  }
  if (enable_tov) {
    TOV::IntegrateTov(tov_pkg.get(), monopole_pkg.get(), eos_pkg.get());
  }
  if (enable_monopole) {
    MonopoleGR::MatterToHost(monopole_pkg.get(), false);
    MonopoleGR::IntegrateHypersurface(monopole_pkg.get());
    MonopoleGR::LinearSolveForAlpha(monopole_pkg.get());
    MonopoleGR::SpacetimeToDevice(monopole_pkg.get());
    if (parthenon::Globals::my_rank == 0) {
      MonopoleGR::DumpToTxt("tov.dat", monopole_pkg.get());
    }
  }
  return packages;
}

TaskListStatus PhoebusDriver::MonteCarloStep() {
  TaskCollection tc;
  TaskListStatus status; // tl;
  const double t0 = tm.time;
  const double dt = tm.dt;
  TaskID none(0);
  const auto &stage_name = integrator->stage_name;

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();

  // Create all particles sourced due to emission during timestep
  {
    TaskCollection tc;
    // TODO(BRR) no longer necessary with iterative task right
    TaskRegion &sync_region0 = tc.AddRegion(1);
    {
      auto &tl = sync_region0[0];
      auto initialize_comms =
          tl.AddTask(none, radiation::InitializeCommunicationMesh, "monte_carlo", blocks);
    }

    //// BEGINNING OF NEW SOLVER REGION
    // TODO(BRR) Currently this doesn't work properly

    /*TaskRegion &sample_region = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      auto &tl = sample_region[i];
      auto &mbd0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto &sc0 = pmb->swarm_data.Get(stage_name[integrator->nstages]);
      auto sample_particles = tl.AddTask(none, radiation::MonteCarloSourceParticles,
                                         pmb.get(), mbd0.get(), sc0.get(), t0, dt);
    }

    int max_iters = 5;
    int check_interval = 1;
    bool fail_flag = true;
    bool warn_flag = true;

    // Iterate over particle send/receives until all particles are finished
    TaskRegion &transport_region = tc.AddRegion(num_task_lists_executed_independently);
    particles_outstanding.val = 0;
    for (int i = 0; i < blocks.size(); i++) {
      printf("%i blocks;size: %i\n", i, blocks.size());

      // for (int i = 0; i < num_task_lists_executed_independently; i++) {
      int reg_dep_id = 0;

      auto &pmb = blocks[i];
      auto &tl = transport_region[i];
      auto &mbd0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto &sc0 = pmb->swarm_data.Get(stage_name[integrator->nstages]);

      auto &solver = tl.AddIteration("particle transport");
      solver.SetMaxIterations(max_iters);
      solver.SetCheckInterval(check_interval);
      solver.SetFailWithMaxIterations(fail_flag);
      solver.SetWarnWithMaxIterations(warn_flag);
      auto transport_particles = solver.AddTask(none, radiation::MonteCarloTransport,
                                                pmb.get(), mbd0.get(), sc0.get(), t0, dt);

      auto send_particles = solver.AddTask(transport_particles, &SwarmContainer::Send,
                                           sc0.get(), BoundaryCommSubset::all);

      auto receive_particles = solver.AddTask(send_particles, &SwarmContainer::Receive,
                                              sc0.get(), BoundaryCommSubset::all);

      // TODO(BRR) This should be a task in parthenon
      // TODO(BRR) to do this I need a separate STATUS variable for each particle to say
      // whether it is being absorbed, being transported, being scattered, done, etc.
      auto add_sent_particles = solver.AddTask(
          receive_particles, radiation::MonteCarloCountCommunicatedParticles, pmb.get(),
          &particles_outstanding.val);

      //        auto count_outstanding_particles = solver.AddTask(receive_particles,
      //        radiation::MonteCarloCountOutstandingParticles, pmb.get(), mbd0.get(),
      //        sc0.get(), &particles_outstanding.val);

      transport_region.AddRegionalDependencies(reg_dep_id, i, add_sent_particles);
      reg_dep_id++;

      auto start_global_reduce =
          (i == 0 ? tl.AddTask(add_sent_particles, &AllReduce<int>::StartReduce,
                               &particles_outstanding, MPI_SUM)
                  : none);

      auto finish_global_reduce = tl.AddTask(
          start_global_reduce, &AllReduce<int>::CheckReduce, &particles_outstanding);

      // Ensure zero particles transported to end transport cycle
      auto check = solver.SetCompletionTask(
          finish_global_reduce,
          [](int *num_transported) {
            printf("[%i] check (%i)\n\n\n\n", Globals::my_rank, *num_transported);
            if (*num_transported == 0) {
              return TaskStatus::complete;
            } else {
              return TaskStatus::iterate;
            }
          },
          // radiation::CheckNoOutstandingParticles,
          &particles_outstanding.val);

      transport_region.AddRegionalDependencies(reg_dep_id, i, check);
      reg_dep_id++;
    }*/

    //// END OF NEW SOLVER REGION

    // TODO(BRR) make transport an async region for ghost cells
    TaskRegion &async_region0 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      auto &tl = async_region0[i];
      auto &mbd0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto &sc0 = pmb->swarm_data.Get(stage_name[integrator->nstages]);
      auto sample_particles = tl.AddTask(none, radiation::MonteCarloSourceParticles,
                                         pmb.get(), mbd0.get(), sc0.get(), t0, dt);
      auto transport_particles =
          tl.AddTask(sample_particles, radiation::MonteCarloTransport, pmb.get(),
                     mbd0.get(), sc0.get(), dt);

      auto send = tl.AddTask(transport_particles, &SwarmContainer::Send, sc0.get(),
                             BoundaryCommSubset::all);

      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sc0.get(), BoundaryCommSubset::all);
    }

    /**
     * NOTE: this task region is size 1
     * In the resolution controls we loop over meshblocks
     * and call MPI reduce.
     * Probably more performant to change this, but will
     * require restructuring some resolution controls.
     **/
    TaskRegion &tuning_region = tc.AddRegion(1);
    {
      particle_resolution.val.resize(4); // made, absorbed, scattered, total
      for (int i = 0; i < 4; i++) {
        particle_resolution.val[i] = 0.;
      }
      int reg_dep_id = 0;
      auto &tl = tuning_region[0];

      auto update_resolution =
          tl.AddTask(none, radiation::MonteCarloUpdateParticleResolution, pmesh,
                     &particle_resolution.val);

      tuning_region.AddRegionalDependencies(reg_dep_id, 0, update_resolution);
      reg_dep_id++;

      auto start_resolution_reduce =
          tl.AddTask(update_resolution, &AllReduce<std::vector<Real>>::StartReduce,
                     &particle_resolution, MPI_SUM);

      auto finish_resolution_reduce =
          tl.AddTask(start_resolution_reduce, &AllReduce<std::vector<Real>>::CheckReduce,
                     &particle_resolution);
      tuning_region.AddRegionalDependencies(reg_dep_id, 0, finish_resolution_reduce);
      reg_dep_id++;

      // Report tuning
      auto report_resolution =
          (Globals::my_rank == 0 || 1
               ? tl.AddTask(
                     finish_resolution_reduce,
                     [](std::vector<Real> *res) {
                       std::cout << "particles made = " << (*res)[0]
                                 << " abs = " << (*res)[1] << " scatt = " << (*res)[2]
                                 << " total = " << (*res)[3] << std::endl;
                       return TaskStatus::complete;
                     },
                     &particle_resolution.val)
               : none);

      auto update_tuning =
          tl.AddTask(finish_resolution_reduce, radiation::MonteCarloUpdateTuning, pmesh,
                     &particle_resolution.val, t0, dt);
    }

    status = tc.Execute();
  }

  // Finalization calls
  {
    TaskCollection tc;
    TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);
    for (int ib = 0; ib < num_task_lists_executed_independently; ib++) {
      auto pmb = blocks[ib].get();
      auto &tl = async_region1[ib];
      auto &sc0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto apply_four_force =
          tl.AddTask(none, radiation::ApplyRadiationFourForce, sc0.get(), dt);
    }
    status = tc.Execute();
  }

  return status;
}

} // namespace phoebus
