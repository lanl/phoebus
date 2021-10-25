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
#include <vector>

//TODO(JCD): this should be exported by parthenon
#include <refinement/refinement.hpp>

// Local Includes
#include "compile_constants.hpp"
#include "fixup/fixup.hpp"
#include "fluid/fluid.hpp"
#include "geometry/geometry.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "microphysics/opac_phoebus/opac_phoebus.hpp"
#include "phoebus_driver.hpp"
#include "phoebus_utils/debug_utils.hpp"
#include "radiation/radiation.hpp"
#include "tov/tov.hpp"
#include "phoebus_boundaries/phoebus_boundaries.hpp"

using namespace parthenon::driver::prelude;

namespace phoebus {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
PhoebusDriver::PhoebusDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : EvolutionDriver(pin, app_in, pm),
      integrator(std::make_unique<StagedIntegrator>(pin)) {

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
  dt_trial *= dt_init_fact;
  dt_init_fact = 1.0;
  if (tm.time + dt_trial > tm.tlim) dt_trial = tm.tlim - tm.time;
  tm.dt = dt_trial;
  integrator->dt = dt_trial;

  for (int stage = 1; stage <= integrator->nstages; stage++) {
    TaskCollection tc = RungeKuttaStage(stage);
    status = tc.Execute();
    if (status != TaskListStatus::complete) break;
  }

  status = RadiationStep();

  return status;
}

TaskCollection PhoebusDriver::RungeKuttaStage(const int stage) {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  BlockList_t &blocks = pmesh->block_list;

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  std::vector<std::string> src_names({fluid_cons::momentum, fluid_cons::energy});

  auto num_independent_task_lists = blocks.size();
  TaskRegion &async_region_1 = tc.AddRegion(num_independent_task_lists);
  for (int ib = 0; ib < num_independent_task_lists; ib++) {
    auto pmb = blocks[ib].get();
    auto &tl = async_region_1[ib];

    // first make other useful containers
    auto &base = pmb->meshblock_data.Get();
    if (stage == 1) {
      pmb->meshblock_data.Add("dUdt", base);
      for (int i = 1; i < integrator->nstages; i++) {
        pmb->meshblock_data.Add(stage_name[i], base);
      }
      pmb->meshblock_data.Add("geometric source terms", base, src_names);
    }

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

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving,
                                 sc1.get(), BoundaryCommSubset::all);

    auto hydro_flux = tl.AddTask(none, fluid::CalculateFluxes, sc0.get());
    auto fix_flux = tl.AddTask(hydro_flux, fixup::FixFluxes, sc0.get());
    auto flux_ct = tl.AddTask(hydro_flux, fluid::FluxCT, sc0.get());
    auto geom_src = tl.AddTask(none, fluid::CalculateFluidSourceTerms, sc0.get(), gsrc.get());

    auto send_flux =
        tl.AddTask(flux_ct, &MeshBlockData<Real>::SendFluxCorrection, sc0.get());

    auto recv_flux = tl.AddTask(
        flux_ct, &MeshBlockData<Real>::ReceiveFluxCorrection, sc0.get());

    // compute the divergence of fluxes of conserved variables
    auto flux_div =
        tl.AddTask(recv_flux, parthenon::Update::FluxDivergence<MeshBlockData<Real>>, sc0.get(), dudt.get());

    auto copy_flux_div = tl.AddTask(flux_div|geom_src, fluid::CopyFluxDivergence, dudt.get());

    auto add_rhs = tl.AddTask(flux_div|geom_src, SumData<std::string,MeshBlockData<Real>>,
                              src_names, dudt.get(), gsrc.get(), dudt.get());
    auto zero = tl.AddTask(add_rhs, fluid::ZeroUpdate, dudt.get());

    #if PRINT_RHS
    auto print_rhs = tl.AddTask(add_rhs, Debug::PrintRHS<MeshBlockData<Real>>, dudt.get());
    auto next = print_rhs;
    #else
    auto next = add_rhs;
    #endif
  }

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &sync_region = tc.AddRegion(num_partitions);
  for (int ib = 0; ib < num_partitions; ib++) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", ib);
    auto &sc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], ib);
    auto &sc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], ib);
    auto &dudt = pmesh->mesh_data.GetOrAdd("dUdt", ib);
    auto &tl = sync_region[ib];

    // update step
    auto avg_data = tl.AddTask(none, AverageIndependentData<MeshData<Real>>,
                               sc0.get(), base.get(), beta);
    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshData<Real>>, sc0.get(),
                             dudt.get(), beta*dt, sc1.get());

    // update ghost cells
    auto send = tl.AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, sc1);
    auto recv = tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, sc1);
    auto fill_from_bufs = tl.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, sc1);
  }

  TaskRegion &async_region_2 = tc.AddRegion(num_independent_task_lists);
  for (int ib = 0; ib < num_independent_task_lists; ib++) {
    auto pmb = blocks[ib].get();
    auto &tl = async_region_2[ib];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto clear_comm_flags =
      tl.AddTask(none, &MeshBlockData<Real>::ClearBoundary, sc1.get(),
                   BoundaryCommSubset::all);

    auto prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, sc1);

    // set physical boundaries
    auto set_bc =
      tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, sc1);

    auto convert_bc = tl.AddTask(set_bc, Boundaries::ConvertBoundaryConditions, sc1);

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(convert_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());

    auto fixup = tl.AddTask(fill_derived, fixup::ConservedToPrimitiveFixup<MeshBlockData<Real>>, sc1.get());

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fixup, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());

      auto divb = tl.AddTask(set_bc, fluid::CalculateDivB, sc1.get());

      // Update refinement
      if (pmesh->adaptive) {
        //using tag_type = TaskStatus(std::shared_ptr<MeshBlockData<Real>> &);
        auto tag_refine = tl.AddTask(
            fixup, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
      }
    }
  }

  return tc;
}

TaskStatus DefaultTask() { return TaskStatus::complete; }

TaskListStatus PhoebusDriver::RadiationStep() {
  // TaskListStatus status;
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

  auto num_independent_task_lists = blocks.size();

  const auto rad_method = rad->Param<std::string>("method");
  if (rad_method == "cooling_function") {
    TaskRegion &async_region = tc.AddRegion(num_independent_task_lists);
    for (int ib = 0; ib < num_independent_task_lists; ib++) {
      auto pmb = blocks[ib].get();
      auto &tl = async_region[ib];
      auto &sc0 = pmb->meshblock_data.Get(stage_name[integrator->nstages]);
      auto calculate_four_force =
          tl.AddTask(none, radiation::CoolingFunctionCalculateFourForce, sc0.get(), dt);
      auto apply_four_force = tl.AddTask(
          calculate_four_force, radiation::ApplyRadiationFourForce, sc0.get(), dt);
    }
  } else if (rad_method == "moment") {
    PARTHENON_FAIL("Moment method not implemented!");
  } else if (rad_method == "monte_carlo") {
    return MonteCarloStep();
  } else if (rad_method == "mocmc") {
    PARTHENON_FAIL("MOCMC not implemented!");
  }

  return tc.Execute();
}

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;

  packages.Add(Microphysics::EOS::Initialize(pin.get()));
  packages.Add(Microphysics::Opacity::Initialize(pin.get()));
  packages.Add(Geometry::Initialize(pin.get()));
  packages.Add(fluid::Initialize(pin.get()));
  packages.Add(radiation::Initialize(pin.get()));
  packages.Add(fixup::Initialize(pin.get()));
  packages.Add(MonopoleGR::Initialize(pin.get())); // Does nothing if not enabled
  packages.Add(TOV::Initialize(pin.get())); // Does nothing if not enabled.

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
    TaskRegion &sync_region0 = tc.AddRegion(1);
    {
      auto &tl = sync_region0[0];
      auto initialize_comms =
          tl.AddTask(none, radiation::InitializeCommunicationMesh, "monte_carlo", blocks);
    }

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
                     mbd0.get(), sc0.get(), t0, dt);

      auto send = tl.AddTask(transport_particles, &SwarmContainer::Send, sc0.get(),
                             BoundaryCommSubset::all);

      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sc0.get(), BoundaryCommSubset::all);
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
