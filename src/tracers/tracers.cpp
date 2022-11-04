// preamble.

#include "tracers.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("tracers");
  const bool active = pin->GetBoolean("physics", "tracers");
  physics->AddParam<bool>("active", active);
  if (!active) return physics;

  // input var???
  int num_tracers = pin->GetOrAddReal("Tracers", "num_tracers", 100);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("Tracers", "rng_seed", 1273);
  physics->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  physics->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  physics->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  physics->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  return physics;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  auto swarm = pmb->swarm_data.Get()->Get("tracers");

  const int max_active_index = swarm->GetMaxActiveIndex();

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  
  auto swarm_d = swarm->GetDeviceContext();
  // update loop.

  return TaskStatus::complete;
} // AdvectTracers
} // namespace tracers
