// preamble.

#include "tracers.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("tracers");

  // input var???
  int num_tracers = pin->GetOrAddReal("Tracers", "num_tracers", 100);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("Tracers", "rng_seed", 1273);
  pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  return pkg;
} // Initialize

} // namespace tracers
