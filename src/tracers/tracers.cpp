// preamble.

#include "phoebus_utils/phoebus_interpolation.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"
#include "phoebus_utils/relativity_utils.hpp"
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
  using namespace LCInterp;
  auto pmb = rc->GetParentPointer();
  auto swarm = pmb->swarm_data.Get()->Get("tracers");

  const int max_active_index = swarm->GetMaxActiveIndex();

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  
  auto swarm_d = swarm->GetDeviceContext();
  namespace p = fluid_prim;

  const std::vector<std::string> vars = {p::velocity};

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;

  auto geom = Geometry::GetCoordinateSystem(rc);

  // update loop.
  pmb->par_for( 
      "Advect Tracers", 0, max_active_index, KOKKOS_LAMBDA( const int n ) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // geom quantities
          Real gcov4[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
          Real shift[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);
          const Real vel[] = {pack(pvel_lo, k, j, i), pack(pvel_lo + 1, k, j, i),
                              pack(pvel_hi, k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vel, gcov4);

          // Get shift, W, lapse
          Real vel_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo) / W; 
          Real vel_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo + 1) / W; 
          Real vel_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_hi) / W; 

          x(n) += lapse * vel_X1 - shift[0];
          y(n) += lapse * vel_X2 - shift[1];
          z(n) += lapse * vel_X3 - shift[2];
       } 
      });  

  return TaskStatus::complete;
} // AdvectTracers
} // namespace tracers
