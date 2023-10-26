// preamble.

#include "tracers.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/phoebus_interpolation.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

namespace tracers {


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("tracers");
  const bool active = pin->GetBoolean("physics", "tracers");
  physics->AddParam<bool>("active", active);
  // Better way to do this? I leave the tracer swarm defined even if we are not
  // doing tracers as it simplifies the pgen initialization.
  // if (!active) return physics;

  Params &params = physics->AllParams();

  int num_tracers = pin->GetOrAddInteger("tracers", "num_tracers", 0);
  params.Add("num_tracers", num_tracers);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", time(NULL));
  physics->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  physics->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  physics->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  physics->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  // thermo variables
  physics->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("temperature", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("ye", swarm_name, real_swarmvalue_metadata);
  //physics->AddSwarmValue("entropy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("energy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);
  //physics->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
  //physics->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
  //physics->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lorentz", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lapse", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("mass", swarm_name, real_swarmvalue_metadata);

  physics->PostFillDerivedBlock = FillTracers;
  
  return physics;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *rc, const Real dt) {
  using namespace LCInterp;
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("tracers");

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();

  auto swarm_d = swarm->GetDeviceContext();

  const std::vector<std::string> vars = {p::velocity};

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;

  auto geom = Geometry::GetCoordinateSystem(rc);

  // update loop.
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect Tracers", 0, max_active_index - 1, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // geom quantities
          Real gcov4[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, z(n), y(n), x(n), gcov4);
          Real lapse = geom.Lapse(CellLocation::Cent, z(n), y(n), x(n));
          Real shift[3];
          geom.ContravariantShift(CellLocation::Cent, z(n), y(n), x(n), shift);

          // Get shift, W, lapse
          const Real Wvel_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo);
          const Real Wvel_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo + 1);
          const Real Wvel_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_hi);
          const Real vel[] = {Wvel_X1, Wvel_X2, Wvel_X3};
          const Real W = phoebus::GetLorentzFactor(vel, gcov4); // Q: Interp this?
          const Real vel_X1 = Wvel_X1 / W;
          const Real vel_X2 = Wvel_X2 / W;
          const Real vel_X3 = Wvel_X3 / W;

          x(n) += (lapse * vel_X1 - shift[0]) * dt;
          y(n) += (lapse * vel_X2 - shift[1]) * dt;
          z(n) += (lapse * vel_X3 - shift[2]) * dt;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
} // AdvectTracers

/**
 * FillDerived function for tracers.
 * Registered Quantities (in addition to t, x, y, z):
 * rho, T, ye, vel, W_lorentz, pressure, lapse, shift, entropy
 **/
void FillTracers(MeshBlockData<Real> *rc) {
  using namespace LCInterp;
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("tracers");
  auto eos = pmb->packages.Get("eos")->Param<EOS>("d.EOS");
  //auto eos = pmb->packages.Get("eos");

  // pull swarm vars
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &v1 = swarm->Get<Real>("vel_x").Get();
  auto &v2 = swarm->Get<Real>("vel_y").Get();
  auto &v3 = swarm->Get<Real>("vel_z").Get();
  //auto &B1 = swarm->Get<Real>("B_x").Get();
  //auto &B2 = swarm->Get<Real>("B_y").Get();
  //auto &B3 = swarm->Get<Real>("B_z").Get();
  auto &s_rho = swarm->Get<Real>("rho").Get();
  auto &s_temperature = swarm->Get<Real>("temperature").Get();
  auto &s_ye = swarm->Get<Real>("ye").Get();
  //auto &s_entropy = swarm->Get<Real>("entropy").Get();
  auto &s_energy = swarm->Get<Real>("energy").Get();
  auto &s_lorentz = swarm->Get<Real>("lorentz").Get();
  auto &s_lapse = swarm->Get<Real>("lapse").Get();
  auto &shift_x = swarm->Get<Real>("shift_x").Get();
  auto &shift_y = swarm->Get<Real>("shift_y").Get();
  auto &shift_z = swarm->Get<Real>("shift_z").Get();


  auto swarm_d = swarm->GetDeviceContext();

  const std::vector<std::string> vars = {p::density, p::temperature, p::velocity, p::energy};

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int pB_lo = imap[p::bfield].first;
  const int pB_hi = imap[p::bfield].second;
  const int prho = imap[p::density].first;
  const int ptemp = imap[p::temperature].first;
  const int pye = imap[p::ye].second;
  const int penergy = imap[p::energy].first;

  auto geom = Geometry::GetCoordinateSystem(rc);
  // update loop.
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Fill Tracers", 0, max_active_index - 1, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // geom quantities
          Real gcov4[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, z(n), y(n), x(n), gcov4);
          Real lapse = geom.Lapse(CellLocation::Cent, z(n), y(n), x(n));
          Real shift[3];
          geom.ContravariantShift(CellLocation::Cent, z(n), y(n), x(n), shift);

          // Interpolate
          const Real Wvel_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo);
          const Real Wvel_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo + 1);
          const Real Wvel_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_hi);
          //const Real B_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_lo);
          //const Real B_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_lo + 1);
          //const Real B_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_hi);
          const Real rho = LCInterp::Do(0, x(n), y(n), z(n), pack, prho);
          const Real temperature = LCInterp::Do(0, x(n), y(n), z(n), pack, ptemp);
          const Real energy = LCInterp::Do(0, x(n), y(n), z(n), pack, penergy);
          const Real Wvel[] = {Wvel_X1, Wvel_X2, Wvel_X3};
          const Real W = phoebus::GetLorentzFactor(Wvel, gcov4);
          const Real vel_X1 = Wvel_X1 / W;
          const Real vel_X2 = Wvel_X2 / W;
          const Real vel_X3 = Wvel_X3 / W;
          //Real ye;
          //if (pye > 0){
          //  ye = LCInterp::Do(0, x(n), y(n), z(n), pack, pye);
          //} else {
          //  ye = 0.0;
         // }
          //Real lambda[2] = {ye, 0.0};
          //const Real entropy = eos.EntropyFromDensityTemperature(rho, temperature, lambda);

          // store
          s_rho(n) = rho;
          s_temperature(n) = temperature;
          //s_ye(n) = ye;
          s_energy(n) = energy;
          //s_entropy(n) = entropy;
          v1(n) = vel_X1;
          v2(n) = vel_X3;
          v3(n) = vel_X2;
          shift_x(n) = shift[0];
          shift_y(n) = shift[1];
          shift_z(n) = shift[2];
          s_lapse(n) = lapse;
          s_lorentz(n) = W;
          //B1(n) = B_X1;
          //B2(n) = B_X2;
          //B3(n) = B_X3;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

} // FillTracers


} // namespace tracers
