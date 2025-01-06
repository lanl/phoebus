// © 2021-2023. Triad National Security, LLC. All rights reserved.
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

#include "tracers.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

namespace tracers {
using namespace parthenon::package::prelude;
using parthenon::MakePackDescriptor;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("tracers");
  const bool active = pin->GetOrAddBoolean("physics", "tracers", false);
  physics->AddParam<bool>("active", active);
  if (!active) return physics;

  Params &params = physics->AllParams();

  const int num_tracers = pin->GetOrAddInteger("tracers", "num_tracers", 0);
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
  physics->AddSwarmValue("entropy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("energy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lorentz", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lapse", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("detgamma", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("mass", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("bernoulli", swarm_name, real_swarmvalue_metadata);

  const bool mhd = pin->GetOrAddBoolean("fluid", "mhd", false);

  if (mhd) {
    physics->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  return physics;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *rc, const Real dt) {
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer();
  Mesh *pmesh = rc->GetMeshPointer();

  const auto ndim = pmb->pmy_mesh->ndim;

  // tracer position swarm pack
  const auto swarm_name = "tracers";
  static auto desc_tracer =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto pack_tracers = desc_tracer.GetPack(rc);

  static const auto vars = {p::velocity::name()};

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;

  const auto geom = Geometry::GetCoordinateSystem(rc);

  // update loop. RK2
  parthenon::par_for(
      "Advect Tracers", 0, pack_tracers.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        const auto [b, n] = pack_tracers.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_tracers.GetContext(b);
        if (swarm_d.IsActive(n)) {
          Real rhs1, rhs2, rhs3;

          const Real x = pack_tracers(b, swarm_position::x(), n);
          const Real y = pack_tracers(b, swarm_position::y(), n);
          const Real z = pack_tracers(b, swarm_position::z(), n);

          // predictor
          tracers_rhs(pack, geom, pvel_lo, pvel_hi, ndim, dt, x, y, z, rhs1, rhs2, rhs3);
          const Real kx = x + 0.5 * dt * rhs1;
          const Real ky = y + 0.5 * dt * rhs2;
          const Real kz = z + 0.5 * dt * rhs3;

          // corrector
          tracers_rhs(pack, geom, pvel_lo, pvel_hi, ndim, dt, kx, ky, kz, rhs1, rhs2,
                      rhs3);

          // update positions
          pack_tracers(b, swarm_position::x(), n) += rhs1 * dt;
          pack_tracers(b, swarm_position::y(), n) += rhs2 * dt;
          pack_tracers(b, swarm_position::z(), n) += rhs3 * dt;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
} // AdvectTracers

/**
 * FillDerived function for tracers.
 * Registered Quantities (in addition to t, x, y, z):
 * rho, T, ye, vel, energy, W_lorentz, pressure,
 * lapse, shift, entropy, detgamma, B, bernoulli
 **/
void FillTracers(MeshBlockData<Real> *rc) {
  using namespace LCInterp;
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer();
  auto fluid = pmb->packages.Get("fluid");
  auto &sc = rc->GetSwarmData();
  auto &swarm = sc->Get("tracers");
  auto eos = pmb->packages.Get("eos")->Param<EOS>("d.EOS");

  const auto mhd = fluid->Param<bool>("mhd");

  // tracer swarm pack
  const auto swarm_name = "tracers";
  static auto desc_tracer_pos =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto pack_tracers_pos = desc_tracer_pos.GetPack(rc);

  // tracer vars pack
  std::vector<std::string> swarm_vars = {
      "vel_x",   "vel_y",    "vel_z",    "rho",      "temperature", "ye",
      "entropy", "energy",   "lorentz",  "lapse",    "shift_x",     "shift_y",
      "shift_z", "detgamma", "pressure", "bernoulli"};

  if (mhd) {
    swarm_vars.push_back("B_x");
    swarm_vars.push_back("B_y");
    swarm_vars.push_back("B_z");
  }

  // make pack and get index map
  static auto desc_tracer_vars = MakeSwarmPackDescriptor<Real>(swarm_name, swarm_vars);
  auto pack_tracers_vars = desc_tracer_vars.GetPack(rc);
  auto pack_tracers_vars_map = desc_tracer_vars.GetMap();

  // TODO(BLB): way to clean this up?
  parthenon::PackIdx spi_vel_x(pack_tracers_vars_map["vel_x"]);
  parthenon::PackIdx spi_vel_y(pack_tracers_vars_map["vel_y"]);
  parthenon::PackIdx spi_vel_z(pack_tracers_vars_map["vel_z"]);
  parthenon::PackIdx spi_B_x(pack_tracers_vars_map["B_x"]);
  parthenon::PackIdx spi_B_y(pack_tracers_vars_map["B_y"]);
  parthenon::PackIdx spi_B_z(pack_tracers_vars_map["B_z"]);
  parthenon::PackIdx spi_rho(pack_tracers_vars_map["rho"]);
  parthenon::PackIdx spi_temperature(pack_tracers_vars_map["temperature"]);
  parthenon::PackIdx spi_ye(pack_tracers_vars_map["ye"]);
  parthenon::PackIdx spi_entropy(pack_tracers_vars_map["entropy"]);
  parthenon::PackIdx spi_energy(pack_tracers_vars_map["energy"]);
  parthenon::PackIdx spi_lorentz(pack_tracers_vars_map["lorentz"]);
  parthenon::PackIdx spi_lapse(pack_tracers_vars_map["lapse"]);
  parthenon::PackIdx spi_shift_x(pack_tracers_vars_map["shift_x"]);
  parthenon::PackIdx spi_shift_y(pack_tracers_vars_map["shift_y"]);
  parthenon::PackIdx spi_shift_z(pack_tracers_vars_map["shift_z"]);
  parthenon::PackIdx spi_detgamma(pack_tracers_vars_map["detgamma"]);
  parthenon::PackIdx spi_pressure(pack_tracers_vars_map["pressure"]);
  parthenon::PackIdx spi_bernoulli(pack_tracers_vars_map["bernoulli"]);

  // hydro vars pack
  std::vector<std::string> vars = {p::density::name(), p::temperature::name(),
                                   p::velocity::name(), p::energy::name(),
                                   p::pressure::name()};
  if (mhd) {
    vars.push_back(p::bfield::name());
  }

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int pB_lo = imap[p::bfield::name()].first;
  const int pB_hi = imap[p::bfield::name()].second;
  const int prho = imap[p::density::name()].first;
  const int ptemp = imap[p::temperature::name()].first;
  const int pye = imap[p::ye::name()].second;
  const int penergy = imap[p::energy::name()].first;
  const int ppres = imap[p::pressure::name()].first;

  auto geom = Geometry::GetCoordinateSystem(rc);
  // update loop.
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Fill Tracers", 0, pack_tracers_pos.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx) {
        const auto [b, n] = pack_tracers_pos.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_tracers_pos.GetContext(b);

        // TODO(BLB): clean up
        const int ivel_x = pack_tracers_vars.GetLowerBound(b, spi_vel_x);
        const int ivel_y = pack_tracers_vars.GetLowerBound(b, spi_vel_y);
        const int ivel_z = pack_tracers_vars.GetLowerBound(b, spi_vel_z);
        const int iB_x = pack_tracers_vars.GetLowerBound(b, spi_B_x);
        const int iB_y = pack_tracers_vars.GetLowerBound(b, spi_B_y);
        const int iB_z = pack_tracers_vars.GetLowerBound(b, spi_B_z);
        const int irho = pack_tracers_vars.GetLowerBound(b, spi_rho);
        const int itemperature = pack_tracers_vars.GetLowerBound(b, spi_temperature);
        const int iye = pack_tracers_vars.GetLowerBound(b, spi_ye);
        const int ientropy = pack_tracers_vars.GetLowerBound(b, spi_entropy);
        const int ienergy = pack_tracers_vars.GetLowerBound(b, spi_energy);
        const int ilorentz = pack_tracers_vars.GetLowerBound(b, spi_lorentz);
        const int ilapse = pack_tracers_vars.GetLowerBound(b, spi_lapse);
        const int ishift_x = pack_tracers_vars.GetLowerBound(b, spi_shift_x);
        const int ishift_y = pack_tracers_vars.GetLowerBound(b, spi_shift_y);
        const int ishift_z = pack_tracers_vars.GetLowerBound(b, spi_shift_z);
        const int idetgamma = pack_tracers_vars.GetLowerBound(b, spi_detgamma);
        const int ipressure = pack_tracers_vars.GetLowerBound(b, spi_pressure);
        const int ibernoulli = pack_tracers_vars.GetLowerBound(b, spi_bernoulli);
        if (swarm_d.IsActive(n)) {

          const Real x = pack_tracers_pos(b, swarm_position::x(), n);
          const Real y = pack_tracers_pos(b, swarm_position::y(), n);
          const Real z = pack_tracers_pos(b, swarm_position::z(), n);

          // geom quantities
          Real gcov4[4][4];
          geom.SpacetimeMetric(0.0, x, y, z, gcov4);
          Real lapse = geom.Lapse(0.0, x, y, z);
          Real shift[3];
          geom.ContravariantShift(0.0, x, y, z, shift);
          const Real gdet = geom.DetGamma(0.0, x, y, z);

          // Interpolate
          const Real Wvel_X1 = LCInterp::Do(0, x, y, z, pack, pvel_lo);
          const Real Wvel_X2 = LCInterp::Do(0, x, y, z, pack, pvel_lo + 1);
          const Real Wvel_X3 = LCInterp::Do(0, x, y, z, pack, pvel_hi);
          Real B_X1 = 0.0;
          Real B_X2 = 0.0;
          Real B_X3 = 0.0;
          if (mhd) {
            B_X1 = LCInterp::Do(0, x, y, z, pack, pB_lo);
            B_X2 = LCInterp::Do(0, x, y, z, pack, pB_lo + 1);
            B_X3 = LCInterp::Do(0, x, y, z, pack, pB_hi);
          }
          const Real rho = LCInterp::Do(0, x, y, z, pack, prho);
          const Real temperature = LCInterp::Do(0, x, y, z, pack, ptemp);
          const Real energy = LCInterp::Do(0, x, y, z, pack, penergy);
          const Real pressure = LCInterp::Do(0, x, y, z, pack, ppres);
          const Real Wvel[] = {Wvel_X1, Wvel_X2, Wvel_X3};
          const Real W = phoebus::GetLorentzFactor(Wvel, gcov4);
          const Real vel_X1 = Wvel_X1 / W;
          const Real vel_X2 = Wvel_X2 / W;
          const Real vel_X3 = Wvel_X3 / W;
          Real ye;
          Real lambda[2] = {0.0, 0.0};
          if (pye > 0) {
            ye = LCInterp::Do(0, x, y, z, pack, pye);
            lambda[1] = ye;
          } else {
            ye = 0.0;
          }
          const Real entropy =
              eos.EntropyFromDensityTemperature(rho, temperature, lambda);

          // bernoulli
          const Real h = 1.0 + energy + pressure / rho;
          const Real bernoulli = -(W / lapse) * h - 1.0;

          // store
          pack_tracers_vars(b, irho, n) = rho;
          pack_tracers_vars(b, itemperature, n) = temperature;
          pack_tracers_vars(b, iye, n) = ye;
          pack_tracers_vars(b, ienergy, n) = energy;
          pack_tracers_vars(b, ientropy, n) = entropy;
          pack_tracers_vars(b, ivel_x, n) = vel_X1;
          pack_tracers_vars(b, ivel_y, n) = vel_X2;
          pack_tracers_vars(b, ivel_z, n) = vel_X3;
          pack_tracers_vars(b, ishift_x, n) = shift[0];
          pack_tracers_vars(b, ishift_y, n) = shift[1];
          pack_tracers_vars(b, ishift_z, n) = shift[2];
          pack_tracers_vars(b, ilapse, n) = lapse;
          pack_tracers_vars(b, ilorentz, n) = W;
          pack_tracers_vars(b, idetgamma, n) = gdet;
          pack_tracers_vars(b, ipressure, n) = pressure;
          pack_tracers_vars(b, ibernoulli, n) = bernoulli;
          if (mhd) {
            pack_tracers_vars(b, iB_x, n) = B_X1;
            pack_tracers_vars(b, iB_y, n) = B_X2;
            pack_tracers_vars(b, iB_z, n) = B_X3;
          }

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
        }
      });

} // FillTracers

} // namespace tracers
