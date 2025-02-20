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

  const Real defrag_frac = pin->GetOrAddReal("tracers", "defrag_frac", 0.0);
  PARTHENON_REQUIRE(defrag_frac >= 0.0 && defrag_frac < 1.0,
                    "Tracer defrag fraction must be >= 0 and less than 1");
  params.Add("defrag_frac", defrag_frac);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", time(NULL));
  physics->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  physics->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  static constexpr auto swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  physics->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  physics->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  // thermo variables
  namespace tv = tracer_variables;
  physics->AddSwarmValue(tv::rho::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::temperature::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::ye::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::entropy::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::pressure::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::energy::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::vel_x::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::vel_y::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::vel_z::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::lorentz::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::lapse::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::detgamma::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::shift_x::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::shift_y::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::shift_z::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::mass::name(), swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue(tv::bernoulli::name(), swarm_name, real_swarmvalue_metadata);

  // TEST
  Metadata Inu_swarmvalue_metadata({Metadata::Real}, std::vector<int>{3, 100});

  const bool mhd = pin->GetOrAddBoolean("fluid", "mhd", false);

  if (mhd) {
    physics->AddSwarmValue(tv::B_x::name(), swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue(tv::B_y::name(), swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue(tv::B_z::name(), swarm_name, real_swarmvalue_metadata);
  }

  return physics;
} // Initialize

TaskStatus AdvectTracers(MeshData<Real> *rc, const Real dt) {
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer();

  const auto ndim = pmb->ndim;

  // tracer swarm pack
  static constexpr auto swarm_name = "tracers";
  static const auto desc_tracer =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto pack_tracers = desc_tracer.GetPack(rc);

  static const auto vars = {p::velocity::name()};
  static const auto desc = MakePackDescriptor<p::velocity>(rc);

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  // TODO(BLB): move to sparse packs. requires reworking tracer_rhs
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
  namespace tv = tracer_variables;

  auto *pmb = rc->GetParentPointer();
  auto fluid = pmb->packages.Get("fluid");
  auto &sc = rc->GetSwarmData();
  auto &swarm = sc->Get("tracers");
  auto eos = pmb->packages.Get("eos")->Param<EOS>("d.EOS");

  const auto mhd = fluid->Param<bool>("mhd");

  // tracer swarm pack
  static constexpr auto swarm_name = "tracers";
  static auto desc_tracers =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              tv::vel_x, tv::vel_y, tv::vel_z, tv::rho, tv::temperature,
                              tv::ye, tv::entropy, tv::energy, tv::lorentz, tv::lapse,
                              tv::shift_x, tv::shift_y, tv::shift_z, tv::detgamma,
                              tv::pressure, tv::bernoulli, tv::B_x, tv::B_y, tv::B_z>(
          swarm_name);
  auto pack_tracers = desc_tracers.GetPack(rc);

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
  pmb->par_for(
      "Fill Tracers", 0, pack_tracers.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        const auto [b, n] = pack_tracers.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_tracers.GetContext(b);

        if (swarm_d.IsActive(n)) {

          const Real x = pack_tracers(b, swarm_position::x(), n);
          const Real y = pack_tracers(b, swarm_position::y(), n);
          const Real z = pack_tracers(b, swarm_position::z(), n);

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
          pack_tracers(b, tv::rho(), n) = rho;
          pack_tracers(b, tv::temperature(), n) = temperature;
          pack_tracers(b, tv::ye(), n) = ye;
          pack_tracers(b, tv::energy(), n) = energy;
          pack_tracers(b, tv::entropy(), n) = entropy;
          pack_tracers(b, tv::vel_x(), n) = vel_X1;
          pack_tracers(b, tv::vel_y(), n) = vel_X2;
          pack_tracers(b, tv::vel_z(), n) = vel_X3;
          pack_tracers(b, tv::shift_x(), n) = shift[0];
          pack_tracers(b, tv::shift_y(), n) = shift[1];
          pack_tracers(b, tv::shift_z(), n) = shift[2];
          pack_tracers(b, tv::lapse(), n) = lapse;
          pack_tracers(b, tv::lorentz(), n) = W;
          pack_tracers(b, tv::detgamma(), n) = gdet;
          pack_tracers(b, tv::pressure(), n) = pressure;
          pack_tracers(b, tv::bernoulli(), n) = bernoulli;
          if (mhd) {
            pack_tracers(b, tv::B_x(), n) = B_X1;
            pack_tracers(b, tv::B_y(), n) = B_X2;
            pack_tracers(b, tv::B_z(), n) = B_X3;
          }

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
        }
      });

} // FillTracers

} // namespace tracers
