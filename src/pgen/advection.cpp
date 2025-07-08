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

#include <cmath>
#include <string>

#include "pgen/pgen.hpp"

namespace advection {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"advection\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();
  auto tracer_pkg = pmb->packages.Get("tracers");
  bool do_tracers = tracer_pkg->Param<bool>("active");

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
                        fluid_prim::bfield, fluid_prim::ye, fluid_prim::pressure, 
                        fluid_prim::temperature, fluid_prim::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  const Real rho = pin->GetOrAddReal("advection", "rho", 1.0);
  const Real u = pin->GetOrAddReal("advection", "u", 1.0);
  const Real vx = pin->GetOrAddReal("advection", "vx", 0.5);
  const Real vy = pin->GetOrAddReal("advection", "vy", 0.5);
  const Real vz = pin->GetOrAddReal("advection", "vz", 0.5);
  const Real rin = pin->GetOrAddReal("advection", "rin", 0.1);
  const int shapedim = pin->GetOrAddInteger("advection", "shapedim", 2);

  auto &coords = pmb->coords;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  pmb->par_for(
      "Phoebus::ProblemGenerator::advection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real eos_lambda[2];
        Real x = coords.Xc<1>(i);
        Real y = (ndim > 1 && shapedim > 1) ? coords.Xc<2>(j) : 0;
        Real z = (ndim > 2 && shapedim > 2) ? coords.Xc<3>(k) : 0;
        Real r = std::sqrt(x * x + y * y + z * z);

        Real gcov[4][4];
        geom.SpacetimeMetric(0.0, x, y, z, gcov);

        if (v.Contains(0, fluid_prim::ye())) {
          v(0, fluid_prim::ye(), k, j, i) = (r * r <= rin * rin) ? 1.0 : 0.0;
          eos_lambda[0] = v(0, fluid_prim::ye(), k, j, i);
        }

        const Real eps = u / (rho + 1e-20);
        const Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, eos_lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(rho, eps, eos_lambda);

        v(0, fluid_prim::density(), k, j, i) = rho;
        v(0, fluid_prim::pressure(), k, j, i) = P;
        v(0, fluid_prim::energy(), k, j, i) = u;
        v(0, fluid_prim::temperature(), k, j, i) = T;
        v(0, fluid_prim::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, fluid_prim::density(), k, j, i), v(0, fluid_prim::temperature(), k, j, i), eos_lambda) /
                           v(0, fluid_prim::pressure(), k, j, i);
        Real vsq = 0.;
        const Real vcon[3] = {vx, vy, vz};
        SPACELOOP2(ii, jj) { vsq += gcov[ii + 1][jj + 1] * vcon[ii] * vcon[jj]; }
        const Real W = 1. / sqrt(1. - vsq);

        v(0, fluid_prim::velocity(0), k, j, i) = W * vx;
        v(0, fluid_prim::velocity(1), k, j, i) = W * vy;
        v(0, fluid_prim::velocity(2), k, j, i) = W * vz;
      });

  fluid::PrimitiveToConserved(rc.get());
}

void PostInitializationModifier(ParameterInput *pin, Mesh *pmesh) {

  const int ndim = pmesh->ndim;
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    auto tracer_pkg = pmb->packages.Get("tracers");
    bool do_tracers = tracer_pkg->Param<bool>("active");
    const Real rin = pin->GetOrAddReal("advection", "rin", 0.1);
    const Real v_inner = ndim == 3   ? (4. / 3.) * M_PI * std::pow(rin, 3.)
                         : ndim == 2 ? M_PI * rin * rin
                                     : rin;

    auto geom = Geometry::GetCoordinateSystem(rc.get());
    auto coords = pmb->coords;
    if (do_tracers) {
      const auto num_tracers_total = tracer_pkg->Param<int>("num_tracers");
      auto rng_pool = tracer_pkg->Param<RNGPool>("rng_pool");
      auto &swarm = rc->GetSwarmData()->Get("tracers");

      auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
      auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
      auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
      Real number_block = 0.0;

      // Get fraction of block containing ye sphere
      pmb->par_reduce(
          "Phoebus::ProblemGenerator::Torus::BlockTracerNumber", kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i,
                        Real &number_block_reduce) {
            const Real dx1 = coords.Dxc<1>(k, j, i);
            const Real dx2 = coords.Dxc<2>(k, j, i);
            const Real dx3 = coords.Dxc<3>(k, j, i);
            const Real x1 = coords.Xc<1>(k, j, i);
            const Real x2 = coords.Xc<2>(k, j, i);
            const Real x3 = coords.Xc<3>(k, j, i);

            if (x1 * x1 + x2 * x2 + x3 * x3 < rin * rin) {
              Real vol_block = dx1 * dx2 * dx3;
              number_block_reduce += vol_block;
            }
          },
          Kokkos::Sum<Real>(number_block));
      number_block /= v_inner;
      number_block = number_block * num_tracers_total;
      const int num_tracers_block = (int)number_block;

      // distribute
      auto new_particles_context = swarm->AddEmptyParticles(number_block);

      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto &id = swarm->Get<int>("id").Get();

      auto swarm_d = swarm->GetDeviceContext();

      const int gid = pmb->gid;
      const int max_active_index = new_particles_context.GetNewParticlesMaxIndex();
      pmb->par_for(
          "ProblemGenerator::Advection::DistributeTracers", 0, max_active_index,
          KOKKOS_LAMBDA(const int new_n) {
            const int n = new_particles_context.GetNewParticleIndex(new_n);
            auto rng_gen = rng_pool.get_state();

            // sample in ye ball
            Real r2 = 1.0 + rin * rin; // init > rin^2
            while (r2 > rin * rin) {
              x(n) = -rin + rng_gen.drand() * 2.0 * rin; // x \in [-rin, +rin]
              y(n) = -rin + rng_gen.drand() * 2.0 * rin;
              z(n) = -rin + rng_gen.drand() * 2.0 * rin;
              r2 = x(n) * x(n) + y(n) * y(n) + z(n) * z(n);
            }
            id(n) = num_tracers_total * gid + n;

            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            rng_pool.free_state(rng_gen);
          });
    }
  }

} // PostInitializationModifier

} // namespace advection
