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

  PackIndexMap imap;
  auto v =
      rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                         fluid_prim::energy::name(), fluid_prim::bfield::name(),
                         fluid_prim::ye::name(), fluid_prim::pressure::name(),
                         fluid_prim::temperature::name(), fluid_prim::gamma1::name()},
                        imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure::name()].first;
  const int itmp = imap[fluid_prim::temperature::name()].first;
  const int igm1 = imap[fluid_prim::gamma1::name()].first;

  const Real rho = pin->GetOrAddReal("advection", "rho", 1.0);
  const Real u = pin->GetOrAddReal("advection", "u", 1.0);
  const Real vx = pin->GetOrAddReal("advection", "vx", 0.5);
  const Real vy = pin->GetOrAddReal("advection", "vy", 0.5);
  const Real vz = pin->GetOrAddReal("advection", "vz", 0.5);
  const Real rin = pin->GetOrAddReal("advection", "rin", 0.1);
  const int shapedim = pin->GetOrAddInteger("advection", "shapedim", 2);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
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

        if (iye > 0) {
          v(iye, k, j, i) = (r * r <= rin * rin) ? 1.0 : 0.0;
          eos_lambda[0] = v(iye, k, j, i);
        }

        const Real eps = u / (rho + 1e-20);
        const Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, eos_lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(rho, eps, eos_lambda);

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = u;
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                           v(iprs, k, j, i);
        Real vsq = 0.;
        const Real vcon[3] = {vx, vy, vz};
        SPACELOOP2(ii, jj) { vsq += gcov[ii + 1][jj + 1] * vcon[ii] * vcon[jj]; }
        const Real W = 1. / sqrt(1. - vsq);

        v(ivlo + 0, k, j, i) = W * vx;
        v(ivlo + 1, k, j, i) = W * vy;
        v(ivlo + 2, k, j, i) = W * vz;
      });

  fluid::PrimitiveToConserved(rc.get());

  /* tracer init section */
  if (do_tracers) {
    auto &sc = pmb->swarm_data.Get();
    auto &swarm = pmb->swarm_data.Get()->Get("tracers");
    auto rng_pool = tracer_pkg->Param<RNGPool>("rng_pool");

    const Real &x_min = pmb->coords.Xf<1>(ib.s);
    const Real &y_min = pmb->coords.Xf<2>(jb.s);
    const Real &z_min = pmb->coords.Xf<3>(kb.s);
    const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
    const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
    const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

    // as for num_tracers on each block... will get too many on multiple blocks
    // TODO: distribute amongst blocks.
    const auto num_tracers_total = tracer_pkg->Param<int>("num_tracers");
    const int number_block = num_tracers_total;

    ParArrayND<int> new_indices;
    swarm->AddEmptyParticles(number_block, new_indices);

    auto &x = swarm->Get<Real>("x").Get();
    auto &y = swarm->Get<Real>("y").Get();
    auto &z = swarm->Get<Real>("z").Get();
    auto &id = swarm->Get<int>("id").Get();

    auto swarm_d = swarm->GetDeviceContext();

    const int gid = pmb->gid;
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "ProblemGenerator::Torus::DistributeTracers", 0, max_active_index,
        KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            auto rng_gen = rng_pool.get_state();

            // sample in ye ball
            Real r2 = 1.0 + rin * rin; // init > rin^2
            while (r2 > rin * rin) {
              x(n) = x_min + rng_gen.drand() * (x_max - x_min);
              y(n) = y_min + rng_gen.drand() * (y_max - y_min);
              z(n) = z_min + rng_gen.drand() * (z_max - z_min);
              r2 = x(n) * x(n) + y(n) * y(n) + z(n) * z(n);
            }
            id(n) = num_tracers_total * gid + n;

            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            rng_pool.free_state(rng_gen);
          }
        });
  }
}

} // namespace advection
