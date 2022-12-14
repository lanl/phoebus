// © 2021-2022. Triad National Security, LLC. All rights reserved.  This
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
#include "radiation/radiation.hpp"

namespace radiation_equilibration {

using radiation::species;
using singularity::RadiationType;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski)),
      "Problem \"radiation_equilibration\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({radmoment_prim::J, radmoment_prim::H,
                              radmoment_internal::xi, radmoment_internal::phi,
                              fluid_prim::density, fluid_prim::temperature,
                              fluid_prim::pressure, fluid_prim::gamma1,
                              fluid_prim::energy, fluid_prim::ye, fluid_prim::velocity},
                             imap);

  auto idJ = imap.GetFlatIdx(radmoment_prim::J);
  auto idH = imap.GetFlatIdx(radmoment_prim::H);
  auto ixi = imap.GetFlatIdx(radmoment_internal::xi);
  auto iphi = imap.GetFlatIdx(radmoment_internal::phi);

  const int iRho = imap[fluid_prim::density].first;
  const int iT = imap[fluid_prim::temperature].first;
  const int iP = imap[fluid_prim::pressure].first;
  const int igm1 = imap[fluid_prim::gamma1].first;
  const int ieng = imap[fluid_prim::energy].first;
  const int pye = imap[fluid_prim::ye].first;
  auto idv = imap.GetFlatIdx(fluid_prim::velocity);

  const auto specB = idJ.GetBounds(1);
  const Real J = pin->GetOrAddReal("radiation_equilibration", "J", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto tracer_pkg = pmb->packages.Get("tracers");
  bool do_tracers = tracer_pkg->Param<bool>("active");
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = pmb->swarm_data.Get()->Get("tracers");
  auto rng_pool_tr = tracer_pkg->Param<RNGPool>("rng_pool");
  const auto num_tracers_total = tracer_pkg->Param<int>("num_tracers");

  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &x_min = pmb->coords.x1f(ib.s);
  const Real &y_min = pmb->coords.x2f(jb.s);
  const Real &z_min = pmb->coords.x3f(kb.s);
  const Real &x_max = pmb->coords.x1f(ib.e + 1);
  const Real &y_max = pmb->coords.x2f(jb.e + 1);
  const Real &z_max = pmb->coords.x3f(kb.e + 1);

  // const int n_tracers_block = (int)((num_tracers_total) / (nx_i * nx_j * nx_k));
  const int n_tracers_block = num_tracers_total;
  auto swarm_d = swarm->GetDeviceContext();

  ParArrayND<int> new_indices;
  swarm->AddEmptyParticles(n_tracers_block, new_indices);

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();

  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  const auto opac =
      pmb->packages.Get("opacity")->template Param<Microphysics::Opacities>("opacities");

  const Real rho0 = pin->GetOrAddReal("radiation_equilibration", "rho0", 1.);
  const Real Tg0 = pin->GetOrAddReal("radiation_equilibration", "Tg0", 1.);
  const Real Tr0 = pin->GetOrAddReal("radiation_equilibration", "Tr0", 1.e-2);
  const Real Ye0 = pin->GetOrAddReal("radiation_equilibration", "Ye0", 0.4);

  // Store runtime parameters for output
  Params &phoebus_params = pmb->packages.Get("phoebus")->AllParams();
  phoebus_params.Add("radiation_equilibration/rho0", rho0);
  phoebus_params.Add("radiation_equilibration/Tg0", Tg0);
  phoebus_params.Add("radiation_equilibration/Tr0", Tr0);
  phoebus_params.Add("radiation_equilibration/Ye0", Ye0);

  /// TODO: (BRR) Fix this junk
  RadiationType dev_species[3] = {species[0], species[1], species[2]};

  pmb->par_for(
      "Phoebus::ProblemGenerator::radiation_equilibration", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real P = eos.PressureFromDensityTemperature(rho0, Tg0);
        const Real eps = eos.InternalEnergyFromDensityTemperature(rho0, Tg0);

        v(iRho, k, j, i) = rho0;
        v(iT, k, j, i) = Tg0;
        v(iP, k, j, i) = P;
        v(ieng, k, j, i) = v(iRho, k, j, i) * eps;
        v(igm1, k, j, i) =
            eos.BulkModulusFromDensityTemperature(v(iRho, k, j, i), v(iT, k, j, i)) /
            v(iP, k, j, i);
        v(pye, k, j, i) = Ye0;
        SPACELOOP(ii) v(idv(ii), k, j, i) = 0.0;

        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {
          SPACELOOP(ii) v(idH(ispec, ii), k, j, i) = 0.0;
          v(idJ(ispec), k, j, i) =
              opac.EnergyDensityFromTemperature(Tr0, dev_species[ispec]);
        }
      });
  if (do_tracers) {
    pmb->par_for(
        "CreateTracers", 0, n_tracers_block - 1, KOKKOS_LAMBDA(const int n) {
          auto rng_gen = rng_pool_tr.get_state();

          // No rejection sampling at the moment.
          // Perhaps this can be improved in the future.
          x(n) = x_min + rng_gen.drand() * (x_max - x_min);
          y(n) = y_min + rng_gen.drand() * (y_max - y_min);
          z(n) = z_min + rng_gen.drand() * (z_max - z_min);

          rng_pool_tr.free_state(rng_gen);
        }); // Create Tracers
  }

  // Initialize samples
  auto radpkg = pmb->packages.Get("radiation");
  if (radpkg->Param<bool>("active")) {
    if (radpkg->Param<std::string>("method") == "mocmc") {
      radiation::MOCMCInitSamples(rc.get());
    }
  }

  fluid::PrimitiveToConserved(rc.get());
  radiation::MomentPrim2Con(rc.get(), IndexDomain::entire);
}

} // namespace radiation_equilibration
