#include "radiation.hpp"

#include "opacity.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
Real GetWeight(const double wgtC, const double nu) { return wgtC / nu; }

TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                     SwarmContainer *sc, const double t0,
                                     const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  namespace iv = internal_variables;
  auto rad = pmb->packages.Get("radiation");
  // auto swarm = pmb->swarm_data.Get()->Get("monte_carlo");
  auto swarm = sc->Get("monte_carlo");
  auto rng_pool = rad->Param<RNGPool>("rng_pool");
  const auto tune_emiss = rad->Param<Real>("tune_emiss");

  const auto nu_min = rad->Param<Real>("nu_min");
  const auto nu_max = rad->Param<Real>("nu_max");
  const Real lnu_min = log(nu_min);
  const Real lnu_max = log(nu_max);
  const auto nu_bins = rad->Param<int>("nu_bins");
  const auto dlnu = rad->Param<Real>("dlnu");
  const auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");

  // TODO(BRR) temporary
  NeutrinoSpecies s = NeutrinoSpecies::Electron;

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.x1f(ib.s);
  const Real &minx_j = pmb->coords.x2f(jb.s);
  const Real &minx_k = pmb->coords.x3f(kb.s);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real LENGTH = unit_conv.GetLengthCodeToCGS();
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  // TODO(BRR) can I do this with AMR?
  const Real dV = dx_i * dx_j * dx_k * dt * pow(LENGTH, 3) * TIME;

  std::vector<std::string> vars({p::ye, "dNdlnu_max", "dNdlnu", "dN"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int iye = imap[p::ye].first;
  const int idNdlnu = imap["dNdlnu"].first;
  const int idNdlnu_max = imap["dNdlnu_max"].first;
  const int idN = imap["dN"].first;

  // TODO(BRR) update this dynamically somewhere else. Get a reasonable starting value
  const Real wgtC = 1.e44 * tune_emiss;

  pmb->par_for(
      "MonteCarlodNdlnu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();

        Real dNdlnu_max = 0.;
        Real dN = 0.;
        for (int n = 0; n <= nu_bins; n++) {
          Real nu = nusamp(n);
          Real wgt = GetWeight(wgtC, nu);
          Real dNdlnu = GetJnu(v(iye, k, j, i), s, nu) / wgt;
          v(idNdlnu + n, k, j, i) = dNdlnu;
          if (dNdlnu > dNdlnu_max) {
            dNdlnu_max = dNdlnu;
          }
          // TODO(BRR) gdet
          dN += dNdlnu * dlnu * dV;
        }

        // Trapezoidal rule
        dN -= 0.5 * (v(idNdlnu, k, j, i) + v(idNdlnu + nu_bins, k, j, i));
        int Ns = static_cast<int>(dN);
        if (dN - Ns > rng_gen.drand()) {
          Ns++;
        }

        v(idN, k, j, i) = static_cast<Real>(Ns);
        v(idNdlnu_max, k, j, i) = dNdlnu_max;
        rng_pool.free_state(rng_gen);
      });

  // Reduce dN over zones
  int dNtot = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreation",
      DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &dNtot) {
        dNtot += static_cast<int>(v(idN, k, j, i));
      },
      Kokkos::Sum<int>(dNtot));

  ParArrayND<int> new_indices;
  const auto new_particles_mask = swarm->AddEmptyParticles(dNtot, new_indices);

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();
  auto &pi = swarm->Get<int>("i").Get();
  auto &pj = swarm->Get<int>("j").Get();
  auto &pk = swarm->Get<int>("k").Get();
  auto &weight = swarm->Get<Real>("weight").Get();
  auto swarm_d = swarm->GetDeviceContext();

  // Calculate array of starting index for each zone to compute particles
  ParArrayND<int> starting_index("Starting index", nx_k, nx_j, nx_i);
  auto starting_index_h = starting_index.GetHostMirror();
  auto dN = rc->Get("dN").data;
  auto dN_h = dN.GetHostMirrorAndCopy();
  int index = 0;
  for (int k = 0; k < nx_k; k++) {
    for (int j = 0; j < nx_j; j++) {
      for (int i = 0; i < nx_i; i++) {
        starting_index_h(k, j, i) = index;
        // index += dN_h(k+nghost, j+nghost, i+nghost);
        index += dN_h(k + kb.s, j + jb.s, i + ib.s);
      }
    }
  }
  starting_index.DeepCopy(starting_index_h);

  auto dNdlnu = rc->Get("dNdlnu").data;
  auto dNdlnu_max = rc->Get("dNdlnu_max").data;

  // Loop over zones and generate appropriate number of particles in each zone
  pmb->par_for(
      "MonteCarloSourceParticles", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int n = 0; n < static_cast<int>(dN(k, j, i)); n++) {
          const int m = new_indices(n);
          auto rng_gen = rng_pool.get_state();
          printf("zone %i %i %i making a particle!\n", k, j, i);

          // Create particles at initial time
          t(n) = t0;

          // Create particles at zone centers
          x(n) = minx_i + (i + 0.5) * dx_i;
          y(n) = minx_j + (j + 0.5) * dx_j;
          z(n) = minx_k + (k + 0.5) * dx_k;

          // Randomly sample direction, v = c
          Real theta = acos(2. * rng_gen.drand() - 1.);
          Real phi = 2. * M_PI * rng_gen.drand();
          vx(n) = sin(theta) * cos(phi);
          vy(n) = sin(theta) * sin(phi);
          vz(n) = cos(theta);

          // Sample energy and set weight
          Real nu;
          int counter = 0;
          do {
            nu = exp(rng_gen.drand()*(lnu_max - lnu_min) + lnu_min);
            counter++;
          } while (rng_gen.drand() > LinearInterpLog(nu, k, j, i, dNdlnu, lnu_min, dlnu)/dNdlnu_max(k, j, i));
          printf("nu: %e\n", nu);


          weight(n) = GetWeight(wgtC, nu);

          rng_pool.free_state(rng_gen);
        }
      });
  exit(-1);

  pmb->par_for(
      "MonteCarloSourceParticles", 0, new_indices.GetSize() - 1,
      KOKKOS_LAMBDA(const int n) {
        printf("%s:%i\n", __FILE__, __LINE__);
        const int m = new_indices(n);
        auto rng_gen = rng_pool.get_state();
        printf("%s:%i\n", __FILE__, __LINE__);

        // Randomly sample in space in this meshblock
        // Create particles at zone centers
        // x(n) = minx_i + nx_i * dx_i * rng_gen.drand();
        // y(n) = minx_j + nx_j * dx_j * rng_gen.drand();
        // z(n) = minx_k + nx_k * dx_k * rng_gen.drand();

        // Randomly sample direction, v = c
        printf("%s:%i\n", __FILE__, __LINE__);
        Real theta = acos(2. * rng_gen.drand() - 1.);
        Real phi = 2. * M_PI * rng_gen.drand();
        printf("%s:%i\n", __FILE__, __LINE__);
        vx(n) = sin(theta) * cos(phi);
        printf("%s:%i\n", __FILE__, __LINE__);
        printf("n: %i val: %e\n", n, sin(theta) * sin(phi));
        vy(n) = sin(theta) * sin(phi);
        printf("%s:%i\n", __FILE__, __LINE__);
        vz(n) = cos(theta);
        printf("%s:%i\n", __FILE__, __LINE__);

        rng_pool.free_state(rng_gen);
      });
  printf("%s:%i\n", __FILE__, __LINE__);

  return TaskStatus::complete;
}

TaskStatus MonteCarloTransport(MeshBlock *pmb, const double dt, const double t0) {
  return TaskStatus::complete;
}
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks) {
  return TaskStatus::complete;
}

} // namespace radiation
