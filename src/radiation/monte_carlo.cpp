#include "radiation.hpp"

#include "opacity.hpp"

namespace radiation {

TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, MeshBlockData<Real> *rc, SwarmContainer *sc, const double t0, const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  namespace iv = internal_variables;
  auto rad = pmb->packages.Get("radiation");
  //auto swarm = pmb->swarm_data.Get()->Get("monte_carlo");
  auto swarm = sc->Get("monte_carlo");
  auto rng_pool = rad->Param<RNGPool>("rng_pool");
  const auto tune_emiss = rad->Param<Real>("tune_emiss");

  const auto nu_min = rad->Param<Real>("nu_min");
  const auto nu_max = rad->Param<Real>("nu_max");
  const auto nu_bins = rad->Param<int>("nu_bins");
  const auto dlnu = rad->Param<Real>("dlnu");
  const auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");

  ParArrayND<int> new_indices;
  const auto new_particles_mask = swarm->AddEmptyParticles(num_particles, new_indices);

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

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real Length = unit_conv.GetLengthCodeToCGS();
  const Real Time = unit_conv.GetTimeCodeToCGS();
  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  // TODO(BRR) can I do this with AMR?
  const Real dV = dx_i*dx_j*dx_k*dt*pow(LENGTH,3)*TIME;

  std::vector<std::string> vars({p::ye, "dEdlnu_max", "dEdlnu"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int iye = imap[p::ye].first;
  const int idEdlnu = imap["dEdlnu"].first;
  const int idEdlnu_max = imap["dEdlnu_max"].first;

  pmb->par_for("MonteCarloDEDlnuMax", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {

      Real dEdlnu_max = 0.;
      Real dE = 0.;
      for (int n = 0; n <= nu_bins; n++) {
        Real nu = nusamp(n);
        Real dEdlnu = GetJnu(v(iye, k, j, i), s, nu)*nu;
        v(idEdlnu + n, k, j, i) = dEdlnu;
        if (dEdlnu > dEdlnu_max) {
          dEdlnu_max = dEdlnu;
        }
        // TODO(BRR) gdet
        dE += dEdlnu*dlnu*dV;
      }

      v(idEdlnu_max, k, j, i) = dEdlnu_max;
  });

  pmb->par_for("MonteCarloNumParticles", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {

    });

  pmb->par_for("MonteCarloSourceParticles", 0, new_indices.GetSize()-1, KOKKOS_LAMBDA(const int n) {
    const int m = new_indices(n);
    auto rng_gen = rng_pool.get_state();

    // Randomly sample in space in this meshblock
    // Create particles at zone centers
    //x(n) = minx_i + nx_i * dx_i * rng_gen.drand();
    //y(n) = minx_j + nx_j * dx_j * rng_gen.drand();
    //z(n) = minx_k + nx_k * dx_k * rng_gen.drand();

    // Randomly sample direction, v = c
    Real theta = acos(2. * rng_gen.drand() - 1.);
    Real phi = 2. * M_PI * rng_gen.drand();
    vx(n) = v * sin(theta) * cos(phi);
    vy(n) = v * sin(theta) * sin(phi);
    vz(n) = v * cos(theta);

    rng_pool.free_state(rng_gen);
  });

  return TaskStatus::complete;
}

TaskStatus MonteCarloTransport(MeshBlock *pmb, const double dt,
                               const double t0) {
  return TaskStatus::complete;
}
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks) {
  return TaskStatus::complete;
}

} // namespace radiation
