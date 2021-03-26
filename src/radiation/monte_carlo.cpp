//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include "radiation.hpp"

#include "opacity.hpp"

#define ANALYTIC (0)
#define NUMERICAL (1)
#define SAMPLING_METHOD NUMERICAL

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
  const auto num_particles = rad->Param<int>("num_particles");

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
  auto geom = Geometry::GetCoordinateSystem(rc);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real MASS = unit_conv.GetMassCodeToCGS();
  const Real LENGTH = unit_conv.GetLengthCodeToCGS();
  const Real TIME = unit_conv.GetTimeCodeToCGS();
  const Real ENERGY = unit_conv.GetEnergyCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  // TODO(BRR) can I do this with AMR?
  const Real dV_cgs = dx_i * dx_j * dx_k * dt * pow(LENGTH, 3) * TIME;
  const Real dV_code = dx_i * dx_j * dx_k * dt;
  const Real d3x_cgs = dx_i * dx_j * dx_k * pow(LENGTH, 3);
  const Real d3x_code = dx_i*dx_j*dx_k;

  std::vector<std::string> vars(
      {p::ye, p::velocity, "dNdlnu_max", "dNdlnu", "dN", "Ns", iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int iye = imap[p::ye].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
  const int idNdlnu = imap["dNdlnu"].first;
  const int idNdlnu_max = imap["dNdlnu_max"].first;
  const int idN = imap["dN"].first;
  const int iNs = imap["Ns"].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  // TODO(BRR) update this dynamically somewhere else. Get a reasonable starting value
  //const Real wgtC = 1.e72 * tune_emiss;
  Real wgtC = 1.e50;

  pmb->par_for(
      "MonteCarlodNdlnu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Reset radiation four-force
        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
        v(Gye, k, j, i) = 0.;

        auto rng_gen = rng_pool.get_state();
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);
        Real ye = v(iye, k, j, i);

        #if SAMPLING_METHOD == NUMERICAL
        Real dN = 0.;
        Real dNdlnu_max = 0.;
        for (int n = 0; n <= nu_bins; n++) {
          Real nu = nusamp(n);
          Real wgt = GetWeight(wgtC, nu);
          Real Jnu = GetJnu(ye, s, nu);

          dN += Jnu * nu / (pc.h * nu * wgt) * dlnu;
        //  printf("dN += %e\n", Jnu * nu / (pc.h * nu * wgt) * dlnu);

          // Factors of nu in numerator and denominator cancel
          Real dNdlnu = Jnu * d3x_cgs * detG / (pc.h * wgt);
          v(idNdlnu + n, k, j, i) = dNdlnu;
          if (dNdlnu > dNdlnu_max) {
            dNdlnu_max = dNdlnu;
          }
        }

        for (int n = 0; n <= nu_bins; n++) {
          v(idNdlnu + n, k, j, i) /= dNdlnu_max;
        }


        // Trapezoidal rule
        Real nu0 = nusamp[0];
        Real nu1 = nusamp[nu_bins];
        dN -= 0.5 * GetJnu(ye, s, nu0) * nu0 / (pc.h * nu0 * GetWeight(wgtC, nu0)) * dlnu;
        dN -= 0.5 * GetJnu(ye, s, nu1) * nu1 / (pc.h * nu1 * GetWeight(wgtC, nu1)) * dlnu;
        printf("dN dN: %e %e\n",
          0.5 * GetJnu(ye, s, nu0) * nu0 / (pc.h * nu0 * GetWeight(wgtC, nu0)) * dlnu,
          0.5 * GetJnu(ye, s, nu1) * nu1 / (pc.h * nu1 * GetWeight(wgtC, nu1)) * dlnu);
        dN *= d3x_cgs * detG * dt * TIME;

        v(idNdlnu_max, k, j, i) = dNdlnu_max;
        #elif SAMPLING_METHOD == ANALYTIC
        Real dN = GetJ(ye, s)*d3x_cgs*detG*dt*TIME/(pc.h*wgtC);
        #endif

        int Ns = static_cast<int>(dN);
        if (dN - Ns > rng_gen.drand()) {
          Ns++;
        }

        // TODO(BRR) Use a ParArrayND<int> instead of these weird static_casts
        v(idN, k, j, i) = dN;
        v(iNs, k, j, i) = static_cast<Real>(Ns);
        rng_pool.free_state(rng_gen);
      });

  // Reduce dN over zones for calibrating weights (requires w ~ wgtC)
  Real dNtot = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreation",
      DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &dNtot) {
        dNtot += v(idN, k, j, i);
      },
      Kokkos::Sum<Real>(dNtot));
  printf("dNtot: %e\n", dNtot);
  // TODO(BRR) Mpi reduction here.......
  Real wgtCfac = static_cast<Real>(num_particles)/dNtot;
  printf("wgtCfac: %e\n", wgtCfac);
  pmb->par_for(
      "MonteCarlodiNsEval", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();

        Real dN_upd = wgtCfac*v(idN, k, j, i);
        int Ns = static_cast<int>(dN_upd);
        if (dN_upd - Ns > rng_gen.drand()) {
          Ns++;
        }
        printf("dN_upd: %e Ns: %i\n", dN_upd, Ns);

        // TODO(BRR) Use a ParArrayND<int> instead of these weird static_casts
        v(iNs, k, j, i) = static_cast<Real>(Ns);
        rng_pool.free_state(rng_gen);
      });
  int Nstot = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreationNs",
      DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &Nstot) {
        Nstot += static_cast<int>(v(iNs, k, j, i));
      },
      Kokkos::Sum<int>(Nstot));
  printf("Nstot = %i\n", Nstot);
  if (dNtot <= 0) {
    return TaskStatus::complete;
  }
  //exit(-1);


  ParArrayND<int> new_indices;
  const auto new_particles_mask = swarm->AddEmptyParticles(Nstot, new_indices);

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &k0 = swarm->Get<Real>("k0").Get();
  auto &k1 = swarm->Get<Real>("k1").Get();
  auto &k2 = swarm->Get<Real>("k2").Get();
  auto &k3 = swarm->Get<Real>("k3").Get();
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
        index += dN_h(k + kb.s, j + jb.s, i + ib.s);
      }
    }
  }
  starting_index.DeepCopy(starting_index_h);

  auto dNdlnu = rc->Get("dNdlnu").data;
  auto dNdlnu_max = rc->Get("dNdlnu_max").data;


  int ng = parthenon::Globals::nghost;
  ParArrayND<Real> dwdlnu("dwdlnu", nu_bins+1, 1, 1, 4+2*ng);

  // Loop over zones and generate appropriate number of particles in each zone
  pmb->par_for(
      "MonteCarloSourceParticles", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

  // temporary
  for (int n = 0; n <=nu_bins; n++) {
    dwdlnu(n,k,j,i) = 0.;
  }

        // Create tetrad transformation once per zone
        Real Gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
        Real Ucon[4];
        Real vel[4] = {0, v(pvlo, k, j, i), v(pvlo + 1, k, j, i), v(pvlo + 2, k, j, i)};
        GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
        Geometry::Tetrads Tetrads(Ucon, Gcov);
        Real detG = geom.DetG(CellLocation::Cent, k, j, i);
        int dNs = v(iNs, k, j, i);

        // Loop over particles to create in this zone
        //for (int n = 0; n < static_cast<int>(dNs(k, j, i)); n++) {
        for (int n = 0; n < static_cast<int>(dNs); n++) {
          const int m = new_indices(n);
          auto rng_gen = rng_pool.get_state();

          // Create particles at initial time
          t(m) = t0;

          // Create particles at zone centers
          x(m) = minx_i + (i + 0.5) * dx_i;
          y(m) = minx_j + (j + 0.5) * dx_j;
          z(m) = minx_k + (k + 0.5) * dx_k;

          // Sample energy and set weight
          Real nu;
          int counter = 0;
#if SAMPLING_METHOD == NUMERICAL
          do {
            nu = exp(rng_gen.drand() * (lnu_max - lnu_min) + lnu_min);
            counter++;
          } while (rng_gen.drand() > LinearInterpLog(nu, k, j, i, dNdlnu, lnu_min, dlnu));
#elif SAMPLING_METHOD == ANALYTIC
          Real ye = v(iye, k, j, i);
          Real dndlnu, dndlnu_max;
          do {
            nu = exp(rng_gen.drand() * (lnu_max - lnu_min) + lnu_min);
            dndlnu = nu*GetJnu(ye, s, nu)/(pc.h*nu*wgtC/nu);
            Real numax = exp(lnu_max);
            dndlnu_max = numax*GetJnu(ye, s, numax)/(pc.h*numax*wgtC/numax);
            counter++;
          } while (rng_gen.drand() > dndlnu/dndlnu_max);
#endif

          weight(m) = GetWeight(wgtC/wgtCfac, nu);
          //printf("weight: %e\n", weight(m));
          dwdlnu((log(nu) - lnu_min)/dlnu, k, j, i) += weight(m);

          // Encode frequency and randomly sample direction
          Real E = nu * pc.h * CENERGY;
          Real theta = acos(2. * rng_gen.drand() - 1.);
          Real phi = 2. * M_PI * rng_gen.drand();
          Real K_tetrad[4] = {-E, E * cos(theta), E * cos(phi) * sin(theta),
                              E * sin(phi) * sin(theta)};
          Real K_coord[4];
          Tetrads.TetradToCoordCov(K_tetrad, K_coord);

          k0(m) = K_coord[0];
          k1(m) = K_coord[1];
          k2(m) = K_coord[2];
          k3(m) = K_coord[3];

          for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
            // detG is in both numerator and denominator
            v(mu, k, j, i) += 1. / dV_code * weight(m) * K_coord[mu - Gcov_lo];
          }
          // TODO(BRR) lepton sign
          v(Gye, k, j, i) -= 1. / dV_code * Ucon[0] * weight(m) * pc.mp / MASS;
          rng_pool.free_state(rng_gen);

          // TODO(BRR) Now throw away particles temporarily
          swarm_d.MarkParticleForRemoval(m);
        }
      });
//  exit(-1);

  swarm->RemoveMarkedParticles();

  return TaskStatus::complete;
}

TaskStatus MonteCarloTransport(MeshBlock *pmb, const double dt, const double t0) {
  return TaskStatus::complete;
}
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks) {
  return TaskStatus::complete;
}

} // namespace radiation
