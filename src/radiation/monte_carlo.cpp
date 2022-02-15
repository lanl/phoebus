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

#include "geodesics.hpp"
#include "phoebus_utils/robust.hpp"
#include "radiation.hpp"

using Geometry::CoordSysMeshBlock;
using Geometry::NDFULL;

namespace radiation {

using namespace singularity::neutrinos;
using singularity::RadiationType;

KOKKOS_INLINE_FUNCTION
Real GetWeight(const double wgtC, const double nu) { return wgtC / nu; }

TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                     SwarmContainer *sc, const double t0,
                                     const double dt) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto opac = pmb->packages.Get("opacity");
  auto rad = pmb->packages.Get("radiation");
  auto swarm = sc->Get("monte_carlo");
  auto rng_pool = rad->Param<RNGPool>("rng_pool");

  const auto nu_min = rad->Param<Real>("nu_min");
  const auto nu_max = rad->Param<Real>("nu_max");
  const Real lnu_min = log(nu_min);
  const Real lnu_max = log(nu_max);
  const auto nu_bins = rad->Param<int>("nu_bins");
  const auto dlnu = rad->Param<Real>("dlnu");
  const auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const auto num_particles = rad->Param<int>("num_particles");
  const auto remove_emitted_particles =
      rad->Param<bool>("remove_emitted_particles");

  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  bool do_species[3] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti"),
                        rad->Param<bool>("do_nu_heavy")};

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i =
      pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j =
      pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k =
      pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.x1f(ib.s);
  const Real &minx_j = pmb->coords.x2f(jb.s);
  const Real &minx_k = pmb->coords.x3f(kb.s);
  auto geom = Geometry::GetCoordinateSystem(rc);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real MASS = unit_conv.GetMassCodeToCGS();
  const Real LENGTH = unit_conv.GetLengthCodeToCGS();
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  const Real d3x = dx_i * dx_j * dx_k;

  const Real h_code = pc::h*TIME/(MASS*LENGTH*LENGTH);

  std::vector<std::string> vars({p::density, p::temperature, p::ye, p::velocity,
                                 "dNdlnu_max", "dNdlnu", "dN", "Ns", iv::Gcov,
                                 iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int iye = imap[p::ye].first;
  const int pdens = imap[p::density].first;
  const int ptemp = imap[p::temperature].first;
  const int pvlo = imap[p::velocity].first;
  const int pvhi = imap[p::velocity].second;
  const int idNdlnu = imap["dNdlnu"].first;
  const int idNdlnu_max = imap["dNdlnu_max"].first;
  const int idN = imap["dN"].first;
  const int iNs = imap["Ns"].first;
  const int Gcov_lo = imap[iv::Gcov].first;
  const int Gcov_hi = imap[iv::Gcov].second;
  const int Gye = imap[iv::Gye].first;

  // TODO(BRR) update this dynamically somewhere else. Get a reasonable starting
  // value
  Real wgtC = 1.e40; // Typical-ish value

  pmb->par_for(
      "MonteCarloZeroFiveForce", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int mu = Gcov_lo; mu <= Gcov_lo + 3; mu++) {
          v(mu, k, j, i) = 0.;
        }
        v(Gye, k, j, i) = 0.;
      });

  for (int sidx = 0; sidx < 3; sidx++) {
    if (do_species[sidx]) {
      auto s = species[sidx];
      pmb->par_for(
          "MonteCarlodNdlnu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto rng_gen = rng_pool.get_state();
            Real detG = geom.DetG(CellLocation::Cent, k, j, i);
            Real ye = v(iye, k, j, i);

            Real dN = 0.;
            Real dNdlnu_max = 0.;
            for (int n = 0; n <= nu_bins; n++) {
              Real nu = nusamp(n);
              Real ener = h_code * nu * TIME;
              Real wgt = GetWeight(wgtC, nu);
              Real Jnu = d_opacity.EmissivityPerNu(v(pdens,k,j,i), v(ptemp,k,j,i), ye, s, nu * TIME);

              //Jnu /= (4.*M_PI);

              dN += Jnu / (ener * wgt) * (nu * dlnu * TIME);

              // Note that factors of nu in numerator and denominator cancel
              Real dNdlnu = Jnu * d3x * detG / (h_code * wgt);
              v(idNdlnu + sidx + n * NumRadiationTypes, k, j, i) = dNdlnu;
              if (dNdlnu > dNdlnu_max) {
                dNdlnu_max = dNdlnu;
              }
            }

            for (int n = 0; n <= nu_bins; n++) {
              v(idNdlnu + sidx + n * NumRadiationTypes, k, j, i) /= dNdlnu_max;
            }

            // Trapezoidal rule
            Real nu0 = nusamp[0];
            Real nu1 = nusamp[nu_bins];
            dN -= 0.5 * d_opacity.EmissivityPerNu(v(pdens,k,j,i), v(ptemp,k,j,i), ye, s, nu0 * TIME) /
              (h_code * GetWeight(wgtC, nu0)) * dlnu;
            dN -= 0.5 * d_opacity.EmissivityPerNu(v(pdens,k,j,i), v(ptemp,k,j,i), ye, s, nu1 * TIME) /
              (h_code * GetWeight(wgtC, nu1)) * dlnu;
            dN *= d3x * detG * dt;

            v(idNdlnu_max + sidx, k, j, i) = dNdlnu_max;

            int Ns = static_cast<int>(dN);
            if (dN - Ns > rng_gen.drand()) {
              Ns++;
            }

            // TODO(BRR) Use a ParArrayND<int> instead of these weird static_casts
            v(idN + sidx, k, j, i) = dN;
            v(iNs + sidx, k, j, i) = static_cast<Real>(Ns);
            rng_pool.free_state(rng_gen);
          });
    }
  }

  // Reduce dN over zones for calibrating weights (requires w ~ wgtC)
  Real dNtot = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreation",
      DevExecSpace(), 0, 2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i,
                    Real &dNtot) {
        if (do_species[sidx]) {
          dNtot += v(idN + sidx, k, j, i);
        }
      },
      Kokkos::Sum<Real>(dNtot));

  // TODO(BRR) Mpi reduction here....... this really needs to be a separate task
  // TODO(BRR) actually use tuning parameter
  Real wgtCfac = static_cast<Real>(num_particles) / dNtot;

  pmb->par_for(
      "MonteCarlodiNsEval", 0, 2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i) {
        if (do_species[sidx]) {
          auto rng_gen = rng_pool.get_state();

          Real dN_upd = wgtCfac * v(idN + sidx, k, j, i);
          int Ns = static_cast<int>(dN_upd);
          if (dN_upd - Ns > rng_gen.drand()) {
            Ns++;
          }

          // TODO(BRR) Use a ParArrayND<int> instead of these weird static_casts
          v(iNs + sidx, k, j, i) = static_cast<Real>(Ns);
          rng_pool.free_state(rng_gen);
        }
      });
  int Nstot = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreationNs",
      DevExecSpace(), 0, 2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i,
                    int &Nstot) {
        if (do_species[sidx]) {
          Nstot += static_cast<int>(v(iNs + sidx, k, j, i));
        }
      },
      Kokkos::Sum<int>(Nstot));

  printf("Nstot: %i\n", Nstot);
  const auto num_emitted = rad->Param<Real>("num_emitted");
  rad->UpdateParam<Real>("num_emitted", num_emitted + Nstot);

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
  auto &swarm_species = swarm->Get<int>("species").Get();
  auto swarm_d = swarm->GetDeviceContext();

  // Calculate array of starting index for each zone to compute particles
  ParArrayND<int> starting_index("Starting index", 3, nx_k, nx_j, nx_i);
  auto starting_index_h = starting_index.GetHostMirror();
  auto dN = rc->Get("Ns").data;
  auto dN_h = dN.GetHostMirrorAndCopy();
  int index = 0;
  for (int sidx = 0; sidx < 3; sidx++) {
    for (int k = 0; k < nx_k; k++) {
      for (int j = 0; j < nx_j; j++) {
        for (int i = 0; i < nx_i; i++) {
          starting_index_h(sidx, k, j, i) = index;
          index += static_cast<int>(dN_h(sidx, k + kb.s, j + jb.s, i + ib.s));
        }
      }
    }
  }
  starting_index.DeepCopy(starting_index_h);

  auto dNdlnu = rc->Get("dNdlnu").data;
  // auto dNdlnu_max = rc->Get("dNdlnu_max").data;

  // Loop over zones and generate appropriate number of particles in each zone
  for (int sidx = 0; sidx < 3; sidx++) {
    if (do_species[sidx]) {
      auto s = species[sidx];

      pmb->par_for(
          "MonteCarloSourceParticles", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // Create tetrad transformation once per zone
            Real Gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, Gcov);
            Real Ucon[4];
            Real vel[3] = {v(pvlo, k, j, i),
                           v(pvlo + 1, k, j, i),
                           v(pvlo + 2, k, j, i)};
            GetFourVelocity(vel, geom, CellLocation::Cent, k, j, i, Ucon);
            Geometry::Tetrads Tetrads(Ucon, Gcov);
            Real detG = geom.DetG(CellLocation::Cent, k, j, i);
            int dNs = v(iNs + sidx, k, j, i);

            // Loop over particles to create in this zone
            for (int n = 0; n < static_cast<int>(dNs); n++) {
              const int m = new_indices(
                  starting_index(sidx, k - kb.s, j - jb.s, i - ib.s) + n);
              auto rng_gen = rng_pool.get_state();

              // Set particle species
              swarm_species(m) = static_cast<int>(s);

              // Create particles at initial time
              t(m) = t0;

              // Create particles at zone centers
              x(m) = minx_i + (i - ib.s + 0.5) * dx_i;
              y(m) = minx_j + (j - jb.s + 0.5) * dx_j;
              z(m) = minx_k + (k - kb.s + 0.5) * dx_k;

              // Sample energy and set weight
              Real nu;
              int counter = 0;
              do {
                nu = exp(rng_gen.drand() * (lnu_max - lnu_min) + lnu_min);
                counter++;
              } while (rng_gen.drand() > LogLinearInterp(nu, sidx, k, j, i,
                                                         dNdlnu, lnu_min,
                                                         dlnu));

              weight(m) = GetWeight(wgtC / wgtCfac, nu);

              // Encode frequency and randomly sample direction
              Real E = nu * TIME * h_code;
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
                v(mu, k, j, i) +=
                    1. / (d3x * dt) * weight(m) * K_coord[mu - Gcov_lo];
              }
              // TODO(BRR) lepton sign
              v(Gye, k, j, i) -=
                  1. / (d3x * dt) * Ucon[0] * weight(m) * pc::mp / MASS;

              rng_pool.free_state(rng_gen);
            }
          });
    }
  }

  if (remove_emitted_particles) {
    pmb->par_for(
        "MonteCarloRemoveEmittedParticles", 0, 2, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e,
        KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i) {
          if (do_species[sidx]) {
            int dNs = v(iNs + sidx, k, j, i);
            // Loop over particles to create in this zone
            for (int n = 0; n < static_cast<int>(dNs); n++) {
              const int m = new_indices(
                  starting_index(sidx, k - kb.s, j - jb.s, i - ib.s) + n);
              swarm_d.MarkParticleForRemoval(m);
            }
          }
        });
    swarm->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

TaskStatus MonteCarloTransport(MeshBlock *pmb, MeshBlockData<Real> *rc,
                               SwarmContainer *sc, const double t0,
                               const double dt) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace iv = internal_variables;
  auto rad = pmb->packages.Get("radiation");
  auto swarm = sc->Get("monte_carlo");
  auto opac = pmb->packages.Get("opacity");
  auto rng_pool = rad->Param<RNGPool>("rng_pool");

  const auto num_particles = rad->Param<int>("num_particles");
  const auto absorption = rad->Param<bool>("absorption");

  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const Real &dx_i =
      pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j =
      pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k =
      pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real d4x = dx_i * dx_j * dx_k * dt;
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &k0 = swarm->Get<Real>("k0").Get();
  auto &k1 = swarm->Get<Real>("k1").Get();
  auto &k2 = swarm->Get<Real>("k2").Get();
  auto &k3 = swarm->Get<Real>("k3").Get();
  auto &weight = swarm->Get<Real>("weight").Get();
  auto &swarm_species = swarm->Get<int>("species").Get();
  auto swarm_d = swarm->GetDeviceContext();

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real MASS = unit_conv.GetMassCodeToCGS();
  const Real LENGTH = unit_conv.GetLengthCodeToCGS();
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  const Real h_code = pc::h*TIME/(MASS*LENGTH*LENGTH);

  std::vector<std::string> vars(
      {p::density, p::ye, p::velocity, p::temperature, iv::Gcov, iv::Gye});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int prho = imap[p::density].first;
  const int iye = imap[p::ye].first;
  const int ivlo = imap[p::velocity].first;
  const int ivhi = imap[p::velocity].second;
  const int itemp = imap[p::temperature].first;
  const int iGcov_lo = imap[iv::Gcov].first;
  const int iGcov_hi = imap[iv::Gcov].second;
  const int iGye = imap[iv::Gye].first;

  ParArray1D<Real> num_interactions("Number interactions", 2);

  printf("Particles: %i\n", swarm->GetNumActive());

  pmb->par_for(
      "MonteCarloTransport", 0, swarm->GetMaxActiveIndex(),
      KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();

          auto s = static_cast<RadiationType>(swarm_species(n));

          // TODO(BRR) Get u^mu, evaluate -k.u
          const Real nu = -k0(n) / h_code;

          // TODO(BRR) Get K^0 via metric
          Real Kcon0 = -k0(n);
          Real dlam = dt / Kcon0;

          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          Real alphanu =
              4.*M_PI * d_opacity.AbsorptionCoefficient(v(prho,k,j,i), v(itemp,k,j,i), v(iye,k,j,i), s, nu);

          Real dtau_abs = alphanu * dt; // c = 1 in code units

          bool absorbed = false;

          // TODO(BRR) This is first order in space to avoid extra communications
          if (absorption) {
            // Process absorption events
            Real xabs = -log(rng_gen.drand());
            if (xabs <= dtau_abs) {
              // Deposit energy-momentum and lepton number in fluid
              Kokkos::atomic_add(&(v(iGcov_lo, k, j, i)),
                                 -1. / d4x * weight(n) * k0(n));
              Kokkos::atomic_add(&(v(iGcov_lo + 1, k, j, i)),
                                 -1. / d4x * weight(n) * k1(n));
              Kokkos::atomic_add(&(v(iGcov_lo + 2, k, j, i)),
                                 -1. / d4x * weight(n) * k2(n));
              Kokkos::atomic_add(&(v(iGcov_lo + 3, k, j, i)),
                                 -1. / d4x * weight(n) * k3(n));
              // TODO(BRR) Add Ucon[0] in the below
              Kokkos::atomic_add(&(v(iGye, k, j, i)),
                                 1. / d4x * weight(n) * pc::mp / MASS);

              absorbed = true;
              Kokkos::atomic_add(&(num_interactions[0]),
                1.);
              swarm_d.MarkParticleForRemoval(n);
            }
          }

          if (absorbed == false) {
            PushParticle(t(n), x(n), y(n), z(n), k0(n), k1(n), k2(n), k3(n), dt,
                         geom);

            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n),
                                          on_current_mesh_block);
          }

          rng_pool.free_state(rng_gen);
        }
      });

  if (absorption) {
    swarm->RemoveMarkedParticles();
  }

  auto num_interactions_h = Kokkos::create_mirror_view(num_interactions);
  Kokkos::deep_copy(num_interactions_h, num_interactions);

  const auto num_absorbed = rad->Param<Real>("num_absorbed");
  const auto num_scattered = rad->Param<Real>("num_scattered");
  rad->UpdateParam<Real>("num_absorbed", num_absorbed + num_interactions_h[0]);
  rad->UpdateParam<Real>("num_scattered", num_scattered + num_interactions_h[1]);

  return TaskStatus::complete;
}
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks) {
  return TaskStatus::complete;
}

// Reduce particle sampling resolution statistics from per-mesh to global as part of global
// reduction.
TaskStatus MonteCarloUpdateParticleResolution(Mesh *pmesh, std::vector<Real> *resolution) {
  auto rad = pmesh->packages.Get("radiation");
  const auto num_emitted = rad->Param<Real>("num_emitted");
  const auto num_absorbed = rad->Param<Real>("num_absorbed");
  const auto num_scattered = rad->Param<Real>("num_scattered");
  (*resolution)[static_cast<int>(ParticleResolution::emitted)] += num_emitted;
  (*resolution)[static_cast<int>(ParticleResolution::absorbed)] += num_absorbed;
  (*resolution)[static_cast<int>(ParticleResolution::scattered)] += num_scattered;
  return TaskStatus::complete;
}

// Update particle resolution tuning parameters and reset counters if it is time for an update.
//TaskStatus MonteCarloUpdateTuning(MeshBlock *pmb, MeshBlockData<Real> *rc,
//  SwarmContainer *sc, const double t0, const double dt) {
TaskStatus MonteCarloUpdateTuning(Mesh *pmesh, std::vector<Real> *resolution,
  const double t0, const double dt) {
  auto rad = pmesh->packages.Get("radiation");
  const auto t_tune_emission = rad->Param<Real>("t_tune_emission");
  const auto dt_tune_emission = rad->Param<Real>("dt_tune_emission");
  const auto t_tune_scattering = rad->Param<Real>("t_tune_scattering");
  const auto dt_tune_scattering = rad->Param<Real>("dt_tune_scattering");
  const auto num_particles = rad->Param<int>("num_particles");

  // TODO(BRR) This should be Rout_rad (add it) or max cartesian size
  const Real L = 1.;

  printf("t_tune_emission: %e t0 + dt: %e\n", t_tune_emission, t0 + dt);
    const auto num_emitted = (*resolution)[static_cast<int>(ParticleResolution::emitted)];
    const auto num_absorbed = (*resolution)[static_cast<int>(ParticleResolution::absorbed)];
    printf("emitted: %e absorbed: %e\n", num_emitted, num_absorbed);

  if (t_tune_emission < t0 + dt) {
    const auto num_emitted = (*resolution)[static_cast<int>(ParticleResolution::emitted)];
    const auto num_absorbed = (*resolution)[static_cast<int>(ParticleResolution::absorbed)];
    printf("emitted: %e absorbed: %e\n", num_emitted, num_absorbed);

    const Real real = num_emitted - num_absorbed;
    const Real ideal = dt_tune_emission*num_particles;
    Real correction = ideal/real;

    // Limit strength of correction
    robust::make_bounded(correction, 3./4., 4./3.);

    const auto tune_emission = rad->Param<Real>("tune_emission");
    rad->UpdateParam<Real>("tune_emission", correction*tune_emission);

    rad->UpdateParam<Real>("num_emitted", 0.);
    rad->UpdateParam<Real>("num_absorbed", 0.);
    rad->UpdateParam<Real>("t_tune_emission", t_tune_emission + dt_tune_emission);
  }

  if (t_tune_scattering < t0 + dt) {
    const auto num_scattered = (*resolution)[static_cast<int>(ParticleResolution::scattered)];

    const Real real = num_scattered;
    const Real ideal = dt_tune_scattering*num_particles;
    Real correction = ideal/real;

    // Limit strength of correction
    robust::make_bounded(correction, 0.5, 2.);

    const auto tune_scattering = rad->Param<Real>("tune_scattering");
    rad->UpdateParam<Real>("tune_scattering", correction*tune_scattering);

    rad->UpdateParam<Real>("num_scattered", 0.);
    rad->UpdateParam<Real>("t_tune_scattering", t_tune_scattering + dt_tune_scattering);
  }

  return TaskStatus::complete;
}

/*TaskStatus MonteCarloEstimateParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
  SwarmContainer *sc, const double t0, const double dt, Real *dNtot) {

  namespace fp = fluid_prim;
  namespace fc = fluid_cons;
  namespace iv = internal_variables;
  auto opac = pmb->packages.Get("opacity");
  auto rad = pmb->packages.Get("radiation");
  auto swarm = sc->Get("monte_carlo");
  auto rng_pool = rad->Param<RNGPool>("rng_pool");
  const auto num_emitted = rad->Param<Real>("tune_emiss");

  const auto nu_min = rad->Param<Real>("nu_min");
  const auto nu_max = rad->Param<Real>("nu_max");
  const Real lnu_min = log(nu_min);
  const Real lnu_max = log(nu_max);
  const auto nu_bins = rad->Param<int>("nu_bins");
  const auto dlnu = rad->Param<Real>("dlnu");
  const auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const auto num_particles = rad->Param<int>("num_particles");
  const auto remove_emitted_particles =
      rad->Param<bool>("remove_emitted_particles");

  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  bool do_species[3] = {rad->Param<bool>("do_nu_electron"),
                        rad->Param<bool>("do_nu_electron_anti"),
                        rad->Param<bool>("do_nu_heavy")};

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i =
      pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j =
      pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k =
      pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
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
  const Real DENSITY = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMPERATURE = unit_conv.GetTemperatureCodeToCGS();
  const Real CENERGY = unit_conv.GetEnergyCGSToCode();
  const Real CDENSITY = unit_conv.GetNumberDensityCGSToCode();
  const Real CTIME = unit_conv.GetTimeCGSToCode();
  const Real CPOWERDENS = CENERGY * CDENSITY / CTIME;

  const Real dV_cgs = dx_i * dx_j * dx_k * dt * pow(LENGTH, 3) * TIME;
  const Real dV_code = dx_i * dx_j * dx_k * dt;
  const Real d3x_cgs = dx_i * dx_j * dx_k * pow(LENGTH, 3);
  const Real d3x_code = dx_i * dx_j * dx_k;

  std::vector<std::string> vars({fp::density, fp::temperature, fp::ye, fp::velocity,
                                 "dNdlnu_max", "dNdlnu", "dN", "Ns"});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int iye = imap[fp::ye].first;
  const int pdens = imap[fp::density].first;
  const int ptemp = imap[fp::temperature].first;
  const int pvlo = imap[fp::velocity].first;
  const int pvhi = imap[fp::velocity].second;
  const int idNdlnu = imap["dNdlnu"].first;
  const int idNdlnu_max = imap["dNdlnu_max"].first;
  const int idN = imap["dN"].first;
  const int iNs = imap["Ns"].first;

  // TODO(BRR) update this dynamically somewhere else. Get a reasonable starting
  // value
  Real wgtC = 1.e50; // Typical-ish value

  for (int sidx = 0; sidx < 3; sidx++) {
    if (do_species[sidx]) {
      auto s = species[sidx];
      pmb->par_for(
          "MonteCarlodNdlnu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto rng_gen = rng_pool.get_state();
            Real detG = geom.DetG(CellLocation::Cent, k, j, i);
            Real ye = v(iye, k, j, i);

            const Real rho_cgs = v(pdens, k, j, i) * DENSITY;
            const Real T_cgs = v(ptemp, k, j, i) * TEMPERATURE;

            Real dN = 0.;
            Real dNdlnu_max = 0.;
            for (int n = 0; n <= nu_bins; n++) {
              Real nu = nusamp(n);
              Real wgt = GetWeight(wgtC, nu);
              Real Jnu = d_opacity.EmissivityPerNu(rho_cgs, T_cgs, ye, s, nu);

              dN += Jnu * nu / (pc::h * nu * wgt) * dlnu;

              // Factors of nu in numerator and denominator cancel
              Real dNdlnu = Jnu * d3x_cgs * detG / (pc::h * wgt);
              v(idNdlnu + sidx + n * NumRadiationTypes, k, j, i) = dNdlnu;
              if (dNdlnu > dNdlnu_max) {
                dNdlnu_max = dNdlnu;
              }
            }

            for (int n = 0; n <= nu_bins; n++) {
              v(idNdlnu + sidx + n * NumRadiationTypes, k, j, i) /= dNdlnu_max;
            }

            // Trapezoidal rule
            Real nu0 = nusamp[0];
            Real nu1 = nusamp[nu_bins];
            dN -= 0.5 * d_opacity.EmissivityPerNu(rho_cgs, T_cgs, ye, s, nu0) *
                  nu0 / (pc::h * nu0 * GetWeight(wgtC, nu0)) * dlnu;
            dN -= 0.5 * d_opacity.EmissivityPerNu(rho_cgs, T_cgs, ye, s, nu1) *
                  nu1 / (pc::h * nu1 * GetWeight(wgtC, nu1)) * dlnu;
            dN *= d3x_cgs * detG * dt * TIME;

            v(idNdlnu_max + sidx, k, j, i) = dNdlnu_max;

            int Ns = static_cast<int>(dN);
            if (dN - Ns > rng_gen.drand()) {
              Ns++;
            }

            // TODO(BRR) Use a ParArrayND<int> instead of these weird static_casts
            v(idN + sidx, k, j, i) = dN;
            v(iNs + sidx, k, j, i) = static_cast<Real>(Ns);
            rng_pool.free_state(rng_gen);
          });
    }
  }

  // Reduce dN over zones for calibrating weights (requires w ~ wgtC)
  Real dNtot_local = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MonteCarloReduceParticleCreation",
      DevExecSpace(), 0, 2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int sidx, const int k, const int j, const int i,
                    Real &dN) {
        if (do_species[sidx]) {
          dN += v(idN + sidx, k, j, i);
        }
      },
      Kokkos::Sum<Real>(dNtot_local));

  printf("[%i] dNtot: %e + %e\n", Globals::my_rank, *dNtot, dNtot_local);

  *dNtot += dNtot_local;

  return TaskStatus::complete;
}*/

TaskStatus MonteCarloCountCommunicatedParticles(MeshBlock *pmb, int *particles_outstanding) {
  auto &swarm = pmb->swarm_data.Get()->Get("monte_carlo");

  *particles_outstanding += swarm->num_particles_sent_;

  // Reset communication flags
  // TODO(BRR) nneighbor instead of nbmax?
  //for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    auto &nb = pmb->pbval->neighbor[n];
    swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    swarm->vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
#endif // MPI_PARALLEL
  }

  #ifdef MPI_PARALLEL
  pmb->exec_space.fence();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    auto &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      PARTHENON_MPI_CHECK(MPI_Wait(&(swarm->vbswarm->bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE));
    }
    swarm->vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
    //printf("[%i] req_send[%i] = MPI_REQUEST_NULL (%i)\n", Globals::my_rank, nb.bufid,
    //  swarm->vbswarm->bd_var_.req_send[nb.bufid]);
  }

  #endif

  return TaskStatus::complete;
}

TaskStatus InitializeCommunicationMesh(const std::string swarmName,
                                       const BlockList_t &blocks) {
  // Boundary transfers on same MPI proc are blocking
#ifdef MPI_PARALLEL
  for (auto &block : blocks) {
    auto swarm = block->swarm_data.Get()->Get(swarmName);
    for (int n = 0; n < block->pbval->nneighbor; n++) {
      auto &nb = block->pbval->neighbor[n];
      #ifdef MPI_PARALLEL
        swarm->vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
      #endif
    }
  }
#endif // MPI_PARALLEL

  /*for (auto &block : blocks) {
    auto &pmb = block;
    auto sc = pmb->swarm_data.Get();
    auto swarm = sc->Get(swarmName);

    for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
      auto &nb = pmb->pbval->neighbor[n];
      swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    }
  }*/

  // Reset boundary statuses
  for (auto &block : blocks) {
    auto &pmb = block;
    auto sc = pmb->swarm_data.Get();
    auto swarm = sc->Get(swarmName);
    for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
      auto &nb = pmb->pbval->neighbor[n];
      swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
