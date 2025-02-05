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

#include <coordinates/coordinates.hpp>
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "closure.hpp"
#include "closure_mocmc.hpp"
#include "geodesics.hpp"
#include "kd_grid.hpp"
#include "phoebus_utils/phoebus_interpolation.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/root_find.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

namespace radiation {

namespace pf = fluid_prim;
namespace cf = fluid_cons;
namespace cr = radmoment_cons;
namespace pr = radmoment_prim;
namespace ir = radmoment_internal;
namespace im = mocmc_internal;
using Microphysics::Opacities;
using Microphysics::RadiationType;
using Microphysics::EOS::EOS;
using vpack_types::FlatIdx;

constexpr int MAX_SPECIES = 3;

// TODO(BRR) add options
KOKKOS_FORCEINLINE_FUNCTION int
get_nsamp_per_zone(const int &k, const int &j, const int &i,
                   const Geometry::CoordSysMeshBlock &geom, const Real &rho,
                   const Real &T, const Real &Ye, const Real &J,
                   const int &nsamp_per_zone_global) {

  return nsamp_per_zone_global;
}

template <class T>
void MOCMCInitSamples(T *rc) {

  auto *pmb = rc->GetParentPointer();
  auto &sc = rc->GetSwarmData();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  auto rng_pool = rad->Param<RNGPool>("rng_pool");

  // Meshblock geometry
  const auto geom = Geometry::GetCoordinateSystem(rc);
  const parthenon::Coordinates_t &coords = pmb->coords;
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = coords.Dxf<1>(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = coords.Dxf<2>(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = coords.Dxf<3>(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = coords.Xf<1>(ib.s);
  const Real &minx_j = coords.Xf<2>(jb.s);
  const Real &minx_k = coords.Xf<3>(kb.s);

  // Microphysics
  auto opac_pkg = pmb->packages.Get("opacity");
  const auto opac = opac_pkg->template Param<Opacities>("opacities");
  StateDescriptor *eos = pmb->packages.Get("eos").get();

  std::vector<std::string> variables{
      pr::J::name(),           pr::H::name(),  pf::density::name(), pf::velocity::name(),
      pf::temperature::name(), pf::ye::name(), im::dnsamp::name()};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto pJ = imap.GetFlatIdx(pr::J::name());
  auto pH = imap.GetFlatIdx(pr::H::name());
  auto pdens = imap[pf::density::name()].first;
  auto pv = imap.GetFlatIdx(fluid_prim::velocity::name());
  auto pT = imap[pf::temperature::name()].first;
  auto pye = imap[pf::ye::name()].first;
  auto dn = imap[im::dnsamp::name()].first;

  const int nblock = v.GetDim(5);
  PARTHENON_REQUIRE_THROWS(nblock == 1, "Packing not currently supported for swarms");

  const int nsamp_per_zone = rad->Param<int>("nsamp_per_zone");
  int nsamp_tot = 0;

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  const auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  // Fill nsamp per zone per species and sum over zones
  // TODO(BRR) make this a separate function and use it to update dnsamp which is then
  // used to decide whether to refine/derefine
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MOCMC::Init::NumSamples", DevExecSpace(), 0,
      nblock - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nsamp) {
        Real Jtot = 0.;
        for (int s = 0; s < num_species; s++) {
          Jtot += v(b, pJ(s), k, j, i);
        }
        v(b, dn, k, j, i) =
            get_nsamp_per_zone(k, j, i, geom, v(b, pdens, k, j, i), v(b, pT, k, j, i),
                               v(b, pye, k, j, i), Jtot, nsamp_per_zone);
        nsamp += v(b, dn, k, j, i);
      },
      Kokkos::Sum<int>(nsamp_tot));

  auto new_particles_context = swarm->AddEmptyParticles(nsamp_tot);

  // Calculate array of starting index for each zone to compute particles
  ParArrayND<int> starting_index("Starting index", nx_k, nx_j, nx_i);
  auto starting_index_h = starting_index.GetHostMirror();
  auto dN = rc->Get(im::dnsamp::name()).data;
  auto dN_h = dN.GetHostMirrorAndCopy();
  int index = 0;
  for (int k = 0; k < nx_k; k++) {
    for (int j = 0; j < nx_j; j++) {
      for (int i = 0; i < nx_i; i++) {
        starting_index_h(k, j, i) = index;
        index += static_cast<int>(dN_h(k + kb.s, j + jb.s, i + ib.s));
      }
    }
  }
  starting_index.DeepCopy(starting_index_h);

  // TODO(BRR) Implement par_scan in parthenon then switch to this
  // int result;
  // Kokkos::parallel_scan("MOCMC::Starting indices", nx_k*nx_j*nx_i,
  //  KOKKOS_LAMBDA(const int idx, int &partial_sum, bool is_final){
  //    const int i = idx / nx_k*nx_j;
  //    const int j = (idx - i * nx_k*nx_j) / nx_j;
  //    const int k = idx - i * nx_k*nx_j - j * nx_j;
  //    if (is_final) {
  //    starting_index(k, j, i) = partial_sum;
  //    }
  //    partial_sum += static_cast<int>(dN_h(k + kb.s, j + jb.s, i + ib.s));
  //    },
  //  result);

  const auto &x = swarm->template Get<Real>(swarm_position::x::name()).Get();
  const auto &y = swarm->template Get<Real>(swarm_position::y::name()).Get();
  const auto &z = swarm->template Get<Real>(swarm_position::z::name()).Get();
  const auto &ncov = swarm->template Get<Real>(mocmc_core::ncov::name()).Get();
  const auto &Inuinv = swarm->template Get<Real>(mocmc_core::Inuinv::name()).Get();

  auto swarm_d = swarm->GetDeviceContext();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::Init::Sample", DevExecSpace(), 0, nblock - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int start_idx = starting_index(k - kb.s, j - jb.s, i - ib.s);
        auto rng_gen = rng_pool.get_state();

        Real cov_g[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, cov_g);
        Real ucon[4];
        const Real vpcon[] = {v(pv(0), k, j, i), v(pv(1), k, j, i), v(pv(2), k, j, i)};
        GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);
        const Real trial[4] = {0., 1., 0., 0.};
        Geometry::Tetrads tetrads(ucon, trial, cov_g);

        for (int nsamp = 0; nsamp < static_cast<int>(v(b, dn, k, j, i)); nsamp++) {
          const int n = new_particles_context.GetNewParticleIndex(start_idx + nsamp);

          // Create particles at zone centers
          x(n) = minx_i + (i - ib.s + rng_gen.drand()) * dx_i;
          y(n) = minx_j + (j - jb.s + rng_gen.drand()) * dx_j;
          z(n) = minx_k + (k - kb.s + rng_gen.drand()) * dx_k;

          const Real Temp = v(b, pT, k, j, i);

          // Sample uniformly in solid angle
          const Real theta = acos(2. * rng_gen.drand() - 1.);
          const Real phi = 2. * M_PI * rng_gen.drand();
          const Real ncov_tetrad[4] = {-1., cos(theta), cos(phi) * sin(theta),
                                       sin(phi) * sin(theta)};

          Real ncov_coord[4];
          tetrads.TetradToCoordCov(ncov_tetrad, ncov_coord);

          SPACETIMELOOP(mu) { ncov(mu, n) = ncov_coord[mu]; }

          Real ndu = 0.;
          SPACETIMELOOP(nu) { ndu -= ncov(nu, n) * ucon[nu]; }

          for (int s = 0; s < num_species; s++) {
            // Get radiation temperature
            const RadiationType type = species_d[s];
            const Real Tr = opac.TemperatureFromEnergyDensity(v(pJ(s), k, j, i), type);
            for (int nubin = 0; nubin < nu_bins; nubin++) {
              const Real nu = nusamp(nubin) * ndu;

              Inuinv(nubin, s, n) = std::max<Real>(
                  robust::SMALL(),
                  opac.ThermalDistributionOfTNu(Temp, type, nu) / pow(nu, 3));
            }
          }
        }

        rng_pool.free_state(rng_gen);
      });

  // Initialize eddington tensor and opacities for first step
  MOCMCReconstruction(rc, rc);
  MOCMCEddington(rc, rc);
  MOCMCFluidSource(rc, rc, 0., false); // Update opacities for asymptotic fluxes
}

template <class T>
TaskStatus MOCMCSampleBoundaries(T *rc_base, T *rc) {
  auto *pmb = rc->GetParentPointer();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  const auto geom = Geometry::GetCoordinateSystem(rc);
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  std::vector<std::string> variables{pr::J::name(), pf::velocity::name(),
                                     ir::tilPi::name()};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  // Microphysics
  auto opac_pkg = pmb->packages.Get("opacity");
  const auto opac = opac_pkg->template Param<Opacities>("opacities");

  static constexpr auto swarm_name = "mocmc";
  static const auto desc_mocmc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              mocmc_core::ncov, mocmc_core::Inuinv, mocmc_core::mu_lo,
                              mocmc_core::mu_hi, mocmc_core::phi_lo, mocmc_core::phi_hi>(
          swarm_name);
  auto pack_mocmc = desc_mocmc.GetPack(rc_base);

  auto pv = imap.GetFlatIdx(pf::velocity::name());
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name());
  auto iJ = imap.GetFlatIdx(pr::J::name());

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[MAX_SPECIES] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  // TODO(BRR) provide *all* MOCMC boundaries

  auto ix1_bc = rad->Param<MOCMCBoundaries>("ix1_bc");
  // auto ox1_bc = rad->Param<MOCMCBoundaries>("ox1_bc");

  Real ix1_temp = 0.;
  // Real ox1_temp = 0.;
  if (ix1_bc == MOCMCBoundaries::fixed_temp) {
    ix1_temp = rad->Param<Real>("ix1_temp");
  }
  // if (ox1_bc == MOCMCBoundaries::fixed_temp) {
  //  ox1_temp = rad->Param<Real>("ox1_temp");
  //}

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::SampleBoundaries", DevExecSpace(), 0,
      pack_mocmc.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        const auto [b, n] = pack_mocmc.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_mocmc.GetContext(b);
        if (swarm_d.IsActive(n)) {
          Real &x = pack_mocmc(b, swarm_position::x(), n);
          Real &y = pack_mocmc(b, swarm_position::y(), n);
          Real &z = pack_mocmc(b, swarm_position::z(), n);

          // Store zone before reflections
          int i, j, k;
          swarm_d.Xtoijk(x, y, z, i, j, k);

          if (x < swarm_d.x_min_global_) {
            // Reflect particle across boundary
            x = swarm_d.x_min_global_ + (swarm_d.x_min_global_ - x);
            // TODO(BRR) normalized in relativity?
            pack_mocmc(b, mocmc_core::ncov(1), n) =
                -pack_mocmc(b, mocmc_core::ncov(1), n);

            for (int s = 0; s < num_species; s++) {
              Real temp = 0.;
              if (ix1_bc == MOCMCBoundaries::outflow) {
                // Temperature from J in ghost zone
                temp =
                    opac.TemperatureFromEnergyDensity(v(b, iJ(s), k, j, i), species_d[s]);
              } else {
                // Fixed temperature
                temp = ix1_temp;
              }

              // Reset intensities
              for (int nubin = 0; nubin < nu_bins; nubin++) {
                const Real nu = nusamp(nubin);
                pack_mocmc(b, mocmc_core::Inuinv(nubin, s), n) =
                    std::max<Real>(robust::SMALL(), opac.ThermalDistributionOfTNu(
                                                        temp, species_d[s], nu)) /
                    std::pow(nu, 3);
              }
            }
          }

          if (x > swarm_d.x_max_global_) {
            // Reflect particle across boundary
            x = swarm_d.x_max_global_ - (x - swarm_d.x_max_global_);
            pack_mocmc(b, mocmc_core::ncov(1), n) =
                -pack_mocmc(b, mocmc_core::ncov(1), n);

            // Reset intensities
            for (int nubin = 0; nubin < nu_bins; nubin++) {
              for (int s = 0; s < num_species; s++) {
                pack_mocmc(b, mocmc_core::Inuinv(nubin, s), n) = robust::SMALL();
              }
            }
          }

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCReconstruction(T *rc_base, T *rc) {
  auto *pmb = rc->GetParentPointer();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  Real num_total = 0;
  // Sort and accumulate total active particles
  // Sorting must be done before forming packs.
  for (int b = 0; b <= rc->NumBlocks() - 1; b++) {
    rc_base->GetSwarmData(b)->Get("mocmc")->SortParticlesByCell();
    num_total += rc_base->GetSwarmData(b)->Get("mocmc")->GetNumActive();
  }
  rad->UpdateParam<Real>("num_total", num_total);

  const auto geom = Geometry::GetCoordinateSystem(rc);
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  std::vector<std::string> variables{pf::velocity::name(), ir::tilPi::name()};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  static constexpr auto swarm_name = "mocmc";
  static const auto desc_mocmc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              mocmc_core::ncov, mocmc_core::Inuinv, mocmc_core::mu_lo,
                              mocmc_core::mu_hi, mocmc_core::phi_lo, mocmc_core::phi_hi>(
          swarm_name);
  auto pack_mocmc = desc_mocmc.GetPack(rc_base);
  auto pv = imap.GetFlatIdx(pf::velocity::name());
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name());

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");

  auto mocmc_recon = rad->Param<MOCMCRecon>("mocmc_recon");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(), 0, rc->NumBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto swarm_d = pack_mocmc.GetContext(b);
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          if (n == 0) {
            pack_mocmc(b, mocmc_core::mu_lo(), nswarm) = 0.0;
            pack_mocmc(b, mocmc_core::mu_hi(), nswarm) = 2.0;
            pack_mocmc(b, mocmc_core::phi_lo(), nswarm) = 0.0;
            pack_mocmc(b, mocmc_core::phi_hi(), nswarm) = 2.0 * M_PI;
            continue;
          }

          Real cov_g[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, cov_g);
          const Real vpcon[] = {v(b, pv(0), k, j, i), v(b, pv(1), k, j, i),
                                v(b, pv(2), k, j, i)};
          const Real trial[4] = {0., 1., 0., 0.};
          Real ucon[4];
          GetFourVelocity(vpcon, geom, CellLocation::Cent, b, k, j, i, ucon);
          Geometry::Tetrads tetrads(ucon, trial, cov_g);

          Real ncov_coord[4] = {pack_mocmc(b, mocmc_core::ncov(0), nswarm),
                                pack_mocmc(b, mocmc_core::ncov(1), nswarm),
                                pack_mocmc(b, mocmc_core::ncov(2), nswarm),
                                pack_mocmc(b, mocmc_core::ncov(3), nswarm)};
          Real ncov_tetrad[4];
          tetrads.CoordToTetradCov(ncov_coord, ncov_tetrad);

          const Real mu = 1.0 - ncov_tetrad[1];
          const Real phi = atan2(ncov_tetrad[3], ncov_tetrad[2]) + M_PI;

          for (int m = 0; m < n; m++) {
            const int mswarm = swarm_d.GetFullIndex(k, j, i, m);
            PARTHENON_DEBUG_REQUIRE(mswarm != nswarm, "Comparing the same particle!");
            if (mu > pack_mocmc(b, mocmc_core::mu_lo(), mswarm) &&
                mu < pack_mocmc(b, mocmc_core::mu_hi(), mswarm) &&
                phi > pack_mocmc(b, mocmc_core::phi_lo(), mswarm) &&
                phi < pack_mocmc(b, mocmc_core::phi_hi(), mswarm)) {
              Real mcov_tetrad[4] = {-1., pack_mocmc(b, mocmc_core::ncov(1), mswarm),
                                     pack_mocmc(b, mocmc_core::ncov(2), mswarm),
                                     pack_mocmc(b, mocmc_core::ncov(3), mswarm)};
              Real mu0 = pack_mocmc(b, mocmc_core::mu_hi(), mswarm) -
                         pack_mocmc(b, mocmc_core::mu_lo(), mswarm);
              Real phi0 = pack_mocmc(b, mocmc_core::phi_hi(), mswarm) -
                          pack_mocmc(b, mocmc_core::phi_lo(), mswarm);
              if (mu0 > phi0) {
                const Real mu_m = 1.0 - ncov_tetrad[1];
                mu0 = 0.5 * (mu + mu_m);
                if (mu < mu0) {
                  pack_mocmc(b, mocmc_core::mu_lo(), nswarm) =
                      pack_mocmc(b, mocmc_core::mu_lo(), mswarm);
                  pack_mocmc(b, mocmc_core::mu_hi(), nswarm) = mu0;
                  pack_mocmc(b, mocmc_core::mu_lo(), mswarm) = mu0;
                } else {
                  pack_mocmc(b, mocmc_core::mu_lo(), nswarm) = mu0;
                  pack_mocmc(b, mocmc_core::mu_hi(), nswarm) =
                      pack_mocmc(b, mocmc_core::mu_hi(), mswarm);
                  pack_mocmc(b, mocmc_core::mu_hi(), mswarm) = mu0;
                }
                pack_mocmc(b, mocmc_core::phi_lo(), nswarm) =
                    pack_mocmc(b, mocmc_core::phi_lo(), mswarm);
                pack_mocmc(b, mocmc_core::phi_hi(), nswarm) =
                    pack_mocmc(b, mocmc_core::phi_hi(), mswarm);
              } else {
                const Real phi_m = atan2(ncov_tetrad[3], ncov_tetrad[2]);
                phi0 = 0.5 * (phi + phi_m);
                if (phi < phi0) {
                  pack_mocmc(b, mocmc_core::phi_lo(), nswarm) =
                      pack_mocmc(b, mocmc_core::phi_lo(), mswarm);
                  pack_mocmc(b, mocmc_core::phi_hi(), nswarm) = phi0;
                  pack_mocmc(b, mocmc_core::phi_lo(), mswarm) = phi0;
                } else {
                  pack_mocmc(b, mocmc_core::phi_lo(), nswarm) = phi0;
                  pack_mocmc(b, mocmc_core::phi_hi(), nswarm) =
                      pack_mocmc(b, mocmc_core::phi_hi(), mswarm);
                  pack_mocmc(b, mocmc_core::phi_hi(), mswarm) = phi0;
                }
                pack_mocmc(b, mocmc_core::mu_lo(), nswarm) =
                    pack_mocmc(b, mocmc_core::mu_lo(), mswarm);
                pack_mocmc(b, mocmc_core::mu_hi(), nswarm) =
                    pack_mocmc(b, mocmc_core::mu_hi(), mswarm);
              }
              break;
            } // if inside
          } // m = 0..n
        } // n = 0..nsamp
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCTransport(T *rc, const Real dt) {
  auto *pmb = rc->GetParentPointer();

  auto geom = Geometry::GetCoordinateSystem(rc);
  static constexpr auto swarm_name = "mocmc";
  static const auto desc_mocmc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              mocmc_core::ncov, mocmc_core::t>(swarm_name);
  auto pack_mocmc = desc_mocmc.GetPack(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::Transport", DevExecSpace(), 0,
      pack_mocmc.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        const auto [b, n] = pack_mocmc.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_mocmc.GetContext(b);
        if (swarm_d.IsActive(n)) {
          Real &t = pack_mocmc(b, mocmc_core::t(), n);
          Real &x = pack_mocmc(b, swarm_position::x(), n);
          Real &y = pack_mocmc(b, swarm_position::y(), n);
          Real &z = pack_mocmc(b, swarm_position::z(), n);
          Real y0 = y;
          Real z0 = z;
          PushParticle(t, x, y, z, pack_mocmc(b, mocmc_core::ncov(0), n),
                       pack_mocmc(b, mocmc_core::ncov(1), n),
                       pack_mocmc(b, mocmc_core::ncov(2), n),
                       pack_mocmc(b, mocmc_core::ncov(3), n), dt, geom);

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCFluidSource(T *rc_base, T *rc, const Real dt, const bool update_fluid) {
  // Assume particles are already sorted from MOCMCReconstruction call!

  auto *pmb = rc->GetParentPointer();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  // Meshblock geometry
  const auto geom = Geometry::GetCoordinateSystem(rc);
  const IndexRange &ib = rc->GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = rc->GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = rc->GetBoundsK(IndexDomain::interior);

  // Microphysics
  auto opac = pmb->packages.Get("opacity");
  const auto opac_d = opac->template Param<Opacities>("opacities");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  const auto eos_d = eos->template Param<EOS>("d.EOS");

  std::vector<std::string> variables{
      cr::E::name(),        cr::F::name(),           pr::J::name(),
      pr::H::name(),        pf::density::name(),     pf::energy::name(),
      pf::velocity::name(), pf::temperature::name(), pf::ye::name(),
      ir::tilPi::name(),    ir::kappaH::name(),      im::dnsamp::name(),
      im::Inu0::name(),     im::Inu1::name(),        im::jinvs::name()};
  if (update_fluid) {
    variables.push_back(cf::energy::name());
    variables.push_back(cf::momentum::name());
    variables.push_back(cf::ye::name());
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto pJ = imap.GetFlatIdx(pr::J::name());
  const auto pH = imap.GetFlatIdx(pr::H::name());
  const auto pdens = imap[pf::density::name()].first;
  const auto peng = imap[pf::energy::name()].first;
  const auto pv = imap.GetFlatIdx(fluid_prim::velocity::name());
  const auto pT = imap[pf::temperature::name()].first;
  const auto pye = imap[pf::ye::name()].first;
  const auto Inu0 = imap.GetFlatIdx(im::Inu0::name());
  const auto Inu1 = imap.GetFlatIdx(im::Inu1::name());
  const auto ijinvs = imap.GetFlatIdx(im::jinvs::name());
  const auto iTilPi = imap.GetFlatIdx(ir::tilPi::name());
  auto idx_E = imap.GetFlatIdx(cr::E::name());
  auto idx_F = imap.GetFlatIdx(cr::F::name());
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH::name());
  int ceng(-1), cmom_lo(-1), cye(-1);
  if (update_fluid) {
    ceng = imap[cf::energy::name()].first;
    cmom_lo = imap[cf::momentum::name()].first;
    cye = imap[cf::ye::name()].first;
  }

  static constexpr auto swarm_name = "mocmc";
  static const auto desc_mocmc =
      MakeSwarmPackDescriptor<mocmc_core::ncov, mocmc_core::mu_lo, mocmc_core::mu_hi,
                              mocmc_core::phi_lo, mocmc_core::phi_hi, mocmc_core::Inuinv>(
          swarm_name);
  auto pack_mocmc = desc_mocmc.GetPack(rc_base);

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  const Real dlnu = rad->Param<Real>("dlnu");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  int nspec = idx_E.DimSize(1);

  if (true) { // update = lagged
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "MOCMC::FluidSource", DevExecSpace(), 0,
        rc->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto swarm_d = pack_mocmc.GetContext(b);
          Real cov_g[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, cov_g);
          Real alpha = geom.Lapse(CellLocation::Cent, b, k, j, i);
          Real con_beta[3];
          geom.ContravariantShift(CellLocation::Cent, b, k, j, i, con_beta);
          const Real vpcon[] = {v(b, pv(0), k, j, i), v(b, pv(1), k, j, i),
                                v(b, pv(2), k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vpcon, cov_g);
          const Real ucon[4] = {W / alpha, vpcon[0] - con_beta[0] * W / alpha,
                                vpcon[1] - con_beta[1] * W / alpha,
                                vpcon[2] - con_beta[2] * W / alpha};

          for (int ispec = 0; ispec < num_species; ++ispec) {
            const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

            // Set up the background state
            Vec con_v{{v(b, pv(0), k, j, i), v(b, pv(1), k, j, i), v(b, pv(2), k, j, i)}};
            Tens2 cov_gamma;
            geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
            Real alpha = geom.Lapse(CellLocation::Cent, b, k, j, i);
            Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);
            LocalThreeGeometry g(geom, CellLocation::Cent, b, k, j, i);
            Real Estar = v(b, idx_E(ispec), k, j, i) / sdetgam;
            Vec cov_Fstar{v(b, idx_F(ispec, 0), k, j, i) / sdetgam,
                          v(b, idx_F(ispec, 1), k, j, i) / sdetgam,
                          v(b, idx_F(ispec, 2), k, j, i) / sdetgam};
            Tens2 con_tilPi;
            SPACELOOP2(ii, jj) {
              con_tilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i);
            }
            Real JBB =
                opac_d.EnergyDensityFromTemperature(v(b, pT, k, j, i), species_d[ispec]);

            ClosureMOCMC<> c(con_v, &g);

            Real dE = 0;
            Vec cov_dF;

            // Calculate Inu0
            for (int bin = 0; bin < nu_bins; bin++) {
              v(b, Inu0(ispec, bin), k, j, i) = 0.;
            }
            for (int n = 0; n < nsamp; n++) {
              const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
              const Real dOmega = (pack_mocmc(b, mocmc_core::mu_hi(), nswarm) -
                                   pack_mocmc(b, mocmc_core::mu_lo(), nswarm)) *
                                  (pack_mocmc(b, mocmc_core::phi_hi(), nswarm) -
                                   pack_mocmc(b, mocmc_core::phi_lo(), nswarm)) /
                                  (4. * M_PI);
              for (int bin = 0; bin < nu_bins; bin++) {
                v(b, Inu0(ispec, bin), k, j, i) +=
                    pack_mocmc(b, mocmc_core::Inuinv(bin, ispec), nswarm) *
                    pow(nusamp(bin), 3) * dOmega;
              }
            }

            // Frequency-average opacities
            Real kappaJ = 0.;
            Real kappaH = 0.;

            Real Itot = 0.;
            for (int bin = 0; bin < nu_bins; bin++) {
              kappaJ += opac_d.AngleAveragedAbsorptionCoefficient(
                            v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                            species_d[ispec], nusamp(bin)) *
                        v(b, Inu0(ispec, bin), k, j, i) * nusamp(bin);
              kappaH += opac_d.TotalScatteringCoefficient(
                            v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                            species_d[ispec], nusamp(bin)) *
                        v(b, Inu0(ispec, bin), k, j, i) * nusamp(bin);
              Itot += v(b, Inu0(ispec, bin), k, j, i) * nusamp(bin);
            }

            // Trapezoidal rule
            kappaJ -= 0.5 *
                      opac_d.AngleAveragedAbsorptionCoefficient(
                          v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                          species_d[ispec], nusamp(0)) *
                      v(b, Inu0(ispec, 0), k, j, i) * nusamp(0);
            kappaJ -= 0.5 *
                      opac_d.AngleAveragedAbsorptionCoefficient(
                          v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                          species_d[ispec], nusamp(nu_bins - 1)) *
                      v(b, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1);
            kappaH -= 0.5 *
                      opac_d.TotalScatteringCoefficient(
                          v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                          species_d[ispec], nusamp(0)) *
                      v(b, Inu0(ispec, 0), k, j, i) * nusamp(0);
            kappaH -= 0.5 *
                      opac_d.TotalScatteringCoefficient(
                          v(b, pdens, k, j, i), v(b, pT, k, j, i), v(b, pye, k, j, i),
                          species_d[ispec], nusamp(nu_bins - 1)) *
                      v(b, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1);
            Itot -= 0.5 * (v(b, Inu0(ispec, 0), k, j, i) * nusamp(0) +
                           v(b, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1));
            kappaJ = robust::ratio(kappaJ, Itot);
            kappaH = robust::ratio(kappaH, Itot) + kappaJ;

            Real tauJ = alpha * dt * kappaJ;
            Real tauH = alpha * dt * kappaH;

            // Store kappaH for asymptotic fluxes
            v(b, idx_kappaH(ispec), k, j, i) = kappaH;

            auto status = c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ,
                                               tauH, &dE, &cov_dF);

            // Add source corrections to conserved radiation variables
            v(b, idx_E(ispec), k, j, i) += sdetgam * dE;
            for (int idir = 0; idir < 3; ++idir) {
              v(b, idx_F(ispec, idir), k, j, i) += sdetgam * cov_dF(idir);
            }

            // Add source corrections to conserved fluid variables
            if (update_fluid) {
              v(b, cye, k, j, i) -= sdetgam * 0.0;
              v(b, ceng, k, j, i) -= sdetgam * dE;
              v(b, cmom_lo + 0, k, j, i) -= sdetgam * cov_dF(0);
              v(b, cmom_lo + 1, k, j, i) -= sdetgam * cov_dF(1);
              v(b, cmom_lo + 2, k, j, i) -= sdetgam * cov_dF(2);
            }

            // Update sample intensities
            for (int n = 0; n < nsamp; n++) {
              const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

              Real nu_fluid0 = nusamp(0);
              Real nu_lab0 = 0.;
              SPACETIMELOOP(nu) {
                nu_lab0 -= pack_mocmc(b, mocmc_core::ncov(nu), nswarm) * ucon[nu];
              }
              nu_lab0 *= nusamp(0);

              const Real shift = (log(nusamp(0)) - log(nu_lab0)) / dlnu;
              interpolation::PiecewiseConstant interp(nu_bins, dlnu, -shift);
              int nubin_shift[interp.maxStencilSize];
              Real nubin_wgt[interp.maxStencilSize];

              // Calculate effective scattering emissivity from angle-averaged intensity
              // TODO(BRR) include dI/ds in this calculation for inelastic scattering
              for (int bin = 0; bin < nu_bins; bin++) {
                const Real nu_fluid = nusamp(bin);
                const Real alphainv_s =
                    nu_fluid * opac_d.TotalScatteringCoefficient(
                                   v(b, pdens, k, j, i), v(b, pT, k, j, i),
                                   v(b, pye, k, j, i), species_d[ispec], nu_fluid);
                v(b, ijinvs(ispec, bin), k, j, i) = v(b, Inu0(ispec, bin), k, j, i) /
                                                    (nu_fluid * nu_fluid * nu_fluid) *
                                                    alphainv_s;
              }

              for (int bin = 0; bin < nu_bins; bin++) {
                const Real nu_fluid = nusamp(bin);
                const Real nu_lab = std::exp(std::log(nu_fluid) - dlnu * shift);
                const Real ds = dt / ucon[0] / nu_fluid;
                const Real alphainv_a =
                    nu_fluid * opac_d.TotalScatteringCoefficient(
                                   v(b, pdens, k, j, i), v(b, pT, k, j, i),
                                   v(b, pye, k, j, i), species_d[ispec], nu_fluid);
                const Real alphainv_s =
                    nu_fluid * opac_d.TotalScatteringCoefficient(
                                   v(b, pdens, k, j, i), v(b, pT, k, j, i),
                                   v(b, pye, k, j, i), species_d[ispec], nu_fluid);
                const Real jinv_a = opac_d.ThermalDistributionOfTNu(
                                        v(b, pT, k, j, i), species_d[ispec], nu_fluid) /
                                    (nu_fluid * nu_fluid * nu_fluid) * alphainv_a;

                // Interpolate invariant scattering emissivity to lab frame
                Real jinv_s = 0.;
                interp.GetIndicesAndWeights(bin, nubin_shift, nubin_wgt);
                for (int isup = 0; isup < interp.StencilSize(); isup++) {
                  jinv_s +=
                      nubin_wgt[isup] * v(b, ijinvs(ispec, nubin_shift[isup]), k, j, i);
                }

                pack_mocmc(b, mocmc_core::Inuinv(bin, ispec), nswarm) =
                    (pack_mocmc(b, mocmc_core::Inuinv(bin, ispec), nswarm) +
                     ds * (jinv_a + jinv_s)) /
                    (1. + ds * (alphainv_a + alphainv_s));
              }
            }
          }
        });
  } else {
    // 2D solve for T, Ye
  } // else else: 5D solve for T, Ye, Momentum

  // Recalculate pi given updated sample intensities
  // TODO(BRR) make this a separate task?
  // MOCMCEddington(rc);

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCEddington(T *rc_base, T *rc) {
  // Assume list is sorted!
  namespace ir = radmoment_internal;

  auto *pmb = rc->GetParentPointer();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto geom = Geometry::GetCoordinateSystem(rc);

  std::vector<std::string> variables{ir::tilPi::name(), pf::velocity::name()};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  static constexpr auto swarm_name = "mocmc";
  static const auto desc_mocmc =
      MakeSwarmPackDescriptor<mocmc_core::ncov, mocmc_core::mu_lo, mocmc_core::mu_hi,
                              mocmc_core::phi_lo, mocmc_core::phi_hi, mocmc_core::Inuinv>(
          swarm_name);
  auto pack_mocmc = desc_mocmc.GetPack(rc_base);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi::name());
  const auto pvel_lo = imap[pf::velocity::name()].first;

  const int nu_bins = rad->Param<int>("nu_bins");
  const Real dlnu = rad->Param<Real>("dlnu");
  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  auto num_species = rad->Param<int>("num_species");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(), 0, rc->NumBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto swarm_d = pack_mocmc.GetContext(b);
        // initialize eddington to zero
        for (int s = 0; s < num_species; s++) {
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = ii; jj < 3; jj++) {
              v(b, iTilPi(s, ii, jj), k, j, i) = 0.0;
            }
          }
        }
        Real energy[MAX_SPECIES] = {0.0};
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

        Real cov_g[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, b, k, j, i, cov_g);
        Real alpha = geom.Lapse(CellLocation::Cent, b, k, j, i);
        Real con_beta[3];
        geom.ContravariantShift(CellLocation::Cent, b, k, j, i, con_beta);
        const Real vpcon[] = {v(b, pvel_lo, k, j, i), v(b, pvel_lo + 1, k, j, i),
                              v(b, pvel_lo + 2, k, j, i)};
        const Real W = phoebus::GetLorentzFactor(vpcon, cov_g);
        const Real ucon[4] = {W / alpha, vpcon[0] - con_beta[0] * W / alpha,
                              vpcon[1] - con_beta[1] * W / alpha,
                              vpcon[2] - con_beta[2] * W / alpha};

        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

          Real nu_fluid0 = nusamp(0);
          Real nu_lab0 = 0.;
          SPACETIMELOOP(nu) {
            nu_lab0 -= pack_mocmc(b, mocmc_core::ncov(nu), nswarm) * ucon[nu];
          }
          nu_lab0 *= nusamp(0);

          const Real shift = (log(nusamp(0)) - log(nu_lab0)) / dlnu;
          interpolation::PiecewiseConstant interp(nu_bins, dlnu, shift);
          int nubin_shift[interp.maxStencilSize];
          Real nubin_wgt[interp.maxStencilSize];

          // get the energy integrated intensity
          Real I[MAX_SPECIES] = {0.0};
          for (int nubin = 0; nubin < nu_bins; nubin++) {
            // First order interpolation
            for (int s = 0; s < num_species; s++) {
              interp.GetIndicesAndWeights(nubin, nubin_shift, nubin_wgt);
              for (int isup = 0; isup < interp.StencilSize(); isup++) {
                I[s] += nubin_wgt[isup] *
                        pack_mocmc(b, mocmc_core::Inuinv(nubin_shift[isup], s), nswarm) *
                        pow(nusamp(nubin), 4);
              }
            }
          }

          Real wgts[6];
          kdgrid::integrate_ninj_domega_quad(pack_mocmc(b, mocmc_core::mu_lo(), nswarm),
                                             pack_mocmc(b, mocmc_core::mu_hi(), nswarm),
                                             pack_mocmc(b, mocmc_core::phi_lo(), nswarm),
                                             pack_mocmc(b, mocmc_core::phi_hi(), nswarm),
                                             wgts);
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = ii; jj < 3; jj++) {
              const int ind = Geometry::Utils::Flatten2(ii, jj, 3);
              for (int s = 0; s < num_species; s++) {
                v(b, iTilPi(s, ii, jj), k, j, i) += wgts[ind] * I[s];
              }
            }
          }
          for (int s = 0; s < num_species; s++) {
            energy[s] += (pack_mocmc(b, mocmc_core::mu_hi(), nswarm) -
                          pack_mocmc(b, mocmc_core::mu_lo(), nswarm)) *
                         (pack_mocmc(b, mocmc_core::phi_hi(), nswarm) -
                          pack_mocmc(b, mocmc_core::phi_lo(), nswarm)) *
                         I[s];
          }
        }

        if (nsamp > 0) {
          for (int s = 0; s < num_species; s++) {
            for (int ii = 0; ii < 3; ii++) {
              for (int jj = ii; jj < 3; jj++) {
                v(b, iTilPi(s, ii, jj), k, j, i) /= energy[s];
              }
              v(b, iTilPi(s, ii, ii), k, j, i) -= 1. / 3.;
            }
          }
        }

        for (int s = 0; s < num_species; s++) {
          v(b, iTilPi(s, 1, 0), k, j, i) = v(b, iTilPi(s, 0, 1), k, j, i);
          v(b, iTilPi(s, 2, 0), k, j, i) = v(b, iTilPi(s, 0, 2), k, j, i);
          v(b, iTilPi(s, 2, 1), k, j, i) = v(b, iTilPi(s, 1, 2), k, j, i);
        }
      });

  return TaskStatus::complete;
}

// Reduce particle sampling resolution statistics from per-mesh to global as part of
// global reduction.
TaskStatus MOCMCUpdateParticleCount(Mesh *pmesh, std::vector<Real> *resolution) {
  auto rad = pmesh->packages.Get("radiation");
  const auto num_total = rad->Param<Real>("num_total");
  (*resolution)[0] += num_total;
  rad->UpdateParam<Real>("num_total", 0.);
  return TaskStatus::complete;
}

template TaskStatus MOCMCSampleBoundaries<MeshData<Real>>(MeshData<Real> *rc_base,
                                                          MeshData<Real> *rc);
// template TaskStatus MOCMCSampleBoundaries<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MOCMCReconstruction<MeshData<Real>>(MeshData<Real> *rc_base,
                                                        MeshData<Real> *rc);
// template TaskStatus MOCMCReconstruction<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MOCMCEddington<MeshData<Real>>(MeshData<Real> *rc_base,
                                                   MeshData<Real> *rc);
// template TaskStatus MOCMCEddington<MeshData<Real>>(MeshData<Real> *rc);
template void MOCMCInitSamples<MeshBlockData<Real>>(MeshBlockData<Real> *);
template TaskStatus MOCMCTransport<MeshData<Real>>(MeshData<Real> *rc, const Real dt);
// template TaskStatus MOCMCFluidSource<MeshData<Real>>(MeshData<Real> *rc, const Real dt,
//                                                     const bool update_fluid);
template TaskStatus MOCMCFluidSource<MeshData<Real>>(MeshData<Real> *rc_base,
                                                     MeshData<Real> *rc, const Real dt,
                                                     const bool update_fluid);

} // namespace radiation
