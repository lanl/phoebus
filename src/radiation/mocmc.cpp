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
  auto &sc = pmb->swarm_data.Get();
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
      pr::J,           pr::H,          pf::density::name(), pf::velocity::name(),
      pf::temperature, pf::ye::name(), im::dnsamp};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto pJ = imap.GetFlatIdx(pr::J);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pdens = imap[pf::density::name()].first;
  auto pv = imap.GetFlatIdx(fluid_prim::velocity::name());
  auto pT = imap[pf::temperature].first;
  auto pye = imap[pf::ye::name()].first;
  auto dn = imap[im::dnsamp].first;

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

  ParArrayND<int> new_indices;
  auto new_mask = swarm->AddEmptyParticles(nsamp_tot, new_indices);

  // Calculate array of starting index for each zone to compute particles
  ParArrayND<int> starting_index("Starting index", nx_k, nx_j, nx_i);
  auto starting_index_h = starting_index.GetHostMirror();
  auto dN = rc->Get(im::dnsamp).data;
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

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();

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
          const int n = new_indices(start_idx + nsamp);

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
  MOCMCReconstruction(rc);
  MOCMCEddington(rc);
  MOCMCFluidSource(rc, 0., false); // Update opacities for asymptotic fluxes
}

template <class T>
TaskStatus MOCMCSampleBoundaries(T *rc) {
  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  const auto geom = Geometry::GetCoordinateSystem(rc);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> variables{pr::J, pf::velocity::name(), ir::tilPi};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  // Microphysics
  auto opac_pkg = pmb->packages.Get("opacity");
  const auto opac = opac_pkg->template Param<Opacities>("opacities");

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  auto pv = imap.GetFlatIdx(pf::velocity::name());
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);
  auto iJ = imap.GetFlatIdx(pr::J);

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[MAX_SPECIES] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  auto swarm_d = swarm->GetDeviceContext();

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

  pmb->par_for(
      "Temporary MOCMC boundaries", 0, swarm->GetMaxActiveIndex(),
      KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {

          // Store zone before reflections
          int i, j, k;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          if (x(n) < swarm_d.x_min_global_) {
            // Reflect particle across boundary
            x(n) = swarm_d.x_min_global_ + (swarm_d.x_min_global_ - x(n));
            // TODO(BRR) normalized in relativity?
            ncov(1, n) = -ncov(1, n);

            for (int s = 0; s < num_species; s++) {
              Real temp = 0.;
              if (ix1_bc == MOCMCBoundaries::outflow) {
                // Temperature from J in ghost zone
                temp = opac.TemperatureFromEnergyDensity(v(iJ(s), k, j, i), species_d[s]);
              } else {
                // Fixed temperature
                temp = ix1_temp;
              }

              // Reset intensities
              for (int nubin = 0; nubin < nu_bins; nubin++) {
                const Real nu = nusamp(nubin);
                Inuinv(nubin, s, n) =
                    std::max<Real>(robust::SMALL(), opac.ThermalDistributionOfTNu(
                                                        temp, species_d[s], nu)) /
                    std::pow(nu, 3);
              }
            }
          }

          if (x(n) > swarm_d.x_max_global_) {
            // Reflect particle across boundary
            x(n) = swarm_d.x_max_global_ - (x(n) - swarm_d.x_max_global_);
            ncov(1, n) = -ncov(1, n);

            // Reset intensities
            for (int nubin = 0; nubin < nu_bins; nubin++) {
              for (int s = 0; s < num_species; s++) {
                Inuinv(nubin, s, n) = robust::SMALL();
              }
            }
          }

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCReconstruction(T *rc) {
  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  const auto geom = Geometry::GetCoordinateSystem(rc);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> variables{pf::velocity::name(), ir::tilPi};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  auto pv = imap.GetFlatIdx(pf::velocity::name());
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");

  swarm->SortParticlesByCell();
  auto swarm_d = swarm->GetDeviceContext();

  rad->UpdateParam<Real>("num_total", swarm->GetNumActive());

  auto mocmc_recon = rad->Param<MOCMCRecon>("mocmc_recon");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          if (n == 0) {
            mu_lo(nswarm) = 0.0;
            mu_hi(nswarm) = 2.0;
            phi_lo(nswarm) = 0.0;
            phi_hi(nswarm) = 2.0 * M_PI;
            continue;
          }

          Real cov_g[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, cov_g);
          const Real vpcon[] = {v(pv(0), k, j, i), v(pv(1), k, j, i), v(pv(2), k, j, i)};
          const Real trial[4] = {0., 1., 0., 0.};
          Real ucon[4];
          GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);
          Geometry::Tetrads tetrads(ucon, trial, cov_g);

          Real ncov_coord[4] = {ncov(0, nswarm), ncov(1, nswarm), ncov(2, nswarm),
                                ncov(3, nswarm)};
          Real ncov_tetrad[4];
          tetrads.CoordToTetradCov(ncov_coord, ncov_tetrad);

          const Real mu = 1.0 - ncov_tetrad[1];
          const Real phi = atan2(ncov_tetrad[3], ncov_tetrad[2]) + M_PI;

          for (int m = 0; m < n; m++) {
            const int mswarm = swarm_d.GetFullIndex(k, j, i, m);
            PARTHENON_DEBUG_REQUIRE(mswarm != nswarm, "Comparing the same particle!");
            if (mu > mu_lo(mswarm) && mu < mu_hi(mswarm) && phi > phi_lo(mswarm) &&
                phi < phi_hi(mswarm)) {
              Real mcov_tetrad[4] = {-1., ncov(1, mswarm), ncov(2, mswarm),
                                     ncov(3, mswarm)};
              Real mu0 = mu_hi(mswarm) - mu_lo(mswarm);
              Real phi0 = phi_hi(mswarm) - phi_lo(mswarm);
              if (mu0 > phi0) {
                const Real mu_m = 1.0 - ncov_tetrad[1];
                mu0 = 0.5 * (mu + mu_m);
                if (mu < mu0) {
                  mu_lo(nswarm) = mu_lo(mswarm);
                  mu_hi(nswarm) = mu0;
                  mu_lo(mswarm) = mu0;
                } else {
                  mu_lo(nswarm) = mu0;
                  mu_hi(nswarm) = mu_hi(mswarm);
                  mu_hi(mswarm) = mu0;
                }
                phi_lo(nswarm) = phi_lo(mswarm);
                phi_hi(nswarm) = phi_hi(mswarm);
              } else {
                const Real phi_m = atan2(ncov_tetrad[3], ncov_tetrad[2]);
                phi0 = 0.5 * (phi + phi_m);
                if (phi < phi0) {
                  phi_lo(nswarm) = phi_lo(mswarm);
                  phi_hi(nswarm) = phi0;
                  phi_lo(mswarm) = phi0;
                } else {
                  phi_lo(nswarm) = phi0;
                  phi_hi(nswarm) = phi_hi(mswarm);
                  phi_hi(mswarm) = phi0;
                }
                mu_lo(nswarm) = mu_lo(mswarm);
                mu_hi(nswarm) = mu_hi(mswarm);
              }
              break;
            } // if inside
          }   // m = 0..n
        }     // n = 0..nsamp
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCTransport(T *rc, const Real dt) {
  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");

  auto geom = Geometry::GetCoordinateSystem(rc);
  auto &t = swarm->template Get<Real>("t").Get();
  auto &x = swarm->template Get<Real>("x").Get();
  auto &y = swarm->template Get<Real>("y").Get();
  auto &z = swarm->template Get<Real>("z").Get();
  auto &ncov = swarm->template Get<Real>("ncov").Get();
  auto swarm_d = swarm->GetDeviceContext();

  pmb->par_for(
      "MOCMC::Transport", 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          Real y0 = y(n);
          Real z0 = z(n);
          PushParticle(t(n), x(n), y(n), z(n), ncov(0, n), ncov(1, n), ncov(2, n),
                       ncov(3, n), dt, geom);

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
}

// TODO(BRR): Hack to get around current lack of support for packing parthenon swarms
template <>
TaskStatus MOCMCFluidSource(MeshData<Real> *rc, const Real dt, const bool update_fluid) {
  for (int n = 0; n < rc->NumBlocks(); n++) {
    MOCMCFluidSource(rc->GetBlockData(n).get(), dt, update_fluid);
  }
  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCFluidSource(T *rc, const Real dt, const bool update_fluid) {
  // Assume particles are already sorted from MOCMCReconstruction call!

  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  // Meshblock geometry
  const auto geom = Geometry::GetCoordinateSystem(rc);
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Microphysics
  auto opac = pmb->packages.Get("opacity");
  const auto opac_d = opac->template Param<Opacities>("opacities");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  const auto eos_d = eos->template Param<EOS>("d.EOS");

  std::vector<std::string> variables{cr::E,
                                     cr::F,
                                     pr::J,
                                     pr::H,
                                     pf::density::name(),
                                     pf::energy::name(),
                                     pf::velocity::name(),
                                     pf::temperature,
                                     pf::ye::name(),
                                     ir::tilPi,
                                     ir::kappaH,
                                     im::dnsamp,
                                     im::Inu0,
                                     im::Inu1,
                                     im::jinvs};
  if (update_fluid) {
    variables.push_back(cf::energy::name());
    variables.push_back(cf::momentum::name());
    variables.push_back(cf::ye::name());
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto pJ = imap.GetFlatIdx(pr::J);
  const auto pH = imap.GetFlatIdx(pr::H);
  const auto pdens = imap[pf::density::name()].first;
  const auto peng = imap[pf::energy::name()].first;
  const auto pv = imap.GetFlatIdx(fluid_prim::velocity::name());
  const auto pT = imap[pf::temperature].first;
  const auto pye = imap[pf::ye::name()].first;
  const auto Inu0 = imap.GetFlatIdx(im::Inu0);
  const auto Inu1 = imap.GetFlatIdx(im::Inu1);
  const auto ijinvs = imap.GetFlatIdx(im::jinvs);
  const auto iTilPi = imap.GetFlatIdx(ir::tilPi);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  int ceng(-1), cmom_lo(-1), cye(-1);
  if (update_fluid) {
    ceng = imap[cf::energy::name()].first;
    cmom_lo = imap[cf::momentum::name()].first;
    cye = imap[cf::ye::name()].first;
  }

  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  const auto swarm_d = swarm->GetDeviceContext();

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  const Real dlnu = rad->Param<Real>("dlnu");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  const int iblock = 0; // No meshblockpacks right now
  int nspec = idx_E.DimSize(1);

  if (true) { // update = lagged
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "MOCMC::FluidSource", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real cov_g[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, iblock, k, j, i, cov_g);
          Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
          Real con_beta[3];
          geom.ContravariantShift(CellLocation::Cent, iblock, k, j, i, con_beta);
          const Real vpcon[] = {v(pv(0), k, j, i), v(pv(1), k, j, i), v(pv(2), k, j, i)};
          const Real W = phoebus::GetLorentzFactor(vpcon, cov_g);
          const Real ucon[4] = {W / alpha, vpcon[0] - con_beta[0] * W / alpha,
                                vpcon[1] - con_beta[1] * W / alpha,
                                vpcon[2] - con_beta[2] * W / alpha};

          for (int ispec = 0; ispec < num_species; ++ispec) {
            const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

            // Set up the background state
            Vec con_v{{v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                       v(iblock, pv(2), k, j, i)}};
            Tens2 cov_gamma;
            geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);
            Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
            Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
            LocalThreeGeometry g(geom, CellLocation::Cent, iblock, k, j, i);
            Real Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
            Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                          v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam};
            Tens2 con_tilPi;
            SPACELOOP2(ii, jj) {
              con_tilPi(ii, jj) = v(iblock, iTilPi(ispec, ii, jj), k, j, i);
            }
            Real JBB = opac_d.EnergyDensityFromTemperature(v(iblock, pT, k, j, i),
                                                           species_d[ispec]);

            ClosureMOCMC<> c(con_v, &g);

            Real dE = 0;
            Vec cov_dF;

            // Calculate Inu0
            for (int bin = 0; bin < nu_bins; bin++) {
              v(iblock, Inu0(ispec, bin), k, j, i) = 0.;
            }
            for (int n = 0; n < nsamp; n++) {
              const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
              const Real dOmega = (mu_hi(nswarm) - mu_lo(nswarm)) *
                                  (phi_hi(nswarm) - phi_lo(nswarm)) / (4. * M_PI);
              for (int bin = 0; bin < nu_bins; bin++) {
                v(iblock, Inu0(ispec, bin), k, j, i) +=
                    Inuinv(bin, ispec, nswarm) * pow(nusamp(bin), 3) * dOmega;
              }
            }

            // Frequency-average opacities
            Real kappaJ = 0.;
            Real kappaH = 0.;

            Real Itot = 0.;
            for (int bin = 0; bin < nu_bins; bin++) {
              kappaJ += opac_d.AngleAveragedAbsorptionCoefficient(
                            v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                            v(iblock, pye, k, j, i), species_d[ispec], nusamp(bin)) *
                        v(iblock, Inu0(ispec, bin), k, j, i) * nusamp(bin);
              kappaH += opac_d.TotalScatteringCoefficient(
                            v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                            v(iblock, pye, k, j, i), species_d[ispec], nusamp(bin)) *
                        v(iblock, Inu0(ispec, bin), k, j, i) * nusamp(bin);
              Itot += v(iblock, Inu0(ispec, bin), k, j, i) * nusamp(bin);
            }

            // Trapezoidal rule
            kappaJ -= 0.5 *
                      opac_d.AngleAveragedAbsorptionCoefficient(
                          v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                          v(iblock, pye, k, j, i), species_d[ispec], nusamp(0)) *
                      v(iblock, Inu0(ispec, 0), k, j, i) * nusamp(0);
            kappaJ -=
                0.5 *
                opac_d.AngleAveragedAbsorptionCoefficient(
                    v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                    v(iblock, pye, k, j, i), species_d[ispec], nusamp(nu_bins - 1)) *
                v(iblock, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1);
            kappaH -= 0.5 *
                      opac_d.TotalScatteringCoefficient(
                          v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                          v(iblock, pye, k, j, i), species_d[ispec], nusamp(0)) *
                      v(iblock, Inu0(ispec, 0), k, j, i) * nusamp(0);
            kappaH -=
                0.5 *
                opac_d.TotalScatteringCoefficient(
                    v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                    v(iblock, pye, k, j, i), species_d[ispec], nusamp(nu_bins - 1)) *
                v(iblock, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1);
            Itot -= 0.5 *
                    (v(iblock, Inu0(ispec, 0), k, j, i) * nusamp(0) +
                     v(iblock, Inu0(ispec, nu_bins - 1), k, j, i) * nusamp(nu_bins - 1));
            kappaJ = robust::ratio(kappaJ, Itot);
            kappaH = robust::ratio(kappaH, Itot) + kappaJ;

            Real tauJ = alpha * dt * kappaJ;
            Real tauH = alpha * dt * kappaH;

            // Store kappaH for asymptotic fluxes
            v(iblock, idx_kappaH(ispec), k, j, i) = kappaH;

            auto status = c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, JBB, tauJ,
                                               tauH, &dE, &cov_dF);

            // Add source corrections to conserved radiation variables
            v(iblock, idx_E(ispec), k, j, i) += sdetgam * dE;
            for (int idir = 0; idir < 3; ++idir) {
              v(iblock, idx_F(ispec, idir), k, j, i) += sdetgam * cov_dF(idir);
            }

            // Add source corrections to conserved fluid variables
            if (update_fluid) {
              v(iblock, cye, k, j, i) -= sdetgam * 0.0;
              v(iblock, ceng, k, j, i) -= sdetgam * dE;
              v(iblock, cmom_lo + 0, k, j, i) -= sdetgam * cov_dF(0);
              v(iblock, cmom_lo + 1, k, j, i) -= sdetgam * cov_dF(1);
              v(iblock, cmom_lo + 2, k, j, i) -= sdetgam * cov_dF(2);
            }

            // Update sample intensities
            for (int n = 0; n < nsamp; n++) {
              const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

              Real nu_fluid0 = nusamp(0);
              Real nu_lab0 = 0.;
              SPACETIMELOOP(nu) { nu_lab0 -= ncov(nu, nswarm) * ucon[nu]; }
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
                                   v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                                   v(iblock, pye, k, j, i), species_d[ispec], nu_fluid);
                v(iblock, ijinvs(ispec, bin), k, j, i) =
                    v(iblock, Inu0(ispec, bin), k, j, i) /
                    (nu_fluid * nu_fluid * nu_fluid) * alphainv_s;
              }

              for (int bin = 0; bin < nu_bins; bin++) {
                const Real nu_fluid = nusamp(bin);
                const Real nu_lab = std::exp(std::log(nu_fluid) - dlnu * shift);
                const Real ds = dt / ucon[0] / nu_fluid;
                const Real alphainv_a =
                    nu_fluid * opac_d.TotalScatteringCoefficient(
                                   v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                                   v(iblock, pye, k, j, i), species_d[ispec], nu_fluid);
                const Real alphainv_s =
                    nu_fluid * opac_d.TotalScatteringCoefficient(
                                   v(iblock, pdens, k, j, i), v(iblock, pT, k, j, i),
                                   v(iblock, pye, k, j, i), species_d[ispec], nu_fluid);
                const Real jinv_a =
                    opac_d.ThermalDistributionOfTNu(v(iblock, pT, k, j, i),
                                                    species_d[ispec], nu_fluid) /
                    (nu_fluid * nu_fluid * nu_fluid) * alphainv_a;

                // Interpolate invariant scattering emissivity to lab frame
                Real jinv_s = 0.;
                interp.GetIndicesAndWeights(bin, nubin_shift, nubin_wgt);
                for (int isup = 0; isup < interp.StencilSize(); isup++) {
                  jinv_s += nubin_wgt[isup] *
                            v(iblock, ijinvs(ispec, nubin_shift[isup]), k, j, i);
                }

                Inuinv(bin, ispec, nswarm) =
                    (Inuinv(bin, ispec, nswarm) + ds * (jinv_a + jinv_s)) /
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

// TODO(BRR): Hack to get around current lack of support for packing parthenon swarms
template <>
TaskStatus MOCMCEddington(MeshData<Real> *rc) {
  for (int n = 0; n < rc->NumBlocks(); n++) {
    MOCMCEddington(rc->GetBlockData(n).get());
  }
  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCEddington(T *rc) {
  // Assume list is sorted!
  namespace ir = radmoment_internal;

  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto geom = Geometry::GetCoordinateSystem(rc);

  std::vector<std::string> variables{ir::tilPi, pf::velocity::name()};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);
  const auto pvel_lo = imap[pf::velocity::name()].first;

  const int nu_bins = rad->Param<int>("nu_bins");
  const Real dlnu = rad->Param<Real>("dlnu");
  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  auto num_species = rad->Param<int>("num_species");

  // TODO(BRR) block packing eventually
  const int iblock = 0;

  auto swarm_d = swarm->GetDeviceContext();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // initialize eddington to zero
        for (int s = 0; s < num_species; s++) {
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = ii; jj < 3; jj++) {
              v(iTilPi(s, ii, jj), k, j, i) = 0.0;
            }
          }
        }
        Real energy[MAX_SPECIES] = {0.0};
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

        Real cov_g[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, iblock, k, j, i, cov_g);
        Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
        Real con_beta[3];
        geom.ContravariantShift(CellLocation::Cent, iblock, k, j, i, con_beta);
        const Real vpcon[] = {v(pvel_lo, k, j, i), v(pvel_lo + 1, k, j, i),
                              v(pvel_lo + 2, k, j, i)};
        const Real W = phoebus::GetLorentzFactor(vpcon, cov_g);
        const Real ucon[4] = {W / alpha, vpcon[0] - con_beta[0] * W / alpha,
                              vpcon[1] - con_beta[1] * W / alpha,
                              vpcon[2] - con_beta[2] * W / alpha};

        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

          Real nu_fluid0 = nusamp(0);
          Real nu_lab0 = 0.;
          SPACETIMELOOP(nu) { nu_lab0 -= ncov(nu, nswarm) * ucon[nu]; }
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
                I[s] += nubin_wgt[isup] * Inuinv(nubin_shift[isup], s, nswarm) *
                        pow(nusamp(nubin), 4);
              }
            }
          }

          Real wgts[6];
          kdgrid::integrate_ninj_domega_quad(mu_lo(nswarm), mu_hi(nswarm), phi_lo(nswarm),
                                             phi_hi(nswarm), wgts);
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = ii; jj < 3; jj++) {
              const int ind = Geometry::Utils::Flatten2(ii, jj, 3);
              for (int s = 0; s < num_species; s++) {
                v(iTilPi(s, ii, jj), k, j, i) += wgts[ind] * I[s];
              }
            }
          }
          for (int s = 0; s < num_species; s++) {
            energy[s] += (mu_hi(nswarm) - mu_lo(nswarm)) *
                         (phi_hi(nswarm) - phi_lo(nswarm)) * I[s];
          }
        }

        if (nsamp > 0) {
          for (int s = 0; s < num_species; s++) {
            for (int ii = 0; ii < 3; ii++) {
              for (int jj = ii; jj < 3; jj++) {
                v(iTilPi(s, ii, jj), k, j, i) /= energy[s];
              }
              v(iTilPi(s, ii, ii), k, j, i) -= 1. / 3.;
            }
          }
        }

        for (int s = 0; s < num_species; s++) {
          v(iTilPi(s, 1, 0), k, j, i) = v(iTilPi(s, 0, 1), k, j, i);
          v(iTilPi(s, 2, 0), k, j, i) = v(iTilPi(s, 0, 2), k, j, i);
          v(iTilPi(s, 2, 1), k, j, i) = v(iTilPi(s, 1, 2), k, j, i);
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

template TaskStatus MOCMCSampleBoundaries<MeshBlockData<Real>>(MeshBlockData<Real> *);
template TaskStatus MOCMCReconstruction<MeshBlockData<Real>>(MeshBlockData<Real> *);
template TaskStatus MOCMCEddington<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
template void MOCMCInitSamples<MeshBlockData<Real>>(MeshBlockData<Real> *);
template TaskStatus MOCMCTransport<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                                        const Real dt);
// template TaskStatus MOCMCFluidSource<MeshData<Real>>(MeshData<Real> *rc, const Real dt,
// const bool update_fluid);
template TaskStatus MOCMCFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                                          const Real dt,
                                                          const bool update_fluid);

} // namespace radiation
