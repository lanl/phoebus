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

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "closure.hpp"
#include "geodesics.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

namespace radiation {

namespace pf = fluid_prim;
namespace cr = radmoment_cons;
namespace pr = radmoment_prim;
namespace ir = radmoment_internal;
namespace im = mocmc_internal;
using namespace singularity::neutrinos;
using singularity::RadiationType;

constexpr int MAX_SPECIES = 3;

template <class T>
void MOCMCAverageOpacities(T *rc);

KOKKOS_INLINE_FUNCTION
int get_nsamp_per_zone(const int &k, const int &j, const int &i,
                       const Geometry::CoordSysMeshBlock &geom, const Real &rho,
                       const Real &T, const Real &Ye, const Real &J,
                       const int &nsamp_per_zone_global) {

  return nsamp_per_zone_global;
}

template <class T>
void MOCMCInitSamples(T *rc) {

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  // auto swarm = pmb->swarm_data.Get()->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  auto rng_pool = rad->Param<RNGPool>("rng_pool");

  // Meshblock geometry
  const auto geom = Geometry::GetCoordinateSystem(rc);
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

  // Microphysics
  auto opac = pmb->packages.Get("opacity");
  const auto d_opac = opac->template Param<Opacity>("d.opacity");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  // auto &dnsamp = rc->Get(mocmc_internal::dnsamp);
  std::vector<std::string> variables{pr::J,           pr::H,  pf::density, pf::velocity,
                                     pf::temperature, pf::ye, im::dnsamp};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto pJ = imap.GetFlatIdx(pr::J);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pdens = imap[pf::density].first;
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  auto pT = imap[pf::temperature].first;
  auto pye = imap[pf::ye].first;
  auto dn = imap[im::dnsamp].first;

  const int nblock = v.GetDim(5);
  PARTHENON_REQUIRE_THROWS(nblock == 1, "Packing not currently supported for swarms");

  const int nsamp_per_zone = rad->Param<int>("nsamp_per_zone");
  int nsamp_tot = 0;

  // Fill nsamp per zone per species and sum over zones
  // TODO(BRR) make this a separate function and use it to update dnsamp which is then
  // used to decide whether to refine/derefine
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "MOCMC::Init::NumSamples", DevExecSpace(), 0, nblock - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nsamp) {
        Real Jtot = 0.;
        for (int s = 0; s < 3; s++) {
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
        starting_index(k, j, i) = index;
        index += static_cast<int>(dN_h(k + kb.s, j + jb.s, i + ib.s));
      }
    }
  }
  starting_index.DeepCopy(starting_index_h);

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();

  auto swarm_d = swarm->GetDeviceContext();

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  const auto B_fake = rad->Param<Real>("B_fake");
  const auto use_B_fake = rad->Param<bool>("use_B_fake");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::Init::Sample", DevExecSpace(), 0, nblock - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int start_idx = starting_index(k - kb.s, j - jb.s, i - ib.s);
        auto rng_gen = rng_pool.get_state();

        for (int nsamp = 0; nsamp < static_cast<int>(v(b, dn, k, j, i)); nsamp++) {
          const int n = new_indices(start_idx + nsamp);

          // Create particles at zone centers
          x(n) = minx_i + (i - ib.s + rng_gen.drand()) * dx_i;
          y(n) = minx_j + (j - jb.s + rng_gen.drand()) * dx_j;
          z(n) = minx_k + (k - kb.s + rng_gen.drand()) * dx_k;

          const Real rho = v(b, pdens, k, j, i);
          const Real Temp = v(b, pT, k, j, i);
          const Real Ye = v(b, pye, k, j, i);
          Real lambda[2] = {Ye, 0.};

          for (int s = 0; s < num_species; s++) {
            const RadiationType type = species_d[s];
            for (int nubin = 0; nubin < nu_bins; nubin++) {
              const Real nu = nusamp(nubin) * TIME;
              Inuinv(nubin, s, n) =
                  d_opac.EmissivityPerNu(rho, Temp, Ye, type, nu, lambda) /
                  d_opac.AbsorptionCoefficient(rho, Temp, Ye, type, nu, lambda) /
                  pow(nu, 3);
              if (use_B_fake) Inuinv(nubin, s, n) = B_fake / pow(nu, 3);
            }
          }

          // Sample uniformly in solid angle
          Real ncov_comov[4] = {0.};
          const Real theta = acos(2. * rng_gen.drand() - 1.);
          const Real phi = 2. * M_PI * rng_gen.drand();
          const Real ncov_tetrad[4] = {-1., cos(theta), cos(phi) * sin(theta),
                                       sin(phi) * sin(theta)};

          // TODO(BRR) do an actual transformation from fluid to lab frame
          SPACETIMELOOP(mu) { ncov(mu, n) = ncov_tetrad[mu]; }
        }

        rng_pool.free_state(rng_gen);
      });

  // Initialize kappaH for diffusion on first step
//  MOCMCAverageOpacities(rc);
}

template <class T>
void MOCMCAverageOpacities(T *rc) {
  // Assume particles are already sorted!

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  StateDescriptor *opac = pmb->packages.Get("opacity").get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{p::density, p::temperature, p::ye,  p::velocity,
                                ir::kappaJ, ir::kappaH,     ir::JBB};

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);

  const int nblock = v.GetDim(5);
  PARTHENON_REQUIRE_THROWS(nblock == 1, "Packing not currently supported for swarms");

  // Get the device opacity object
  using namespace singularity::neutrinos;
  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  // Mainly for testing purposes, probably should be able to do this with the opacity code
  // itself
  const auto B_fake = rad->Param<Real>("B_fake");
  const auto use_B_fake = rad->Param<bool>("use_B_fake");
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::AverageOpacities", DevExecSpace(), 0, nblock - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real enu = 10.0; // Assume we are gray for now or can take the peak opacity
                               // at enu = 10 MeV
        const Real rho = v(b, prho, k, j, i);
        const Real Temp = v(b, pT, k, j, i);
        const Real Ye = v(b, pYe, k, j, i);
        const Real T_code = v(b, pT, k, j, i);

        for (int ispec = 0; ispec < num_species; ispec++) {
          Real kappa =
              d_opacity.AbsorptionCoefficient(rho, Temp, Ye, species_d[ispec], enu);
          const Real emis = d_opacity.Emissivity(rho, Temp, Ye, species_d[ispec]);
          Real B = emis / kappa;
          if (use_B_fake) B = B_fake;

          v(b, idx_JBB(ispec), k, j, i) = B;
          v(b, idx_kappaJ(ispec), k, j, i) = kappa * (1.0 - scattering_fraction);
          v(b, idx_kappaH(ispec), k, j, i) = kappa;
        }
      });
}

template <class T>
TaskStatus MOCMCReconstruction(T *rc) {

  namespace ir = radmoment_internal;

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> variables{ir::tilPi};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[MAX_SPECIES] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  swarm->SortParticlesByCell();
  auto swarm_d = swarm->GetDeviceContext();

  auto mocmc_recon = rad->Param<MOCMCRecon>("mocmc_recon");

  if (mocmc_recon == MOCMCRecon::constdmudphi) {
    // TODO: Allocate dmu dphi grid of intensities per species

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "MOCMC::ConstDmuDphi", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);
          for (int n = 0; n < nsamp; n++) {
            const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

            // TODO(BRR): Convert ncov from lab frame to fluid frame
            Real ncov_tetrad[4] = {-1., ncov(1, nswarm), ncov(2, nswarm),
                                   ncov(3, nswarm)};

            const Real mu = ncov_tetrad[3];
            const Real phi = atan2(ncov_tetrad[2], ncov_tetrad[1]);
            Real I[MAX_SPECIES] = {0.};
            for (int s = 0; s < num_species; s++) {
              const RadiationType type = species_d[s];
              for (int nubin = 0; nubin < nu_bins; nubin++) {
                const Real nu = nusamp(nubin); // ignore frequency units b.c. norm
                I[s] += Inuinv(nubin, s, nswarm) * pow(nu, 3); // dlnu = const
              }
            }

            // TODO: Deposit I on mu, phi grid
          }

          // TODO: subtract mean from all mu, phi cells
        });

    // TODO: Fill in iTilPi -> v(0, iTilPi(s, ii, jj), k, j, i)
  }

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCTransport(T *rc, const Real dt) {
  auto *pmb = rc->GetParentPointer().get();
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

  auto *pmb = rc->GetParentPointer().get();
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
  const auto d_opac = opac->template Param<Opacity>("d.opacity");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  std::vector<std::string> variables{pr::J,           pr::H,   pf::density, pf::velocity,
                                     pf::temperature, pf::ye,  ir::tilPi,   im::dnsamp,
                                     im::Inu0,        im::Inu1};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto pJ = imap.GetFlatIdx(pr::J);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pdens = imap[pf::density].first;
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  auto pT = imap[pf::temperature].first;
  auto pye = imap[pf::ye].first;
  auto Inu0 = imap.GetFlatIdx(im::Inu0);
  auto Inu1 = imap.GetFlatIdx(im::Inu1);
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  auto swarm_d = swarm->GetDeviceContext();

  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  const int nu_bins = rad->Param<int>("nu_bins");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  printf("num_species: %i nu_bins: %i\n", num_species, nu_bins);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::FluidSource", DevExecSpace(), kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int b = 0;
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

        // TODO(BRR): relativity -> dtau = dt / u^0
        const Real dtau = dt;

        const Real dOmega =
            4. * M_PI / nsamp; // TODO(BRR): Get a real calculation of dOmega

        // Angle-average specific intensity over samples
        // Sample Inuinv to tetrad Inu
        // if (angle_averaging == MOCMCAngleAveraging::first_order)

        for (int s = 0; s < num_species; s++) {
          for (int bin = 0; bin < nu_bins; bin++) {
            v(b, Inu0(s, bin), k, j, i) = 0.;
          }
        }

        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          Real Iold = 0.;
          Real Inew = 0.;

          for (int s = 0; s < num_species; s++) {
            for (int bin = 0; bin < nu_bins; bin++) {
              // TODO(BRR) shift in frequency
              v(b, Inu0(s, bin), k, j, i) +=
                  Inuinv(bin, s, nswarm) * pow(nusamp(bin), 3) / dOmega;
              Iold += v(b, Inu0(s, bin), k, j, i);
              Inew += v(b, Inu0(s, bin), k, j, i); // After frequency shift
            }
            // Normalize shifted spectrum
            for (int bin = 0; bin < nu_bins; bin++) {
              v(b, Inu0(s, bin), k, j, i) *= Iold / Inew;
            }
          }
        }
      });

  return TaskStatus::complete;
}

template TaskStatus MOCMCReconstruction<MeshBlockData<Real>>(MeshBlockData<Real> *);
template void MOCMCInitSamples<MeshBlockData<Real>>(MeshBlockData<Real> *);
template TaskStatus MOCMCTransport<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                                        const Real dt);
// template TaskStatus MOCMCFluidSource<MeshData<Real>>(MeshData<Real> *rc, const Real dt,
// const bool update_fluid);
template TaskStatus MOCMCFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *rc,
                                                          const Real dt,
                                                          const bool update_fluid);
template void MOCMCAverageOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace radiation
