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

// singularity
#include <singularity-eos/eos/eos.hpp>
#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

#include "closure.hpp"
#include "geodesics.hpp"
#include "kd_grid.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"
#include "phoebus_utils/root_find.hpp"

namespace radiation {

namespace pf = fluid_prim;
namespace cr = radmoment_cons;
namespace pr = radmoment_prim;
namespace ir = radmoment_internal;
namespace im = mocmc_internal;
using namespace singularity;
using namespace singularity::neutrinos;
using singularity::RadiationType;
using vpack_types::FlatIdx;

constexpr int MAX_SPECIES = 3;

template <class T>
void MOCMCAverageOpacities(T *rc);

// TODO(BRR) add options
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
              Inuinv(nubin, s, n) = d_opac.ThermalDistributionOfTNu(Temp, type, nu) / pow(nu,3);
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

  // Initialize eddington tensor and opacities for first step
  MOCMCReconstruction(rc);
  MOCMCEddington(rc);
  MOCMCAverageOpacities(rc);
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
  std::vector<std::string> vars{p::density, p::temperature, p::ye,   p::velocity,
                                ir::kappaJ, ir::kappaH,     ir::JBB, im::Inu0};

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);
  auto Inu0 = imap.GetFlatIdx(im::Inu0);

  const int nblock = v.GetDim(5);
  PARTHENON_REQUIRE_THROWS(nblock == 1, "Packing not currently supported for swarms");

  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  auto swarm_d = swarm->GetDeviceContext();

  // Get the device opacity object
  using namespace singularity::neutrinos;
  const auto d_opacity = opac->Param<Opacity>("d.opacity");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real TIME = unit_conv.GetTimeCodeToCGS();

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
      DEFAULT_LOOP_PATTERN, "MOCMC::AverageOpacities", DevExecSpace(), 0, nblock - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real enu = 10.0; // Assume we are gray for now or can take the peak opacity
                               // at enu = 10 MeV
        const Real rho = v(b, prho, k, j, i);
        const Real Temp = v(b, pT, k, j, i);
        const Real Ye = v(b, pYe, k, j, i);
        const Real T_code = v(b, pT, k, j, i);

        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);

        // Angle-average intensities
        for (int s = 0; s < num_species; s++) {
          for (int bin = 0; bin < nu_bins; bin++) {
            v(b, Inu0(s, bin), k, j, i) = 0.;
          }
        }
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          const Real dOmega =
              (mu_hi(nswarm) - mu_lo(nswarm)) * (phi_hi(nswarm) - phi_lo(nswarm));
          for (int s = 0; s < num_species; s++) {
            for (int bin = 0; bin < nu_bins; bin++) {
              v(b, Inu0(s, bin), k, j, i) +=
                  Inuinv(s, bin) * pow(nusamp(bin), 3) * dOmega;
            }
          }
        }
        for (int s = 0; s < num_species; s++) {
          for (int bin = 0; bin < nu_bins; bin++) {
            v(b, Inu0(s, bin), k, j, i) /= (4. * M_PI);
          }
        }

        // Frequency- (and angle-)average opacities
        for (int ispec = 0; ispec < num_species; ispec++) {
          v(b, idx_kappaJ(ispec), k, j, i) = 0.;
          Real weight = 0.;
          for (int bin = 0; bin < nu_bins; bin++) {
            v(b, idx_kappaJ(ispec), k, j, i) +=
                d_opacity.AbsorptionCoefficient(rho, Temp, Ye, species_d[ispec],
                                                nusamp(bin) * TIME) *
                v(b, Inu0(ispec, bin), k, j, i);
            v(b, idx_kappaH(ispec), k, j, i) +=
                d_opacity.AbsorptionCoefficient(rho, Temp, Ye, species_d[ispec],
                                                nusamp(bin) * TIME) *
                (1.0 - scattering_fraction) * v(b, Inu0(ispec, bin), k, j, i);
            weight += v(b, Inu0(ispec, bin), k, j, i);
          }
          v(b, idx_kappaJ(ispec), k, j, i) /= weight;
          v(b, idx_kappaH(ispec), k, j, i) /= weight;
          v(b, idx_JBB(ispec), k, j, i) =
              d_opacity.ThermalDistributionOfT(Temp, species_d[ispec]);
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
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
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

          // TODO(BRR): Convert ncov from lab frame to fluid frame
          Real ncov_tetrad[4] = {-1., ncov(1, nswarm), ncov(2, nswarm), ncov(3, nswarm)};

          const Real mu = 1.0 - ncov_tetrad[1];
          const Real phi = atan2(ncov_tetrad[3], ncov_tetrad[2]);

          for (int m = 0; m < n; m++) {
            const int mswarm = swarm_d.GetFullIndex(k, j, i, m);
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

class Residual {
 public:
  KOKKOS_FUNCTION
  Residual(const Real &dtau, const EOS &eos, const Opacity &opac,
           const VariablePack<Real> &var, const int &iprho, const int &ipeng,
           const FlatIdx &iJ, const FlatIdx &iInu0, const FlatIdx &iInu1,
           const int &nspecies, const int &nbins, const RadiationType *species,
           const ParArray1D<Real> &nusamp, const Real &dlnu, const int &b, const int &k,
           const int &j, const int &i)
      : dtau_(dtau), eos_(eos), opac_(opac), var_(var), iprho_(iprho), ipeng_(ipeng),
        iJ_(iJ), iInu0_(iInu0), iInu1_(iInu1), nspecies_(nspecies), nbins_(nbins),
        species_(species), nusamp_(nusamp), dlnu_(dlnu), b_(b), k_(k), j_(j), i_(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Real ug1, const Real Ye1, Real res[2]) {
    const Real &rho = var_(b_, iprho_, k_, j_, i_);
    const Real &ug0 = var_(b_, ipeng_, k_, j_, i_);
    printf("b: %i peng: %i k j i: %i %i %i ug0: %e\n", b_, ipeng_, k_, j_, i_, ug0);
    Real lambda[2] = {Ye1, 0.};
    const Real temp1 = eos_.TemperatureFromDensityInternalEnergy(rho, ug1 / rho, lambda);

    // Update Inu1 from Inu0, use Inu1 to average opacities
    Real Jem = 0.;
    Real kappaJ = 0.;
    Real kappaH = 0.;
    Real weight = 0.;
    Real ur0 = 0.;
    for (int s = 0; s < nspecies_; s++) {
      ur0 += var_(b_, iJ_(s), k_, j_, i_);
      for (int bin = 0; bin < nbins_; bin++) {
        Real trapezoidal_rule = 1.;
        if (bin == 0 || bin == nbins_ - 1) {
          trapezoidal_rule = 0.5;
        }

        const Real nu = nusamp_(bin);
        Real &Inu0 = var_(b_, iInu0_(s, bin), k_, j_, i_);
        // Real &Inu1 = var_(b_, iInu1_(s, bin), k_, j_, i_);
        Real Jnu = opac_.EmissivityPerNu(rho, temp1, Ye1, species_[s], nu);
        Real kappanu =
            opac_.AngleAveragedAbsorptionCoefficient(rho, temp1, Ye1, species_[s], nu);
        Real Inu1 = (Inu0 + dtau_ * Jnu) / (1. + dtau_ * kappanu);

        Jem += trapezoidal_rule * Jnu * nu * dlnu_;
        // TODO(BRR) scattering fraction = 0
        kappaJ += trapezoidal_rule * kappanu * Inu1 * nu * dlnu_;
        kappaH += trapezoidal_rule * kappanu * Inu1 * nu * dlnu_;
        weight += trapezoidal_rule * Inu1 * nu * dlnu_;
      }
    }

    kappaJ /= weight;
    kappaH /= weight;

    if (i_ == 10) {
      printf("Jem: %e kappaJ: %e kappaH: %e\n", Jem, kappaJ, kappaH);
    }

    // TODO(BRR): Add energy change due to inelastic scattering
    const Real dur = 0.;

    res[0] = ((ug1 - ug0) + (ur0 + dtau_ * Jem) / (1. + dtau_ * kappaJ) - ur0 + dur) /
           (ug0 + ur0);
           // TODO(BRR) actually do Ye update
    res[1] = 0.;

    if (i_ == 10) {
      printf("ug1: %e ug0: %e ur0: %e Jem: %e kappaJ: %e dur: %e\n",
        ug1, ug0, ur0, Jem, kappaJ, dur);
      printf("resid: %e %e\n", res[0], res[1]);
    }
  }

 private:
  const Real &dtau_;
  const EOS &eos_;
  const Opacity &opac_;
  const VariablePack<Real> &var_;
  const int &iprho_, &ipeng_;
  const FlatIdx &iJ_;
  const FlatIdx &iInu0_, &iInu1_;
  const int &nspecies_, &nbins_;
  const RadiationType *species_;
  const ParArray1D<Real> &nusamp_;
  const Real &dlnu_;
  const int &b_, &k_, &j_, &i_;
};

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
  const auto opac_d = opac->template Param<Opacity>("d.opacity");
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  const auto eos_d = eos->template Param<EOS>("d.EOS");
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real TIME = unit_conv.GetTimeCodeToCGS();

  std::vector<std::string> variables{
      pr::J,  pr::H,     pf::density, pf::energy, pf::velocity, pf::temperature,
      pf::ye, ir::tilPi, im::dnsamp,  im::Inu0,   im::Inu1};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  const auto pJ = imap.GetFlatIdx(pr::J);
  const auto pH = imap.GetFlatIdx(pr::H);
  const auto pdens = imap[pf::density].first;
  const auto peng = imap[pf::energy].first;
  const auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  const auto pT = imap[pf::temperature].first;
  const auto pye = imap[pf::ye].first;
  const auto Inu0 = imap.GetFlatIdx(im::Inu0);
  const auto Inu1 = imap.GetFlatIdx(im::Inu1);
  const auto iTilPi = imap.GetFlatIdx(ir::tilPi);

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

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::AverageOpacities", DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);
        const Real dtau = dt; // TODO(BRR): dtau = dt / u^0

        // Angle-average samples
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          const Real dOmega =
              (mu_hi(nswarm) - mu_lo(nswarm)) * (phi_hi(nswarm) - phi_lo(nswarm));
        }

        auto res = Residual(dtau, eos_d, opac_d, v, pdens, peng, pJ, Inu0, Inu1,
                            num_species, nu_bins, species_d, nusamp, dlnu, 0, k, j, i);

        Real guess[2] = {v(pdens, k, j, i), v(pye, k, j, i)};
        root_find::RootFindStatus status;
        root_find::broyden2(res, guess, 50, 1.e-10, &status);
        if (i == 10) {
          printf("status = %i\n", static_cast<int>(status));
        }

        //Real error[2];
        //res(v(pdens, k, j, i), v(pye, k, j, i), error);


      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCEddington(T *rc) {
  // Assume list is sorted!
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

  const auto &Inuinv = swarm->template Get<Real>("Inuinv").Get();
  const auto &mu_lo = swarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swarm->template Get<Real>("phi_hi").Get();
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  const int nu_bins = rad->Param<int>("nu_bins");
  auto nusamp = rad->Param<ParArray1D<Real>>("nusamp");
  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }
  constexpr int MAX_SPECIES = 3;

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
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);

          // get the energy integrated intensity
          // TODO(jcd): do we need a dlnu or something here
          Real I[MAX_SPECIES] = {0.0};
          for (int nubin = 0; nubin < nu_bins; nubin++) {
            for (int s = 0; s < num_species; s++) {
              I[s] += Inuinv(nubin, s, nswarm) * pow(nusamp(nubin), 3);
            }
          }
          if (i == 10) {
            printf("I: %e %e %e\n", I[0], I[1], I[2]);
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
        for (int s = 0; s < num_species; s++) {
          for (int ii = 0; ii < 3; ii++) {
            for (int jj = ii; jj < 3; jj++) {
              v(iTilPi(s, ii, jj), k, j, i) /= energy[s];
            }
          }
          v(iTilPi(s, 1, 0), k, j, i) = v(iTilPi(s, 0, 1), k, j, i);
          v(iTilPi(s, 2, 0), k, j, i) = v(iTilPi(s, 0, 2), k, j, i);
          v(iTilPi(s, 2, 1), k, j, i) = v(iTilPi(s, 1, 2), k, j, i);
        }

        if (i == 10) {
          SPACELOOP2(ii, jj) {
            printf("tilPi[%i %i] = %e\n", i, j, v(iTilPi(0, 1, 0), k, j, i));
          }
        }
      });
  /*Real Iold = 0.;
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
});*/

  return TaskStatus::complete;
}

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
template void MOCMCAverageOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace radiation
