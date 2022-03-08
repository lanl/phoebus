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
#include "kd_grid.hpp"
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
        printf("%s:%i\n", __FILE__, __LINE__);
        Real Jtot = 0.;
        for (int s = 0; s < 3; s++) {
          Jtot += v(b, pJ(s), k, j, i);
        }
        printf("%s:%i\n", __FILE__, __LINE__);
        v(b, dn, k, j, i) =
            get_nsamp_per_zone(k, j, i, geom, v(b, pdens, k, j, i), v(b, pT, k, j, i),
                               v(b, pye, k, j, i), Jtot, nsamp_per_zone);
        nsamp += v(b, dn, k, j, i);
        printf("%s:%i\n", __FILE__, __LINE__);
        // Kokkos::atomic_add(&(nsamptot(b)), static_cast<int>(v(b, dn, k, j, i)));
        printf("%s:%i\n", __FILE__, __LINE__);
      },
      Kokkos::Sum<int>(nsamp_tot));

  printf("nsamptot: %i\n", nsamp_tot);

  ParArrayND<int> new_indices;
  auto new_mask = swarm->AddEmptyParticles(nsamp_tot, new_indices);

  printf("new_indices.GetDim(1): %i\n", new_indices.GetDim(1));

  const auto &x = swarm->template Get<Real>("x").Get();
  const auto &y = swarm->template Get<Real>("y").Get();
  const auto &z = swarm->template Get<Real>("z").Get();
  const auto &ncov = swarm->template Get<Real>("ncov").Get();
  const auto &Inu = swarm->template Get<Real>("Inu").Get();

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
      //parthenon::loop_pattern_flatrange_tag, "MOCMC:Init::Sample", DevExecSpace(),
      DEFAULT_LOOP_PATTERN, "MOCMC:Init::Sample", DevExecSpace(),
      0, nblock - 1,
      0, new_indices.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int m) {
        const int n = new_indices(m);
        auto rng_gen = rng_pool.get_state();

        int i, j, k;
        swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

        const Real rho = v(b, pdens, k, j, i);
        const Real Temp = v(b, pT, k, j, i);
        const Real Ye = v(b, pye, k, j, i);
        Real lambda[2] = {Ye, 0.};

        for (int s = 0; s < num_species; s++) {
          const RadiationType type = species_d[s];
          for (int nubin = 0; nubin < nu_bins; nubin++) {

            const Real nu = nusamp(nubin) * TIME;
            Inu(nubin, s, n) = d_opac.EmissivityPerNu(rho, Temp, Ye, type, nu, lambda) /
              d_opac.AbsorptionCoefficient(rho, Temp, Ye, type, nu, lambda);
            if (use_B_fake) Inu(nubin, s, n) = B_fake;
          }
        }

        // Sample uniformly in solid angle
        Real ncov_comov[4] = {0.};
        const Real theta = acos(2. * rng_gen.drand() - 1.);
        const Real phi = 2. * M_PI * rng_gen.drand();
        const Real ncov_tetrad[4] = {-1.,
                                     cos(theta),
                                     cos(phi) * sin(theta),
                                     sin(phi) * sin(theta)};

        // TODO(BRR) do an actual transformation from fluid to lab frame
        SPACETIMELOOP(mu) {
          ncov(mu, n) = ncov_tetrad[mu];
        }

        // Set intensity to thermal equilibrium
        rng_pool.free_state(rng_gen);
      });
}

template <class T>
TaskStatus MOCMCTransport(T *rc) {
  return TaskStatus::complete;
}
template TaskStatus MOCMCTransport<MeshBlockData<Real>>(MeshBlockData<Real> *);

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
  const auto &Inu = swarm->template Get<Real>("Inu").Get();
  const auto &mu_lo = swaarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swaarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swaarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swaarm->template Get<Real>("phi_hi").Get();
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }
  const int nu_bins = rad->Param<int>("nu_bins");
  constexpr int MAX_SPECIES = 3;

  swarm->SortParticlesByCell();
  auto swarm_d = swarm->GetDeviceContext();

  auto mocmc_recon = rad->Param<MOCMCRecon>("mocmc_recon");

  if (mocmc_recon == MOCMCRecon::kdgrid) {
    parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(),
      kb.s, kb.e,
      jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int nsamp = swarm_d.GetParticleCountPerCell(k, j, i);
        for (int n = 0; n < nsamp; n++) {
          const int nswarm = swarm_d.GetFullIndex(k, j, i, n);
          if (n == 0) {
            mu_lo(nswarm) = 0.0;
            mu_hi(nswarm) = 2.0;
            phi_lo(nswarm) = 0.0;
            phi_hi(nswarm) = 2.0*M_PI;
            continue;
          }

          // TODO(BRR): Convert ncov from lab frame to fluid frame
          Real ncov_tetrad[4] = {-1., ncov(1, nswarm), ncov(2, nswarm), ncov(3, nswarm)};

          const Real mu = 1.0 - ncov_tetrad[1];
          const Real phi = atan2(ncov_tetrad[3], ncov_tetrad[2]);

          for (int m = 0; m < n; m++) {
            const int mswarm = sward_d.GetFullIndex(k, j, i, m);
            if (mu > mu_lo(mswarm) && mu < mu_hi(mswarm) && phi > phi_lo(mswarm) && phi < phi_hi(mswarm)) {
              Real mcov_tetrad[4] = {-1., ncov(1, mswarm), ncov(2, mswarm), ncov(3, mswarm)};
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
                  mu_hi(nswarm) = mu_hi(mswarm)
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
          } // m = 0..n
        } // n = 0..nsamp
      });
  }

  return TaskStatus::complete;
}

template <class T>
TaskStatus MOCMCEddington(T *rc) {
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

  const auto &Inu = swarm->template Get<Real>("Inu").Get();
  const auto &mu_lo = swaarm->template Get<Real>("mu_lo").Get();
  const auto &mu_hi = swaarm->template Get<Real>("mu_hi").Get();
  const auto &phi_lo = swaarm->template Get<Real>("phi_lo").Get();
  const auto &phi_hi = swaarm->template Get<Real>("phi_hi").Get();
  auto iTilPi = imap.GetFlatIdx(ir::tilPi);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }
  const int nu_bins = rad->Param<int>("nu_bins");
  constexpr int MAX_SPECIES = 3;

  // TODO(jcd): we don't need to sort again do we.  already did this in recon
  //swarm->SortParticlesByCell();
  auto swarm_d = swarm->GetDeviceContext();
  parthenon::par_for(
    DEFAULT_LOOP_PATTERN, "MOCMC::kdgrid", DevExecSpace(),
    kb.s, kb.e,
    jb.s, jb.e,
    ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      // initialize eddington to zero
      for (int s = 0; s < num_species; s++) {
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = ii; jj < 3; jj++) {
            v(iTilPi(s,ii,jj), k, j, i) = 0.0;
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
            I[s] += Inu(nubin, s, nswarm);
          }
        }

        Real wgts[6];
        kdgrid::integrate_ninj_domega_quad(mu_lo(nswarm), mu_hi(nswarm), phi_lo(nswarm), phi_hi(nswarm), wgts);
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = ii; jj < 3; jj++) {
            const int ind = geometry_utils::Flatten2(ii,jj,3);
            for (int s = 0; s < num_species; s++) {
              v(iTilPi(s,ii,jj), k, j, i) += wgts[ind] * I[s];
            }
          }
        }
        for (int s = 0; s < num_species; s++) {
          energy[s] += (mu_hi(nswarm) - mu_lo(nswarm)) * (phi_hi(nswarm) - phi_lo(nswarm)) * I[s];
        }
      }
      for (int s = 0; s < num_species; s++) {
        for (int ii = 0; ii < 3; ii++) {
          for (int jj = ii; jj < 3; jj++) {
            v(iTili(s,ii,jj), k, j, i) /= energy[s];
          }
        }
        v(iTilPi(s,1,0),k,j,i) = v(iTilPi(s,0,1),k,j,i);
        v(iTilPi(s,2,0),k,j,i) = v(iTilPi(s,0,2),k,j,i);
        v(iTilPi(s,2,1),k,j,i) = v(iTilPi(s,1,2),k,j,i);
      }
    });
  return TaskStatus::complete;
}

template TaskStatus MOCMCReconstruction<MeshBlockData<Real>>(MeshBlockData<Real> *);
template void MOCMCInitSamples<MeshBlockData<Real>>(MeshBlockData<Real> *);

} // namespace radiation
