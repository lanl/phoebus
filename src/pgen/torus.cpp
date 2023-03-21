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

// C/C++ includes
#include <cstdio>
#include <limits>

// Kokkos
#include "Kokkos_Random.hpp"

// Parthenon
#include <globals.hpp>

// Phoebus
#include "fluid/con2prim_robust.hpp"
#include "geometry/boyer_lindquist.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/adiabats.hpp"
#include "phoebus_utils/reduction.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "radiation/radiation.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace torus {

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

using namespace radiation;
using Microphysics::Opacities;
using Microphysics::EOS::EOS;

// Hack to avoid using a string
enum EosType { IdealGas, StellarCollapse };

class GasRadTemperatureResidual {
 public:
  KOKKOS_FUNCTION
  GasRadTemperatureResidual(const Real Ptot, const Real rho, const Opacities &opac,
                            const EOS &eos, RadiationType type, const Real Ye)
      : Ptot_(Ptot), rho_(rho), opac_(opac), eos_(eos), type_(type), Ye_(Ye) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real T) {
    Real lambda[2] = {Ye_, 0.0};
    return Ptot_ - (4. / 3. - 1.) * opac_.EnergyDensityFromTemperature(T, type_) -
           eos_.PressureFromDensityTemperature(rho_, T, lambda);
  }

 private:
  const Real Ptot_;
  const Real rho_;
  const Opacities &opac_;
  const EOS &eos_;
  const RadiationType type_;
  const Real Ye_;
};

/**
 * enthalpy residual for use with nuclear eos
 * computes fishbone enthalpy - nuclear EOS enthalpy to find
 * density, temperature given the correct enthalpy
 **/
class EnthalpyResidual {
 public:
  KOKKOS_FUNCTION
  EnthalpyResidual(const Real hm1, const Spiner::DataBox T, const Real Ye,
                   const Real h_min_sc, const EOS &eos)
      : hm1_(hm1), T_(T), Ye_(Ye), h_min_sc_(h_min_sc), eos_(eos) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real rho) {
    const Real h_sc = enthalpy_sc(rho);
    return hm1_ - h_sc + h_min_sc_; // hm1 = h_sc - h_min_sc
  }

 private:
  const Real hm1_;
  const Spiner::DataBox T_;
  const Real Ye_;
  const Real h_min_sc_;
  const EOS &eos_;

  KOKKOS_INLINE_FUNCTION
  Real enthalpy_sc(const Real rho) {
    Real lambda[2];
    lambda[0] = Ye_;
    const Real T = T_.interpToReal(std::log10(rho));
    const Real P = eos_.PressureFromDensityTemperature(rho, T, lambda);
    const Real e = eos_.InternalEnergyFromDensityTemperature(rho, T, lambda);
    return 1.0 + e + P / rho;
  }
};

enum class InitialRadiation { none, thermal };

// Prototypes
// ----------------------------------------------------------------------
KOKKOS_FUNCTION
Real lfish_calc(Real r, Real a);
KOKKOS_FUNCTION
Real log_enthalpy(const Real r, const Real th, const Real a, const Real rin, const Real l,
                  Real &uphi);
KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]);
void ComputeBetas(Mesh *pmesh, const Real rho_min_bnorm, Real &beta_min_global,
                  Real &beta_pmax);
KOKKOS_FUNCTION
void GetStateFromEnthalpy(const EOS &eos, const EosType eos_type, const Real hm1,
                          const Spiner::DataBox rho, const Spiner::DataBox temp,
                          const Real Ye, const Real h_min_sc, const Real kappa,
                          const Real gam, const Real Cv, const Real rho_rmax,
                          Real &rho_out, Real &u_out);
// ----------------------------------------------------------------------

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
                    "Problem \"torus\" requires FMKS geometry!");

  auto rc = pmb->meshblock_data.Get().get();

  auto rad_pkg = pmb->packages.Get("radiation");
  bool do_rad = rad_pkg->Param<bool>("active");

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density, fluid_prim::velocity,
                              fluid_prim::energy, fluid_prim::bfield, fluid_prim::ye,
                              fluid_prim::pressure, fluid_prim::temperature,
                              fluid_prim::gamma1, radmoment_prim::J, radmoment_prim::H},
                             imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int iblo = imap[fluid_prim::bfield].first;
  const int ibhi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  auto iJ = imap.GetFlatIdx(radmoment_prim::J, false);
  auto iH = imap.GetFlatIdx(radmoment_prim::H, false);
  const auto specB = iJ.IsValid() ? iJ.GetBounds(1) : parthenon::IndexRange();

  // this only works with ideal gases
  // The Fishbone solver needs to know about Ye
  // and the eos machinery needs to construct adiabats.
  const std::string eos_type = pin->GetString("eos", "type");
  const EosType eos_type_enum = (eos_type == "IdealGas") ? IdealGas : StellarCollapse;
  PARTHENON_REQUIRE_THROWS(eos_type == "IdealGas" || eos_type == "StellarCollapse",
                           "Torus setup only works with ideal gas");
  const Real gam = pin->GetReal("eos", "Gamma");
  const Real Cv = pin->GetReal("eos", "Cv");

  const Real rin = pin->GetOrAddReal("torus", "rin", 6.0);
  const Real rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
  const Real kappa = pin->GetOrAddReal("torus", "kappa", 1.e-2);
  const Real u_jitter = pin->GetOrAddReal("torus", "u_jitter", 1.e-2);
  const int seed = pin->GetOrAddInteger("torus", "seed", time(NULL));
  const int nsub = pin->GetOrAddInteger("torus", "nsub", 1);
  const Real Ye = pin->GetOrAddReal("torus", "Ye", 0.5);
  Real S = pin->GetOrAddReal("torus", "entropy", 4.0);
  Real lambda[2];
  lambda[0] = Ye;

  const Real a = pin->GetReal("geometry", "a");
  auto bl = Geometry::BoyerLindquist(a);

  // Solution constants
  const Real angular_mom = lfish_calc(rmax, a);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<EOS>("d.EOS");
  auto eos_h = pmb->packages.Get("eos")->Param<EOS>("h.EOS");
  auto floor = pmb->packages.Get("fixup")->Param<fixup::Floors>("floor");
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  S *= unit_conv.GetEntropyCGSToCode();

  auto geom = Geometry::GetCoordinateSystem(rc);

  // TODO(BRR) Make this an input parameter
  InitialRadiation init_rad = InitialRadiation::thermal;

  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  auto opacities = opac->Param<Opacities>("opacities");
  std::vector<RadiationType> species;
  int num_species = 0;
  RadiationType species_d[MaxNumRadiationSpecies] = {};
  if (do_rad) {
    const std::string init_rad_str =
        pin->GetOrAddString("torus", "initial_radiation", "None");
    if (init_rad_str == "None") {
      init_rad = InitialRadiation::none;
    } else if (init_rad_str == "thermal") {
      init_rad = InitialRadiation::thermal;
    } else {
      PARTHENON_FAIL("\"torus/initial_radiation\" not recognized!");
    }

    species = rad_pkg->Param<std::vector<RadiationType>>("species");
    num_species = rad_pkg->Param<int>("num_species");
    for (int s = 0; s < num_species; s++) {
      species_d[s] = species[s];
    }
  }

  // set up transformation stuff
  auto gpkg = pmb->packages.Get("geometry");
  bool derefine_poles = gpkg->Param<bool>("derefine_poles");
  Real h = gpkg->Param<Real>("h");
  Real xt = gpkg->Param<Real>("xt");
  Real alpha = gpkg->Param<Real>("alpha");
  Real x0 = gpkg->Param<Real>("x0");
  Real smooth = gpkg->Param<Real>("smooth");
  auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

  RNGPool rng_pool(seed);

  // Long term it might be nice to just compute adiabats for IdealGas too,
  // once it is exposed in singularity-eos, to unify code
  int nsamps = 180; // TODO move this?
  if (eos_type != "StellarCollapse") {
    nsamps = 1;
  }
  Spiner::DataBox rho_h(nsamps);
  Spiner::DataBox temp_h(nsamps);

  // compute adiabats
  // TODO: transition this to a package.
  const Real rho_min = pmb->packages.Get("eos")->Param<Real>("rho_min");
  const Real rho_max = pmb->packages.Get("eos")->Param<Real>("rho_max");
  const Real lrho_min = std::log10(rho_min);
  const Real lrho_max = std::log10(rho_max);
  const Real T_min = pmb->packages.Get("eos")->Param<Real>("T_min");
  const Real T_max = pmb->packages.Get("eos")->Param<Real>("T_max");
  // Get adiabat databoxes on device

  Real lrho_min_adiabat, lrho_max_adiabat; // rho bounds for adiabat
  Real h_min_sc;
  if (eos_type == "StellarCollapse") {
    GetRhoBounds(eos_h, rho_min, rho_max, T_min, T_max, Ye, S, lrho_min_adiabat,
                 lrho_max_adiabat);
    temp_h.setRange(0, lrho_min_adiabat, lrho_max_adiabat, nsamps);
    const Real rho_min_adiabat = std::pow(10.0, lrho_min_adiabat);
    const Real rho_max_adiabat = std::pow(10.0, lrho_max_adiabat);

    SampleRho(rho_h, lrho_min_adiabat, lrho_max_adiabat, nsamps);
    ComputeAdiabats(rho_h, temp_h, eos_h, Ye, S, T_min, T_max, nsamps);
    h_min_sc = MinEnthalpy(rho_h, temp_h, Ye, eos_h, nsamps);
  }
  Real uphi_rmax;
  const Real hm1_rmax =
      std::exp(log_enthalpy(rmax, 0.5 * M_PI, a, rin, angular_mom, uphi_rmax)) - 1.0;

  Real rho_rmax, u_rmax;
  GetStateFromEnthalpy(eos_h, eos_type_enum, hm1_rmax, rho_h, temp_h, Ye, h_min_sc, kappa,
                       gam, Cv, 1.0, rho_rmax, u_rmax);

  auto rho_d = rho_h.getOnDevice();
  auto temp_d = temp_h.getOnDevice();
  pmb->par_for(
      "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto rng_gen = rng_pool.get_state();
        const Real dx_sub = coords.CellWidthFA(X1DIR, k, j, i) / nsub;
        const Real dy_sub = coords.CellWidthFA(X2DIR, k, j, i) / nsub;
        v(irho, k, j, i) = 0.0;
        v(ieng, k, j, i) = 0.0;
        SPACELOOP(d) { v(ivlo + d, k, j, i) = 0.0; }
        const Real x3 = coords.Xc<3>(k, j, i);
        for (int m = 0; m < nsub; m++) {
          for (int n = 0; n < nsub; n++) {
            const Real x1 = coords.Xf<1>(k, j, i) + (m + 0.5) * dx_sub;
            const Real x2 = coords.Xf<2>(k, j, i) + (n + 0.5) * dy_sub;

            Real r = tr.bl_radius(x1);
            Real th = tr.bl_theta(x1, x2);

            Real lnh = -1.0;
            Real hm1;
            Real uphi;
            if (r > rin) lnh = log_enthalpy(r, th, a, rin, angular_mom, uphi);

            if (lnh > 0.0) {
              Real hm1 = std::exp(lnh) - 1.;
              Real rho, u;
              GetStateFromEnthalpy(eos, eos_type_enum, hm1, rho_d, temp_d, Ye, h_min_sc,
                                   kappa, gam, Cv, rho_rmax, rho, u);

              Real ucon_bl[] = {0.0, 0.0, 0.0, uphi};
              Real gcov[4][4];
              bl.SpacetimeMetric(0.0, r, th, x3, gcov);
              ucon_bl[0] = ucon_norm(ucon_bl, gcov);
              Real ucon[4];
              tr.bl_to_fmks(x1, x2, x3, a, ucon_bl, ucon);
              const Real lapse = geom.Lapse(0.0, x1, x2, x3);
              Real beta[3];
              geom.ContravariantShift(0.0, x1, x2, x3, beta);
              geom.SpacetimeMetric(0.0, x1, x2, x3, gcov);
              ucon[0] = ucon_norm(ucon, gcov);
              const Real W = lapse * ucon[0];
              Real vcon[] = {ucon[1] / W + beta[0] / lapse, ucon[2] / W + beta[1] / lapse,
                             ucon[3] / W + beta[2] / lapse};

              v(irho, k, j, i) += rho / (nsub * nsub);
              v(ieng, k, j, i) += u / (nsub * nsub);
              for (int d = 0; d < 3; d++) {
                v(ivlo + d, k, j, i) += W * vcon[d] / (nsub * nsub);
              }
            }
          }
        }
        const Real x1v = coords.Xc<1>(k, j, i);
        const Real x2v = coords.Xc<2>(k, j, i);

        v(ieng, k, j, i) *= (1. + u_jitter * (rng_gen.drand() - 0.5));

        // fixup
        Real rhoflr = 0;
        Real epsflr;
        floor.GetFloors(x1v, x2v, x3, rhoflr, epsflr);
        Real lambda[2] = {Ye, 0.0};
        if (iye > 0) {
          v(iye, k, j, i) = lambda[0];
        }
        v(irho, k, j, i) = v(irho, k, j, i) < rhoflr ? rhoflr : v(irho, k, j, i);
        v(ieng, k, j, i) = v(ieng, k, j, i) / v(irho, k, j, i) < epsflr
                               ? v(irho, k, j, i) * epsflr
                               : v(ieng, k, j, i);
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            v(irho, k, j, i), v(ieng, k, j, i) / v(irho, k, j, i), lambda);
        v(iprs, k, j, i) = eos.PressureFromDensityTemperature(v(irho, k, j, i),
                                                              v(itmp, k, j, i), lambda);
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);

        Real ucon[4] = {0};
        Real vpcon[3] = {v(ivlo, k, j, i), v(ivlo + 1, k, j, i), v(ivlo + 2, k, j, i)};
        Real gcov[4][4] = {0};
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        GetFourVelocity(vpcon, geom, CellLocation::Cent, k, j, i, ucon);

        // Radiation
        if (do_rad) {

          Real r = tr.bl_radius(x1v);
          Real th = tr.bl_theta(x1v, x2v);
          Real lnh = -1.0;
          Real uphi;
          if (r > rin) lnh = log_enthalpy(r, th, a, rin, angular_mom, uphi);

          // Set radiation inside the disk according to the Fishbone gas solution
          if (lnh > 0.0) {

            // TODO(BRR) only first species right now
            for (int ispec = specB.s; ispec < 1; ispec++) {
              // Given total pressure, calculate temperature such that fluid and radiation
              // pressures sum to total pressure
              // TODO(BRR) Generalize to all neutrino species
              const Real Ptot = v(iprs, k, j, i);
              const Real T = v(itmp, k, j, i);
              const Real Ye = iye > 0 ? v(iye, k, j, i) : 0.5;

              if (init_rad == InitialRadiation::thermal) {
                root_find::RootFind root_find;
                GasRadTemperatureResidual res(v(iprs, k, j, i), v(irho, k, j, i),
                                              opacities, eos, species_d[ispec], Ye);
                v(itmp, k, j, i) = root_find.regula_falsi(res, 0, T, 1.e-6 * T, T);
              }

              // Set fluid u/P/T and radiation J using equilibrium temperature
              Real lambda[2] = {Ye, 0.0};
              v(ieng, k, j, i) =
                  v(irho, k, j, i) * eos.InternalEnergyFromDensityTemperature(
                                         v(irho, k, j, i), v(itmp, k, j, i), lambda);
              v(iprs, k, j, i) = eos.PressureFromDensityTemperature(
                  v(irho, k, j, i), v(itmp, k, j, i), lambda);

              if (init_rad == InitialRadiation::thermal) {
                v(iJ(ispec), k, j, i) = opacities.EnergyDensityFromTemperature(
                    v(itmp, k, j, i), species_d[ispec]);
              } else {
                v(iJ(ispec), k, j, i) = 1.e-5 * v(ieng, k, j, i);
              }

              // Zero comoving frame fluxes
              SPACELOOP(ii) { v(iH(ispec, 0), k, j, i) = 0.; }
            }
          } else {
            // In the atmosphere set some small radiation energy
            for (int ispec = specB.s; ispec < 1; ispec++) {
              v(iJ(ispec), k, j, i) = 1.e-5 * v(ieng, k, j, i);

              // Zero comoving frame fluxes
              SPACELOOP(ii) { v(iH(ispec, 0), k, j, i) = 0.; }
            }
          }
        }

        rng_pool.free_state(rng_gen);
      });
  free(rho_h);
  free(rho_d);
  free(temp_h);
  free(temp_d);

  // get vector potential
  ParArrayND<Real> A("vector potential", jb.e + 1, ib.e + 1);
  pmb->par_for(
      "Phoebus::ProblemGenerator::Torus2", jb.s + 1, jb.e, ib.s + 1, ib.e,
      KOKKOS_LAMBDA(const int j, const int i) {
        const Real rho_av =
            0.25 * (v(irho, kb.s, j, i) + v(irho, kb.s, j, i - 1) +
                    v(irho, kb.s, j - 1, i) + v(irho, kb.s, j - 1, i - 1));
        // JMM: Classic HARM divides by rho_max here, to normalize rho_av.
        // However, we have already normalized rho, and thus rho_av. So
        // we should not renormalize.
        // This will change in the case of realistic EOS, when we
        // can't simply renormalize by density and must instead do
        // something clever with units.
        // const Real q = rho_av / rho_rmax - 0.2;
        const Real q = rho_av - 0.2;
        A(j, i) = (q > 0 ? q : 0.0);
      });

  // Initialize B field lines, to be normalized in PostInitializationModifier
  if (ibhi > 0) {
    pmb->par_for(
        "Phoebus::ProblemGenerator::Torus3", kb.s, kb.e, jb.s, jb.e - 1, ib.s, ib.e - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // JMM: HARM/bhlight divides by gdet, not gamdet.
          // This means the HARM primitives are smaller than the Phoebus
          // primitives by a factor of alpha.
          const Real gamdet = geom.DetGamma(CellLocation::Cent, k, j, i);
          v(iblo, k, j, i) = -(A(j, i) - A(j + 1, i) + A(j, i + 1) - A(j + 1, i + 1)) /
                             (2.0 * coords.CellWidthFA(X2DIR, k, j, i) * gamdet);
          v(iblo + 1, k, j, i) = (A(j, i) + A(j + 1, i) - A(j, i + 1) - A(j + 1, i + 1)) /
                                 (2.0 * coords.CellWidthFA(X1DIR, k, j, i) * gamdet);
          v(ibhi, k, j, i) = 0.0;
        });
  }

  fluid::PrimitiveToConserved(rc);
  if (do_rad) {
    radiation::MomentPrim2Con(rc);
  }

  fixup::ApplyFloors(rc);
}

void ProblemModifier(ParameterInput *pin) {
  Real router = pin->GetOrAddReal("coordinates", "r_outer", 40.0);
  Real x1max = log(router);
  pin->SetReal("parthenon/mesh", "x1max", x1max);

  Real a = pin->GetReal("geometry", "a");
  Real Rh = 1.0 + sqrt(1.0 - a * a);
  Real xh = log(Rh);
  int ninside = pin->GetOrAddInteger("torus", "n_inside_horizon", 5);
  bool cutout = pin->GetOrAddBoolean("torus", "cutout", false);
  int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
  Real x1min;
  if (cutout) {
    int nx1_target = pin->GetInteger("torus", "nx1_target");
    PARTHENON_REQUIRE(nx1_target >= nx1, "nx1_target should be >= nx1");
    Real dx = (x1max - xh) / (nx1_target - ninside);
    x1min = x1max - nx1 * dx;
  } else {
    Real dx = (x1max - xh) / (nx1 - ninside);
    x1min = xh - (ninside + 0.5) * dx;
  }
  pin->SetReal("parthenon/mesh", "x1min", x1min);

  if (parthenon::Globals::my_rank == 0) {
    printf("Torus: Setting inner radius to %g\n"
           "\tThis translates to x1min, x1max = %g %g\n",
           std::exp(x1min), x1min, x1max);
  }

  // Set Cv properly for radiation
  const bool do_rad = pin->GetOrAddBoolean("physics", "rad", false);
  if (do_rad) {
    const std::string eos_type = pin->GetString("eos", "type");
    // TODO: allow other eos
    if (eos_type == "IdealGas") {
      const Real Gamma = pin->GetReal("eos", "Gamma");
      PARTHENON_WARN("Resetting Cv assuming Ye = 0.5!");
      PARTHENON_WARN("Don't currently have neutron mass!");
      const Real mn = pc::mp + pc::me; // Oops no binding energy
      const Real mu = (pc::mp + mn + pc::me) / (3. * pc::mp);
      const Real cv = pc::kb / ((Gamma - 1.) * mu * pc::mp);
      pin->SetReal("eos", "Cv", cv);
    } else {
      PARTHENON_FAIL(
          "eos_type not supported for initializing radiation torus currently!");
    }
  }
}

void PostInitializationModifier(ParameterInput *pin, Mesh *pmesh) {
  const bool magnetized = pin->GetOrAddBoolean("torus", "magnetized", true);
  const Real beta_target = pin->GetOrAddReal("torus", "target_beta", 100.);
  const bool harm_style_beta =
      pin->GetOrAddBoolean("torus", "harm_beta_normalization", true);
  const Real rho_min_bnorm = pin->GetOrAddReal("torus", "rho_min_bnorm", 1.e-4);

  Real beta_min, beta_pmax;
  ComputeBetas(pmesh, rho_min_bnorm, beta_min, beta_pmax);
  if (parthenon::Globals::my_rank == 0) {
    printf("Torus before normalization: beta_min, beta_pmax = %.14e %.14e\n", beta_min,
           beta_pmax);
  }
  const Real beta_norm = harm_style_beta ? beta_pmax : beta_min;
  const Real B_field_fac = magnetized ? std::sqrt(beta_norm / beta_target) : 0;
  if (parthenon::Globals::my_rank == 0) {
    printf("Torus normalization factor = %.14e\n", B_field_fac);
  }

  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    auto geom = Geometry::GetCoordinateSystem(rc.get());

    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    auto v = rc->PackVariables({fluid_prim::bfield}, imap);
    const int iblo = imap[fluid_prim::bfield].first;
    const int ibhi = imap[fluid_prim::bfield].second;

    pmb->par_for(
        "Phoebus::ProblemGenerator::Torus::BFieldNorm", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          for (int ib = iblo; ib <= ibhi; ib++) {
            v(ib, k, j, i) *= B_field_fac;
          }
        });

    fluid::PrimitiveToConserved(rc.get());
  }

  Real beta_min_new, beta_pmax_new;
  ComputeBetas(pmesh, rho_min_bnorm, beta_min_new, beta_pmax_new);
  if (parthenon::Globals::my_rank == 0) {
    printf("Torus after normalization: beta_min, beta_pmax = %.14e %.14e\n", beta_min_new,
           beta_pmax_new);
  }
}

KOKKOS_FUNCTION
Real lfish_calc(Real r, Real a) {
  return (
      ((std::pow(a, 2) - 2. * a * std::sqrt(r) + std::pow(r, 2)) *
       ((-2. * a * r * (std::pow(a, 2) - 2. * a * std::sqrt(r) + std::pow(r, 2))) /
            std::sqrt(2. * a * std::sqrt(r) + (-3. + r) * r) +
        ((a + (-2. + r) * std::sqrt(r)) * (std::pow(r, 3) + std::pow(a, 2) * (2. + r))) /
            std::sqrt(1 + (2. * a) / std::pow(r, 1.5) - 3. / r))) /
      (std::pow(r, 3) * std::sqrt(2. * a * std::sqrt(r) + (-3. + r) * r) *
       (std::pow(a, 2) + (-2. + r) * r)));
}

KOKKOS_FUNCTION
Real log_enthalpy(const Real r, const Real th, const Real a, const Real rin, const Real l,
                  Real &uphi) {
  const Real sth = sin(th);
  const Real cth = cos(th);

  const Real DD = r * r - 2. * r + a * a;
  const Real AA = (r * r + a * a) * (r * r + a * a) - DD * a * a * sth * sth;
  const Real SS = r * r + a * a * cth * cth;

  const Real thin = M_PI / 2.;
  const Real sthin = sin(thin);
  const Real cthin = cos(thin);

  const Real DDin = rin * rin - 2. * rin + a * a;
  const Real AAin =
      (rin * rin + a * a) * (rin * rin + a * a) - DDin * a * a * sthin * sthin;
  const Real SSin = rin * rin + a * a * cthin * cthin;
  uphi = 0.0;
  const Real lnh =
      0.5 * std::log((1. + std::sqrt(1. + 4. * (l * l * SS * SS) * DD /
                                              (AA * AA * sth * sth))) /
                     (SS * DD / AA)) -
      0.5 * std::sqrt(1. + 4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth)) -
      2. * a * r * l / AA -
      (0.5 * std::log((1. + std::sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
                                               (AAin * AAin * sthin * sthin))) /
                      (SSin * DDin / AAin)) -
       0.5 * std::sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
                                (AAin * AAin * sthin * sthin)) -
       2. * a * rin * l / AAin);
  if (lnh > 0.0) {
    Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
    Real up1 = std::sqrt((-1. + std::sqrt(1. + 4. * l * l * expm2chi)) / 2.);
    uphi = 2. * a * r * std::sqrt(1. + up1 * up1) / std::sqrt(AA * SS * DD) +
           std::sqrt(SS / AA) * up1 / sth;
  }
  return lnh;
}

/**
 * cheeky way to make this pgen work for ideal and nuclear EoS
 * There's probably a better way to do this
 **/
KOKKOS_FUNCTION
void GetStateFromEnthalpy(const EOS &eos, const EosType eos_type, const Real hm1,
                          const Spiner::DataBox rho, const Spiner::DataBox temp,
                          const Real Ye, const Real h_min_sc, const Real kappa,
                          const Real gam, const Real Cv, const Real rho_rmax,
                          Real &rho_out, Real &u_out) {
  if (eos_type == IdealGas) { // Ideal Gas
    rho_out = std::pow(hm1 * (gam - 1.) / (kappa * gam), 1. / (gam - 1.));
    u_out = kappa * std::pow(rho_out, gam) / (gam - 1.) / rho_rmax;
    rho_out /= rho_rmax;
  } else { // StellarCollapse
    const int N = rho.size() - 1;
    const Real rho_min = std::pow(10.0, rho(0));
    const Real rho_max = std::pow(10.0, rho(N));

    EnthalpyResidual res_h(hm1, temp, Ye, h_min_sc, eos);
    root_find::RootFind rf;
    const Real guess_h = 0.5 * (rho_max - rho_min);

    rho_out = rf.regula_falsi(res_h, rho_min, rho_max, 1e-8 * guess_h, guess_h);
    Real T = temp.interpToReal(std::log10(rho_out));

    Real lambda[2];
    lambda[0] = Ye;
    u_out = eos.InternalEnergyFromDensityTemperature(rho_out, T, lambda) * rho_out;
  }
}

KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]) {
  Real AA = gcov[0][0];
  Real BB = 2. * (gcov[0][1] * ucon[1] + gcov[0][2] * ucon[2] + gcov[0][3] * ucon[3]);
  Real CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + gcov[2][2] * ucon[2] * ucon[2] +
            gcov[3][3] * ucon[3] * ucon[3] +
            2. * (gcov[1][2] * ucon[1] * ucon[2] + gcov[1][3] * ucon[1] * ucon[3] +
                  gcov[2][3] * ucon[2] * ucon[3]);
  Real discr = BB * BB - 4. * AA * CC;
  if (discr < 0) printf("discr = %g   %g %g %g\n", discr, AA, BB, CC);
  PARTHENON_REQUIRE(discr >= 0, "discr < 0");
  return (-BB - std::sqrt(discr)) / (2. * AA);
}

// TODO(JMM): Should this be elsewhere in Phoebus?
void ComputeBetas(Mesh *pmesh, Real rho_min_bnorm, Real &beta_min_global,
                  Real &beta_pmax) {
  Real beta_min = std::numeric_limits<Real>::infinity();
  Real press_max = -std::numeric_limits<Real>::infinity();
  Real bsq_max = -std::numeric_limits<Real>::infinity();

  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    auto geom = Geometry::GetCoordinateSystem(rc.get());

    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    auto v =
        rc->PackVariables({fluid_prim::density, fluid_prim::velocity, fluid_prim::bfield,
                           fluid_prim::pressure, radmoment_prim::J},
                          imap);

    const int irho = imap[fluid_prim::density].first;
    const int ivlo = imap[fluid_prim::velocity].first;
    const int ivhi = imap[fluid_prim::velocity].second;
    const int iblo = imap[fluid_prim::bfield].first;
    const int ibhi = imap[fluid_prim::bfield].second;
    const int iprs = imap[fluid_prim::pressure].first;

    auto idx_J = imap.GetFlatIdx(radmoment_prim::J, false);

    if (ibhi < 0) return;

    Real beta_min_local;
    pmb->par_reduce(
        "Phoebus::ProblemGenerator::Torus::BFieldNorm::beta_min", kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &beta_min) {
          const Real bsq =
              GetMagneticFieldSquared(CellLocation::Cent, k, j, i, geom, v, ivlo, iblo);
          Real Ptot = v(iprs, k, j, i);
          // TODO(BRR) multiple species
          if (idx_J.IsValid()) {
            int ispec = 0;
            Ptot += 1. / 3. * v(idx_J(ispec), k, j, i);
          }
          const Real beta = robust::ratio(v(iprs, k, j, i), 0.5 * bsq);
          if (v(irho, k, j, i) > rho_min_bnorm && beta < beta_min) beta_min = beta;
        },
        Kokkos::Min<Real>(beta_min_local));
    beta_min = std::min<Real>(beta_min_local, beta_min);

    Real bsq_max_local;
    pmb->par_reduce(
        "Phoebus::ProblemGenerator::Torus::BFieldNorm::bsq_max", kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &bsq_max) {
          const Real bsq =
              GetMagneticFieldSquared(CellLocation::Cent, k, j, i, geom, v, ivlo, iblo);
          bsq_max = std::max(bsq, bsq_max);
        },
        Kokkos::Max<Real>(bsq_max_local));
    bsq_max = std::max<Real>(bsq_max, bsq_max_local);

    Real press_max_local;
    pmb->par_reduce(
        "Phoebus::ProblemGenerator::Torus::BFieldNorm::press_max", kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &P_max) {
          P_max = std::max(v(iprs, k, j, i), P_max);
        },
        Kokkos::Max<Real>(press_max_local));
    press_max = std::max<Real>(press_max, press_max_local);
  }

  beta_min_global = reduction::Min(beta_min);
  const Real bsq_max_global = reduction::Max(bsq_max);
  const Real Pmax_global = reduction::Max(press_max);
  beta_pmax = robust::ratio(Pmax_global, 0.5 * bsq_max_global);
  return;
}

} // namespace torus
