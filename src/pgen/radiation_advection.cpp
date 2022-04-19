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

#include <cmath>
#include <string>

#include "pgen/pgen.hpp"
#include "phoebus_utils/programming_utils.hpp"

namespace radiation_advection {

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;

constexpr Real rho0_cgs = 1.e10;
constexpr Real T0_cgs = 1.e10;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"advection\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      std::vector<std::string>({radmoment_prim::J, radmoment_prim::H, fluid_prim::density,
                                fluid_prim::temperature, fluid_prim::energy,
                                fluid_prim::velocity, radmoment_internal::xi,
                                radmoment_internal::phi}),
      imap);

  auto idJ = imap.GetFlatIdx(radmoment_prim::J);
  auto idH = imap.GetFlatIdx(radmoment_prim::H);
  auto idv = imap.GetFlatIdx(fluid_prim::velocity);
  auto ixi = imap.GetFlatIdx(radmoment_internal::xi);
  auto iphi = imap.GetFlatIdx(radmoment_internal::phi);
  const int prho = imap[fluid_prim::density].first;
  const int pT = imap[fluid_prim::temperature].first;
  const int peng = imap[fluid_prim::energy].first;

  Params &phoebus_params = pmb->packages.Get("phoebus")->AllParams();

  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  const auto opac_d =
      pmb->packages.Get("opacity")->template Param<singularity::neutrinos::Opacity>(
          "d.opacity");
  auto &unit_conv =
      pmb->packages.Get("eos")->Param<phoebus::UnitConversions>("unit_conv");
  auto &constants =
      pmb->packages.Get("eos")->Param<phoebus::CodeConstants>("code_constants");
  const Real MASS = unit_conv.GetMassCGSToCode();
  const Real LENGTH = unit_conv.GetLengthCGSToCode();
  const Real RHO = unit_conv.GetMassDensityCGSToCode();
  const Real TEMP = unit_conv.GetTemperatureCGSToCode();

  const auto specB = idJ.GetBounds(1);
  const bool scale_free = pin->GetOrAddBoolean("units", "scale_free", true);
  const Real J0 = pin->GetOrAddReal("radiation_advection", "J", 1.0);
  const Real Hx = pin->GetOrAddReal("radiation_advection", "Hx", 0.0);
  const Real Hy = pin->GetOrAddReal("radiation_advection", "Hy", 0.0);
  const Real Hz = pin->GetOrAddReal("radiation_advection", "Hz", 0.0);
  const Real vx = pin->GetOrAddReal("radiation_advection", "vx", 0.0);
  PARTHENON_REQUIRE(std::fabs(vx) < 1., "Subluminal velocity required!");
  const Real width = pin->GetOrAddReal("radiation_advection", "width", sqrt(2.0));
  const Real kappa = pin->GetReal("opacity", "gray_kappa") * LENGTH * LENGTH / MASS;
  const bool boost = pin->GetOrAddBoolean("radiation_advection", "boost_profile", false);
  const int shapedim = pin->GetOrAddInteger("radiation_advection", "shapedim", 1);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real rho0;
  Real T0;
  if (scale_free) {
    rho0 = 1.;
    T0 = 1.;
    PARTHENON_REQUIRE(programming::soft_equiv(pin->GetReal("eos", "Cv"), 1.0),
                      "Specific heat is incorrect!");
  } else {
    rho0 = rho0_cgs * RHO;
    T0 = T0_cgs * TEMP;
    PARTHENON_REQUIRE(
        programming::soft_equiv(pin->GetReal("eos", "Cv"),
                                (pin->GetReal("eos", "Gamma") - 1.) * pc::kb / pc::mp),
        "Specific heat is incorrect!");
  }

  // Store runtime parameters for output
  phoebus_params.Add("radiation_advection/J0", J0);
  phoebus_params.Add("radiation_advection/Hx", Hx);
  phoebus_params.Add("radiation_advection/Hy", Hy);
  phoebus_params.Add("radiation_advection/Hz", Hz);

  auto rad = pmb->packages.Get("radiation").get();
  auto species = rad->Param<std::vector<singularity::RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  singularity::RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  const Real W = 1 / sqrt(1 - vx * vx);
  const Real t0p = 1.5 * kappa * width * width;
  const Real t0 = t0p;
  const Real x0p = (0.5 - vx * t0) * W;
  pmb->par_for(
      "Phoebus::ProblemGenerator::radiation_advection", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.x1v(i);
        Real y = (ndim > 1 && shapedim > 1) ? coords.x2v(j) : 0;
        Real z = (ndim > 2 && shapedim > 2) ? coords.x3v(k) : 0;
        Real r = std::sqrt(x * x + y * y + z * z);

        v(prho, k, j, i) = rho0;
        v(pT, k, j, i) = T0;
        Real lambda[2] = {0.};
        v(peng, k, j, i) =
            v(prho, k, j, i) * eos.InternalEnergyFromDensityTemperature(
                                   v(prho, k, j, i), v(pT, k, j, i), lambda);

        v(idv(0), k, j, i) = vx;
        v(idv(1), k, j, i) = 0.0;
        v(idv(2), k, j, i) = 0.0;

        // Write down boosted diffusion initial condition
        Real tp = W * (t0 - vx * x);
        Real xp = W * (x - vx * t0);
        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {

          Real J;
          if (scale_free) {
            J = J0;
          } else {
            J = opac_d.EnergyDensityFromTemperature(T0, species_d[ispec]);
          }

          J = J0;

          v(ixi(ispec), k, j, i) = 0.0;
          v(iphi(ispec), k, j, i) = acos(-1.0) * 1.000001;

          if (boost) {
            v(idJ(ispec), k, j, i) = std::max(
                J * sqrt(t0p / tp) * exp(-3 * kappa * std::pow(xp - x0p, 2) / (4 * tp)),
                1.e-10 * J);
          } else {
            v(idJ(ispec), k, j, i) =
                std::max(J * exp(-std::pow((x - 0.5) / width, 2) / 2.0), 1.e-10 * J);
          }

          PARTHENON_DEBUG_REQUIRE(std::fabs(Hx) < J && std::fabs(Hy) < J &&
                                      std::fabs(Hz) < J,
                                  "Fluxes are incorrectly normalized!");

          v(idH(0, ispec), k, j, i) = Hx;
          v(idH(1, ispec), k, j, i) = Hy;
          v(idH(2, ispec), k, j, i) = Hz;
          printf("[%i %i %i] J = %e H = %e %e %e\n", k, j, i, v(idJ(ispec), k, j, i),
                 v(idH(0, ispec), k, j, i), v(idH(1, ispec), k, j, i),
                 v(idH(2, ispec), k, j, i));
        }
      });

  // Initialize samples
  auto radpkg = pmb->packages.Get("radiation");
  if (radpkg->Param<bool>("active")) {
    if (radpkg->Param<std::string>("method") == "mocmc") {
      radiation::MOCMCInitSamples(rc.get());
    }
  }

  radiation::MomentPrim2Con(rc.get(), IndexDomain::interior);
}

void ProblemModifier(ParameterInput *pin) {
  const std::string method = pin->GetOrAddString("radiation", "method", "None");
  if (method == "monte_carlo" || method == "mocmc") {
    pin->SetBoolean("units", "scale_free", false);
    pin->SetPrecise("units", "geom_length_cm", 1.e10);
    pin->SetPrecise("units", "fluid_mass_g", 1.e40);
  }

  auto unit_conv = phoebus::UnitConversions(pin);
  const Real LENGTH = unit_conv.GetLengthCodeToCGS();
  const bool scale_free = pin->GetOrAddBoolean("units", "scale_free", true);
  printf("scale_free: %i\n", scale_free);
  const Real rho0 = scale_free ? 1. : rho0_cgs;

  const Real Gamma = pin->GetReal("eos", "Gamma");
  const Real cv = scale_free ? 1. : (Gamma - 1.) * pc::kb / pc::mp;
  printf("cv: %e\n", cv);
  pin->SetPrecise("eos", "Cv", cv);
  printf("now cv: %e\n", pin->GetReal("eos", "Cv"));

  const Real dx1 = (pin->GetReal("parthenon/mesh", "x1max") -
                    pin->GetReal("parthenon/mesh", "x1min")) *
                   LENGTH;

  // Optical depth over the entire X1 simulation region
  const Real tau = pin->GetOrAddReal("radiation_advection", "tau", 1.e3);

  const Real kappa = tau / (rho0 * dx1);
  printf("tau: %e rho0: %e dx1: %e\n", tau, rho0, dx1);

  pin->SetPrecise("opacity", "gray_kappa", kappa);
}

} // namespace radiation_advection
