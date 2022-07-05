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
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
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
  PARTHENON_DEBUG_REQUIRE(std::fabs(Hx) < J0 && std::fabs(Hy) < J0 && std::fabs(Hz) < J0,
                          "Fluxes are incorrectly normalized!");
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

  Real rho0 = 1.;
  Real T0 = 1.;

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
          v(ixi(ispec), k, j, i) = 0.0;
          v(iphi(ispec), k, j, i) = acos(-1.0) * 1.000001;

          if (boost) {
            v(idJ(ispec), k, j, i) =
                J0 * std::max(sqrt(t0p / tp) *
                                  exp(-3 * kappa * std::pow(xp - x0p, 2) / (4 * tp)),
                              1.e-10);
          } else {
            v(idJ(ispec), k, j, i) =
                J0 * std::max(exp(-std::pow((x - 0.5) / width, 2) / 2.0), 1.e-10);
          }

          v(idH(ispec, 0), k, j, i) = Hx;
          v(idH(ispec, 1), k, j, i) = Hy;
          v(idH(ispec, 2), k, j, i) = Hz;
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
  auto unit_conv = phoebus::UnitConversions(pin);
  const Real rho0 = 1. * unit_conv.GetMassDensityCodeToCGS();

  const Real dx1 = (pin->GetReal("parthenon/mesh", "x1max") -
                    pin->GetReal("parthenon/mesh", "x1min")) *
                   unit_conv.GetLengthCodeToCGS();

  // Optical depth over the entire X1 simulation region
  const Real tau = pin->GetOrAddReal("radiation_advection", "tau", 1.e3);

  const Real kappa = tau / (rho0 * dx1);

  pin->SetReal("opacity", "gray_kappa", kappa);
}

} // namespace radiation_advection
