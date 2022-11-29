// © 2022. Triad National Security, LLC. All rights reserved.  This
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

namespace hohlraum {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"hohlraum\" requires \"Minkowski\" geometry!");

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
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  const Real MASS = unit_conv.GetMassCGSToCode();
  const Real LENGTH = unit_conv.GetLengthCGSToCode();
  const Real RHO = unit_conv.GetMassDensityCGSToCode();
  const Real TEMP = unit_conv.GetTemperatureCGSToCode();

  const auto specB = idJ.GetBounds(1);
  const bool scale_free = pin->GetOrAddBoolean("units", "scale_free", true);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real rho0 = 1.;
  Real T0 = 1.;

  auto rad = pmb->packages.Get("radiation").get();
  auto species = rad->Param<std::vector<singularity::RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");

  pmb->par_for(
      "Phoebus::ProblemGenerator::hohlraum", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(prho, k, j, i) = rho0;
        v(pT, k, j, i) = T0;
        Real lambda[2] = {0.};
        v(peng, k, j, i) =
            v(prho, k, j, i) * eos.InternalEnergyFromDensityTemperature(
                                   v(prho, k, j, i), v(pT, k, j, i), lambda);

        SPACELOOP(ii) {
          v(idv(ii), k, j, i) = 0.0;
        }

        // Write down void initial condition
        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {
          v(ixi(ispec), k, j, i) = 0.0;
          v(iphi(ispec), k, j, i) = acos(-1.0) * 1.000001;
          v(idJ(ispec), k, j, i) = 1.e-50;
          SPACELOOP(ii) {
            v(idH(ispec, ii), k, j, i) = 0.0;
          }
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

} // namespace radiation_advection
