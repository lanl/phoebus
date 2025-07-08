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

#include "pgen/pgen.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "radiation/radiation.hpp"
#include "utils/constants.hpp"

// Optically thin neutrino cooling.
// As described in the nubhlight test suite
// Miller, J. M., Ryan, B. R., & Dolence, J. C. 2019, ApJS, 241, 30
// doi:10.3847/1538-4365/ab09fc

namespace thin_cooling {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<p::density, p::velocity, p::energy,
                        p::bfield, p::ye, p::pressure, 
                        p::temperature, p::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eospkg = pmb->packages.Get("eos");
  auto eos = eospkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");

  const Real rho0 = 1.e6 * unit_conv.GetMassDensityCGSToCode();
  const Real u0 =
      1.e20 * unit_conv.GetEnergyCGSToCode() * unit_conv.GetNumberDensityCGSToCode();

  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const std::string rad_method = pin->GetString("radiation", "method");

  const bool do_nu_e = pin->GetBoolean("radiation", "do_nu_electron");
  const bool do_nu_ebar = pin->GetBoolean("radiation", "do_nu_electron_anti");
  PARTHENON_REQUIRE(do_nu_e != do_nu_ebar,
                    "Thincooling only supports nu_e or nu_e_bar neutrinos, not both.");

  // set Ye based on radiation type
  const Real Ye0 = (do_nu_e) ? 0.5 : 0.0;
  /*if (x1max > 1.e-7 && rad_method == "cooling_function") {
    PARTHENON_THROW("Set x1max = 1.e-7 for the cooling_function rad method to get small "
                    "enough timesteps!");
  }*/

  pmb->par_for(
      "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        Real lambda[2] = {Ye0, 0.};
        if (v.Contains(0, p::ye())) {
          v(0, p::ye(), k, j, i) = lambda[0];
        }

        v(0, p::density(), k, j, i) = rho0;
        v(0, p::energy(), k, j, i) = u0;
        v(0, p::temperature(), k, j, i) =
            eos.TemperatureFromDensityInternalEnergy(rho0, u0 / rho0, lambda);
        v(0, p::pressure(), k, j, i) = eos.PressureFromDensityInternalEnergy(rho0, u0 / rho0, lambda);
        v(0, p::gamma1(), k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(0, p::density(), k, j, i), v(0, p::temperature(), k, j, i), lambda) /
                           v(0, p::pressure(), k, j, i);
        v(0, p::ye(), k, j, i) = Ye0;

        for (int d = 0; d < 3; d++)
          v(0, p::velocity(d), k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace thin_cooling
