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

#include "geometry/geometry.hpp"

#include "pgen.hpp"
#include "phoebus_utils/root_find.hpp"

namespace phoebus {

using Microphysics::EOS::EOS;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  std::string name = pin->GetString("phoebus", "problem");

  if (name == "phoebus" || pgen_dict.count(name) == 0) {
    std::stringstream s;
    s << "Invalid problem name in input file.  Valid options include:" << std::endl;
    for (const auto &p : pgen_dict) {
      if (p.first != "phoebus") s << "   " << p.first << std::endl;
    }
    PARTHENON_THROW(s);
  }

  auto f = pgen_dict[name];
  f(pmb, pin);
}

void ProblemModifier(ParameterInput *pin) {
  std::string name = pin->GetString("phoebus", "problem");
  if (name == "phoebus" || pmod_dict.count(name) == 0) {
    return;
  }

  auto f = pmod_dict[name];
  f(pin);
}

void PostInitializationModifier(ParameterInput *pin, Mesh *pmesh) {
  std::string name = pin->GetString("phoebus", "problem");
  if (name == "phoebus" || pinitmod_dict.count(name) == 0) {
    return;
  }

  auto f = pinitmod_dict[name];
  f(pin, pmesh);
}

class PressResidual {
 public:
  KOKKOS_INLINE_FUNCTION
  PressResidual(const EOS &eos, const Real rho, const Real P, const Real Ye)
      : eos_(eos), rho_(rho), P_(P) {
    lambda_[0] = Ye;
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real e) {
    return eos_.PressureFromDensityInternalEnergy(rho_, e, lambda_) - P_;
  }

 private:
  const EOS &eos_;
  Real rho_, P_;
  Real lambda_[2];
};

KOKKOS_FUNCTION
Real energy_from_rho_P(const EOS &eos, const Real rho, const Real P, const Real emin,
                       const Real emax, const Real Ye) {
  PARTHENON_REQUIRE(P >= 0, "Pressure is negative!");
  PressResidual res(eos, rho, P, Ye);
  root_find::RootFind root;
  Real eroot = root.regula_falsi(res, emin, emax, 1.e-10 * P, emin - 1.e10);
  return rho * eroot;
}

class MachResidual {
 public:
  KOKKOS_INLINE_FUNCTION
  MachResidual(const EOS &eos, const Real rho, const Real vr0, const Real target_mach,
               const Real Ye)
      : eos_(eos), rho_(rho), vr0_(vr0), target_mach_(target_mach) {
    lambda_[0] = Ye;
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real T) {
    Real P = eos_.PressureFromDensityTemperature(rho_, T, lambda_);
    Real eps = eos_.InternalEnergyFromDensityTemperature(rho_, T, lambda_);
    Real bmod = eos_.BulkModulusFromDensityTemperature(rho_, T, lambda_);
    Real u = rho_ * eps;           // convert eps / V to specific internal energy
    Real w = rho_ + P + u;         // h = 1 + eps + P/rho | w = rho * h == rho + u + P
    Real cs = std::sqrt(bmod / w); // cs^2 = bmod / w
    Real mach = vr0_ / cs;         // radial component of preshock velocity
    // printf("Mach Residual Func produced: u, w, cs, Mach, vr0 = %g %g %g %g %g\n", u, w,
    // cs, mach, vr0_);
    return mach - target_mach_;
  }

 private:
  const EOS &eos_;
  Real rho_, vr0_, target_mach_;
  Real lambda_[2];
};

KOKKOS_FUNCTION
Real temperature_from_rho_mach(const EOS &eos, const Real rho, const Real target_mach,
                               const Real Tmin, const Real Tmax, const Real vr0,
                               const Real Ye) {
  // printf("passing values of: rho, vr0, Mach_target, Ye = %g %g %g %g\n", rho, vr0,
  // target_mach, Ye);
  MachResidual res(eos, rho, vr0, target_mach, Ye);
  root_find::RootFind root;
  Real Troot =
      root.regula_falsi(res, Tmin, Tmax, 1.e-10 * target_mach, std::max(Tmin, 1.e-10));
  return Troot;
}

} // namespace phoebus
