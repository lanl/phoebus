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

#ifndef TOV_TOV_HPP_
#define TOV_TOV_HPP_

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "phoebus_utils/adiabats.hpp"

namespace TOV {
constexpr int NTOV = 3;

using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int M = 0;
constexpr int P = 1;
constexpr int PHI = 2;

constexpr int NINTRINSIC = 2;

constexpr int RHO0 = 0;
constexpr int EPS = 1;

/*
 * Only valid for polytropic EOS's
 */
KOKKOS_INLINE_FUNCTION
void PolytropeThermoFromP(const Real P, const Real K, const Real Gamma, Real &rho,
                          Real &eps) {
  rho = std::pow(P / K, 1 / Gamma);
  eps = (K / (Gamma - 1)) * std::pow(P / K, (Gamma - 1) / Gamma);
}

KOKKOS_INLINE_FUNCTION
Real PolytropeK(const Real s, const Real Gamma) { return std::exp(s * (Gamma - 1)); }

KOKKOS_INLINE_FUNCTION
Real GetMRHS(const Real r, const Real rho_adm) { return 4 * M_PI * r * r * rho_adm; }

KOKKOS_INLINE_FUNCTION
Real GetPhiRHS(const Real r, const Real rho_adm, const Real m, const Real P) {
  if (r == 0) return 0;
  return (m + 4 * M_PI * r * r * r * P) / (r * (r - 2 * m));
}

KOKKOS_INLINE_FUNCTION
Real GetPRHS(const Real r, const Real rho_adm, const Real m, const Real P,
             const Real Pmin) {
  if (P < Pmin) return 0;
  Real phirhs = GetPhiRHS(r, rho_adm, m, P);
  return -(rho_adm + P) * phirhs;
}

KOKKOS_INLINE_FUNCTION
void TovRHS(Real r, const Real in[NTOV], const Real K, const Real Gamma, const Real Pmin,
            Real out[NTOV]) {
  Real m = in[TOV::M];
  Real P = in[TOV::P];
  // Real phi = in[TOV::PHI];
  if (P < Pmin) P = 0;

  Real rho, eps;
  // TODO(JMM): THIS ASSUMES A POLYTROPIC EOS.
  // More generality requires exposing entropy in singularity-eos
  PolytropeThermoFromP(P, K, Gamma, rho, eps);
  Real rho_adm = rho * (1 + eps);
  out[TOV::M] = GetMRHS(r, rho_adm);
  out[TOV::P] = GetPRHS(r, rho_adm, m, P, Pmin);
  out[TOV::PHI] = GetPhiRHS(r, rho_adm, m, P);
}

  
KOKKOS_INLINE_FUNCTION
void TovRHS_StellarCollapse(Real r, const Real in[NTOV], const Real Pmin,
			    Real out[NTOV], Real s, Real nsamps, ParameterInput *pin, StateDescriptor *eospkg) {
  Real m = in[TOV::M];
  Real P = in[TOV::P];
  // Real phi = in[TOV::PHI];
  if (P < Pmin) P = 0;

  using Microphysics::EOS::EOS;
  auto eos = eospkg->Param<EOS>("d.EOS");
  auto eos_h = eospkg->Param<EOS>("h.EOS");

  using DataBox = Spiner::DataBox<Real>;
  DataBox rho_h(nsamps);
  DataBox temp_h(nsamps);
  Real Ye = 0.5;
  Real lambda[2];
  lambda[0] = Ye;

  const Real rho_min = eospkg->Param<Real>("rho_min");
  const Real rho_max = eospkg->Param<Real>("rho_max");
  const Real lrho_min = std::log10(rho_min);
  const Real lrho_max = std::log10(rho_max);
  const Real T_min = eospkg->Param<Real>("T_min");                                                                         
  const Real T_max = eospkg->Param<Real>("T_max");
  Real lrho_min_adiabat, lrho_max_adiabat;
  Adiabats::GetRhoBounds(eos, rho_min, rho_max, T_min, T_max, Ye, s, lrho_min_adiabat,
			 lrho_max_adiabat);
  Adiabats::SampleRho(rho_h, lrho_min_adiabat, lrho_max_adiabat, nsamps);
  Adiabats::ComputeAdiabats(rho_h, temp_h, eos_h, Ye, s, T_min, T_max, nsamps);

  const Real rho_min_adiabat = std::pow(10.0, lrho_min_adiabat);
  const Real rho_max_adiabat = std::pow(10.0, lrho_max_adiabat);
  auto unit_conv = phoebus::UnitConversions(pin);
  const Real density_conversion_factor = 6.6835442e-18;
  const Real press_conversion_factor = 1.3447284e+38;

  auto target = [&](const Real rho) {
    return eos.PressureFromDensityTemperature(rho, temp_h(std::log10(rho)), lambda) - P*press_conversion_factor;
    }; 

  const Real guess0 = (rho_max_adiabat-rho_min_adiabat)/2.0;
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const Real guess = guess0;
  root_find::RootFind root_find;
  Real rho = root_find.regula_falsi(target, rho_min, rho_max, epsilon * guess, guess);

  Real rho_code = rho * density_conversion_factor;
  Real temp_root = temp_h(std::log10(rho));
  Real eps = eos.InternalEnergyFromDensityTemperature(rho, temp_root, lambda)/(press_conversion_factor*density_conversion_factor);
  Real rho_adm = rho_code * (1 + eps);
  out[TOV::M] = GetMRHS(r, rho_adm);
  out[TOV::P] = GetPRHS(r, rho_adm, m, P, Pmin);
  out[TOV::PHI] = GetPhiRHS(r, rho_adm, m, P);
  std::cout<<"r="<<r<<" "<<"P="<<P<<" "<<"rho_adm="<<rho_adm<<" "<<"MRHS="<<GetMRHS(r, rho_adm)<<std::endl;

}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus IntegrateTov(StateDescriptor *tovpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg, ParameterInput *pin);

} // namespace TOV

#endif // TOV_TOV_HPP_
