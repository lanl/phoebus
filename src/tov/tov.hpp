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
#include "radiation/light_bulb_constants.hpp"

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

//MG: Maybe it is useful to have this for constructing PNS profile after deleptonization.
  
KOKKOS_INLINE_FUNCTION
void AdiabatThermoFromP(const Real P, Real s, Real nsamps, ParameterInput *pin, StateDescriptor *eospkg, Real &rho, Real &eps){
  using Microphysics::EOS::EOS;
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
  Adiabats::GetRhoBounds(eos_h, rho_min, rho_max, T_min, T_max, Ye, s, lrho_min_adiabat,
			 lrho_max_adiabat);
  Adiabats::SampleRho(rho_h, lrho_min_adiabat, lrho_max_adiabat, nsamps);
  Adiabats::ComputeAdiabats(rho_h, temp_h, eos_h, Ye, s, T_min, T_max, nsamps);

  const Real rho_min_adiabat = std::pow(10.0, lrho_min_adiabat);
  const Real rho_max_adiabat = std::pow(10.0, lrho_max_adiabat);
  auto unit_conv = phoebus::UnitConversions(pin);
  auto target = [&](const Real rho) {
    return eos_h.PressureFromDensityTemperature(rho, temp_h(std::log10(rho)), lambda) - P;

  }; 

  const Real guess0 = (rho_max_adiabat-rho_min_adiabat)/2.0;
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const Real guess = guess0;
  Real rho_adm;
  root_find::RootFind root_find;
  rho = root_find.regula_falsi(target, rho_min, rho_max, epsilon * guess, guess);
  //rho = rho_cgs * density_conversion_factor;
  Real temp_root = temp_h(std::log10(rho));
  eps = eos_h.InternalEnergyFromDensityTemperature(rho, temp_root, lambda);  
}
  


KOKKOS_INLINE_FUNCTION
void PNSThermoFromP(const Real P, Real s, Real nsamps, ParameterInput *pin, StateDescriptor *eospkg, Real &rho, Real &eps){
  using Microphysics::EOS::EOS;
  auto eos_h = eospkg->Param<EOS>("h.EOS");
  Real lambda[2];

    
  auto unit_conv = phoebus::UnitConversions(pin);
  auto temp = [&](const Real rho){
     constexpr Real bh0 = LightBulb::Liebendorfer::BH0;
     constexpr Real bh1 = LightBulb::Liebendorfer::BH1;
     constexpr Real bl0 = LightBulb::Liebendorfer::BL0;
     constexpr Real bl1 = LightBulb::Liebendorfer::BL1;
     constexpr Real bl2 = LightBulb::Liebendorfer::BL2;
     constexpr Real bl3 = LightBulb::Liebendorfer::BL3;
     constexpr Real bl4 = LightBulb::Liebendorfer::BL4;
     constexpr Real bl5 = LightBulb::Liebendorfer::BL5;
     constexpr Real bl6 = LightBulb::Liebendorfer::BL6;
     constexpr Real bl7 = LightBulb::Liebendorfer::BL7;

     constexpr Real logrho_Tmax = LightBulb::Liebendorfer::LOGRHO_TMAX;
     const Real lRho = std::log10(rho);
     const Real lRho2 = lRho * lRho;
     const Real lRho3 = lRho2 * lRho;
     const Real lRho4 = lRho2 * lRho2;
     const Real lRho5 = lRho4 * lRho;
     const Real lRho6 = lRho3 * lRho3;
     const Real lRho7 = lRho4 * lRho3;

     if (lRho > logrho_Tmax){
       return bh0 + bh1 * lRho;
     }
     else{
       return bl0 + bl1 * lRho + bl2* lRho2 + bl3 * lRho3 + bl4 * lRho4 + bl5 * lRho5 + bl6 * lRho6 + bl7 * lRho7;
     }
  };

  auto ye = [&](const Real rho){
     constexpr Real c0 = LightBulb::Liebendorfer::C0;
     constexpr Real c1 = LightBulb::Liebendorfer::C1;
     constexpr Real c2 = LightBulb::Liebendorfer::C2;
     constexpr Real c3 = LightBulb::Liebendorfer::C3;
     constexpr Real c4 = LightBulb::Liebendorfer::C4;
     /*     constexpr Real c5 = LightBulb::Liebendorfer::C5;
     constexpr Real c6 = LightBulb::Liebendorfer::C6;
     constexpr Real c7 = LightBulb::Liebendorfer::C7;
     constexpr Real c8 = LightBulb::Liebendorfer::C8;
     constexpr Real c9 = LightBulb::Liebendorfer::C9;
     constexpr Real c10 = LightBulb::Liebendorfer::C10;
     constexpr Real c11 = LightBulb::Liebendorfer::C11;
     constexpr Real c12 = LightBulb::Liebendorfer::C12;
     constexpr Real c13 = LightBulb::Liebendorfer::C13;
     constexpr Real c14 = LightBulb::Liebendorfer::C14;
     */
     const Real lRho = std::log10(rho);
     const Real lRho2 = lRho * lRho;
     const Real lRho3 = lRho2 * lRho;
     const Real lRho4 = lRho2 * lRho2;
       
     return c0 + c1 * lRho + c2 * lRho2 + c3 * lRho3 + c4 * lRho4;
 
  };
 
  const Real density_conv_factor = unit_conv.GetMassDensityCodeToCGS();
  const Real density_inv_conv_factor = unit_conv.GetMassDensityCGSToCode();
  
  auto target = [&](const Real rho) {
    const Real rho_cgs = rho * density_conv_factor;
    Real temperature;

    if (rho * density_conv_factor < pow(10,10)){
      lambda[0] = 0.5;
      temperature=rho_cgs/pow(10,10)*0.2e11;
    }else{
      lambda[0] = ye(rho_cgs);
      temperature = temp(rho_cgs);
    }
    Real Pcalc = eos_h.PressureFromDensityTemperature(rho, temperature * unit_conv.GetTemperatureCGSToCode(), lambda);
    if (std::abs(Pcalc-P)/P>1e-3){
      return Pcalc - P;
    }else{
      return 0.0;
    }
  }; 

  const Real rho_min = pow(10,9) * density_inv_conv_factor;
  const Real rho_max = 6 * pow(10,14) * density_inv_conv_factor;

  const Real guess0 = (rho_max-rho_min)/2.0;
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const Real guess = guess0;
  Real rho_adm;
  Real T;
  root_find::RootFind root_find;
  rho = root_find.regula_falsi(target, rho_min, rho_max, epsilon * 10 * guess, guess);
  if (rho < rho_min){
    lambda[0] = 0.5;
    T = 1e6;
  }else{
    lambda[0] = ye(rho*density_conv_factor);
    T = temp(rho*density_conv_factor);
  }
  eps = eos_h.InternalEnergyFromDensityTemperature(rho, T*unit_conv.GetTemperatureCGSToCode(), lambda);  
}
  

  
KOKKOS_INLINE_FUNCTION
void TovRHS_StellarCollapse(Real r, const Real in[NTOV], const Real Pmin,
			    Real out[NTOV], Real s, Real nsamps, ParameterInput *pin, StateDescriptor *eospkg) {
  Real m = in[TOV::M];
  Real P = in[TOV::P];
  // Real phi = in[TOV::PHI];
  if (P < Pmin) P = 0;
  Real rho, eps, rho_adm;
  if (P > Pmin){
    PNSThermoFromP(P, s, nsamps, pin, eospkg, rho, eps);
    rho_adm = rho * (1 + eps);
  }
  else{
    rho_adm = 0;
  }
  out[TOV::M] = GetMRHS(r, rho_adm);
  out[TOV::P] = GetPRHS(r, rho_adm, m, P, Pmin);
  out[TOV::PHI] = GetPhiRHS(r, rho_adm, m, P);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus IntegrateTov(StateDescriptor *tovpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg, ParameterInput *pin);

} // namespace TOV

#endif // TOV_TOV_HPP_
