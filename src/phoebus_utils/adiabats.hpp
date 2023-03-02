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

#ifndef PHOEBUS_UTILS_ADIABATS_HPP_
#define PHOEBUS_UTILS_ADIABATS_HPP_

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include <spiner/databox.hpp>

#include "phoebus_utils/root_find.hpp"

// a namespace?

template <typename EOS>
void GetRhoBounds(const EOS &eos, const Real rho_min, const Real rho_max,
                  const Real T_min, const Real T_max, const Real Ye, const Real S0,
                  Real &lrho_min_new, Real &lrho_max_new) {

  root_find::RootFind root_find;
  root_find::RootFindStatus status;
  Real lambda[2];
  lambda[0] = Ye;

  // lower density bound
  Real guess = 1.1 * rho_min;
  Real T = T_min;
  auto target = [&](const Real Rho) {
    return eos.EntropyFromDensityTemperature(Rho, T, lambda) - S0;
  };
  lrho_min_new =
      root_find.regula_falsi(target, rho_min, rho_max, 1.e-8 * guess, guess, &status);
  if (status == root_find::RootFindStatus::failure) {
    lrho_min_new = rho_min;
  };
  lrho_min_new = std::log10(lrho_min_new);

  // upper density bound
  guess = 0.9 * rho_max;
  T = T_max;
  lrho_max_new =
      root_find.regula_falsi(target, rho_min, rho_max, 1.e-8 * guess, guess, &status);
  if (status == root_find::RootFindStatus::failure) {
    lrho_max_new = rho_max;
  };
  lrho_max_new = std::log10(lrho_max_new);
}

template <typename D>
void SampleRho(D rho, const Real rho_min, const Real rho_max, const int n_samps) {
  const Real drho = (rho_max - rho_min) / n_samps;

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "Adiabats::SampleRho", DevExecSpace(), 0,
      n_samps, KOKKOS_LAMBDA(const int i) { rho(i) = rho_min + i * drho; });
}

template <typename D, typename EOS>
void ComputeAdiabats(D rho, D temp, const EOS &eos, const Real Ye, const Real S0,
                     const Real T_min, const Real T_max, const int n_samps) {

  const Real guess0 = (T_max - T_min) / 2.0;

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "Adiabats::ComputeAdiabats", DevExecSpace(),
      0, n_samps, KOKKOS_LAMBDA(const int i) {
        const Real Rho = std::pow(10.0, rho(i));
        Real lambda[2];
        lambda[0] = Ye;

        auto target = [&](const Real T) {
          return eos.EntropyFromDensityTemperature(Rho, T, lambda) - S0;
        };

        const Real guess = guess0;
        root_find::RootFind root_find;
        temp(i) = root_find.regula_falsi(target, T_min, T_max, 1.e-10 * guess, guess);
      });
}

/**
 * Find the minimum enthalpy along as adiabat as computed above
 **/
template <typename D, typename EOS>
Real MinEnthalpy(D rho, D temp, const Real Ye, const EOS &eos, const int n_samps) {
  Real min_h = 0.0;
  Real min_enthalpy = 1e30;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Adiabats::MinEnthalpy", DevExecSpace(), 0,
      n_samps, 0, n_samps,
      KOKKOS_LAMBDA(const int i, const int j, Real &min_enthalpy) {
        const Real Rho = std::pow(10.0, rho(i));
        const Real T = temp(i);
        Real lambda[2];
        lambda[0] = Ye;

        auto enthalpy_sc = [&]() {
          const Real P = eos.PressureFromDensityTemperature(Rho, T, lambda);
          const Real e = eos.InternalEnergyFromDensityTemperature(Rho, T, lambda);
          return 1.0 + e + P / Rho;
        };

        const Real h = enthalpy_sc();
        min_enthalpy = (h < min_enthalpy) ? h : min_enthalpy;
      },
      Kokkos::Min<Real>(min_h));
  return min_h;
}
#endif // PHOEBUS_UTILS_ADIABATS_HPP_
