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

#include "phoebus_utils/root_find.hpp"

// a namespace?
using singularity::EOS;

template <typename D>
void SampleRho(D rho, const Real rho_min, const Real rho_max, const int n_samps) {
  const Real drho = (rho_max - rho_min) / n_samps;

  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "Adiabats::SampleRho", DevExecSpace(), 0, n_samps, 
      KOKKOS_LAMBDA(const int i) {
        rho(i) = rho_min + i * drho;
      });
}

template <typename D>
void ComputeAdiabat(D rho, D temp, EOS &eos, const Real Ye, const Real S0, 
    const Real T_min, const Real T_max, const int n_samps) {


  const Real guess0 = (T_max - T_min) / 2.0;

  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "Adiabats::ComputeAdiabats", DevExecSpace(), 0, n_samps, 
      KOKKOS_LAMBDA(const int i) {

        const Real rho = Rho(i);
        Real lambda[2]; 
        lambda[0] = Ye;

        Real target = [&](const Real T) {
          return eos.EntropyFromDensityTemperature(rho, T, lambda) - S0;
        };

        const Real guess = guess0;
        root_find::RootFind root_find;
        temp(i) = root_find.regula_falsi(target, T_min, T_max, 1.e-6 * guess, guess);
      });
}
#endif // PHOEBUS_UTILS_ADIABATS_HPP_
