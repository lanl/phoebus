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

// stdlib
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Singularity
#include <singularity-eos/eos/eos.hpp>

// Phoebus
#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "monopole_gr/monopole_gr_utils.hpp"

#include "tov.hpp"

namespace TOV {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tov = std::make_shared<StateDescriptor>("tov");
  Params &params = tov->AllParams();

  bool do_tov = pin->GetOrAddBoolean("TOV", "enabled", false);
  params.Add("enabled", do_tov);
  if (!do_tov) return tov; // short-circuit with nothing

  bool do_monopole_gr = pin->GetOrAddBoolean("monopole_gr", "enabled", false);
  if (!do_monopole_gr) {
    std::stringstream msg;
    msg << "Currently MonopoleGR must be enabled for TOV." << std::endl;
    PARTHENON_THROW(msg);
  }

  // Points. Note this is the same as for the MonopoleGR package.
  int npoints = pin->GetOrAddInteger("monopole_gr", "npoints", 100);
  {
    using MonopoleGR::MIN_NPOINTS;

    std::stringstream msg;
    msg << "npoints must be at least " << MIN_NPOINTS << std::endl;
    PARTHENON_REQUIRE_THROWS(npoints >= MIN_NPOINTS, msg);
  }
  params.Add("npoints", npoints);

  // Central pressure
  Real pc = pin->GetOrAddReal("TOV", "Pc", 10);
  params.Add("pc", pc);

  // Pressure floor
  Real pmin = pin->GetOrAddReal("TOV", "Pmin", 1e-9);
  params.Add("pmin", pmin);

  // Etnropy
  Real s = pin->GetOrAddReal("TOV", "entropy", 8);
  params.Add("entropy", s);

  // Arrays for TOV stuff
  TOV::State_t tov_state("TOV state", TOV::NTOV, npoints);
  auto tov_state_h = Kokkos::create_mirror_view(tov_state);

  TOV::State_t tov_intrinsic("TOV intrinsic vars", TOV::NINTRINSIC, npoints);
  auto tov_intrinsic_h = Kokkos::create_mirror_view(tov_intrinsic);

  params.Add("tov_state", tov_state);
  params.Add("tov_state_h", tov_state_h);
  params.Add("tov_intrinsic", tov_intrinsic);
  params.Add("tov_intrinsic_h", tov_intrinsic_h);

  return tov;
}

TaskStatus IntegrateTov(StateDescriptor *tovpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg) {
  PARTHENON_REQUIRE_THROWS(tovpkg->label() == "tov", "Requires the tov package");
  PARTHENON_REQUIRE_THROWS(monopolepkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  PARTHENON_REQUIRE_THROWS(eospkg->label() == "eos", "REquires the eos package");

  auto &params = tovpkg->AllParams();

  auto tov_enabled = tovpkg->Param<bool>("enabled");
  if (!tov_enabled) return TaskStatus::complete;
  auto monopole_enabled = monopolepkg->Param<bool>("enable_monopole_gr");
  if (!monopole_enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = monopolepkg->Param<MonopoleGR::Radius>("radius");

  auto alpha = monopolepkg->Param<MonopoleGR::Alpha_t>("lapse");
  auto alpha_h = monopolepkg->Param<MonopoleGR::Alpha_host_t>("lapse_h");
  Kokkos::deep_copy(alpha_h, alpha);

  auto matter_h = monopolepkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  auto state_h = params.Get<TOV::State_host_t>("tov_state_h");
  auto intrinsic_h = params.Get<TOV::State_host_t>("tov_intrinsic_h");

  auto pc = params.Get<Real>("pc");
  auto s = params.Get<Real>("entropy");
  auto Pmin = params.Get<Real>("pmin");
  auto gm1 = eospkg->Param<Real>("gm1");
  const Real Gamma = gm1 + 1;
  const Real K = PolytropeK(s, Gamma);
  const Real dr = radius.dx();

  state_h(TOV::M, 0) = 0;
  state_h(TOV::P, 0) = pc;
  state_h(TOV::PHI, 0) = std::log(alpha_h(0));

  Real state[NTOV];
  Real k1[NTOV];
  Real rhs[NTOV];
  Real rhs_k[NTOV];

  // first loop, to solve for pressure
  for (int i = 0; i < npoints - 1; ++i) {
    Real r = radius.x(i);
#pragma omp simd
    for (int v = 0; v < NTOV; ++v) {
      state[v] = state_h(v, i);
    }
    TovRHS(r, state, K, Gamma, Pmin, rhs);
#pragma omp simd
    for (int v = 0; v < NTOV; ++v) {
      k1[v] = state[v] + 0.5 * dr * rhs[v];
    }
    TovRHS(r, k1, K, Gamma, Pmin, rhs_k);
#pragma omp simd
    for (int v = 0; v < NTOV; ++v) {
      state_h(v, i + 1) = state_h(v, i) + dr * rhs_k[v];
    }
  }

  // second loop, to set density, specific energy, and matter state
  for (int i = 0; i < npoints; ++i) {
    Real mass = state_h(TOV::M, i);
    Real press = state_h(TOV::P, i);
    Real rho, eps;
    if (press <= 1.1 * Pmin) {
      press = rho = eps = 0;
    } else {
      PolytropeThermoFromP(press, K, Gamma, rho, eps);
    }
    intrinsic_h(TOV::RHO0, i) = rho;
    intrinsic_h(TOV::EPS, i) = eps;
    matter_h(MonopoleGR::Matter::RHO, i) = rho * (1 + eps); // ADM mass
    matter_h(MonopoleGR::Matter::J_R, i) = 0;               // momentum
    matter_h(MonopoleGR::Matter::trcS, i) = 3 * press;      // in rest frame of fluid
    matter_h(MonopoleGR::Matter::Srr, i) = press;
  }

  // Copy to device
  auto matter_d = monopolepkg->Param<MonopoleGR::Matter_t>("matter");
  auto state_d = params.Get<TOV::State_t>("tov_state");
  auto intrinsic_d = params.Get<TOV::State_t>("tov_intrinsic");
  Kokkos::deep_copy(matter_d, matter_h);
  Kokkos::deep_copy(state_d, state_h);
  Kokkos::deep_copy(intrinsic_d, intrinsic_h);

  return TaskStatus::complete;
}

} // namespace TOV
