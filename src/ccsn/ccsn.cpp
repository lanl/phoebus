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
#include "monopole_gr/monopole_gr_base.hpp"
#include "monopole_gr/monopole_gr_interface.hpp"
#include "monopole_gr/monopole_gr_utils.hpp"
#include "phoebus_utils/initial_model_reader.hpp"

#include "ccsn.hpp"

namespace CCSN {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ccsn = std::make_shared<StateDescriptor>("ccsn");
  Params &params = ccsn->AllParams();

  bool do_ccsn = pin->GetOrAddBoolean("ccsn", "enabled", false);
  params.Add("enabled", do_ccsn);
  if (!do_ccsn) return ccsn; // short-circuit with nothing

  bool do_monopole_gr = pin->GetOrAddBoolean("monopole_gr", "enabled", false);
  if (!do_monopole_gr) {
    std::stringstream msg;
    msg << "Currently MonopoleGR must be enabled for CCSN." << std::endl;
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

  // fixed paramater here
  //  Real pc = pin->GetOrAddReal("tov", "Pc", 10);
  //  params.Add("pc", pc);

  // Arrays for CCSN stuff
  CCSN::State_t ccsn_state("CCSN state", CCSN::NCCSN, npoints);
  auto ccsn_state_h = Kokkos::create_mirror_view(ccsn_state);

  CCSN::State_t ccsn_intrinsic("CCSN intrinsic vars", CCSN::NINTRINSIC, npoints);
  auto ccsn_intrinsic_h = Kokkos::create_mirror_view(ccsn_intrinsic);

  params.Add("ccsn_state", ccsn_state);
  params.Add("ccsn_state_h", ccsn_state_h);
  params.Add("ccsn_intrinsic", ccsn_intrinsic);
  params.Add("ccsn_intrinsic_h", ccsn_intrinsic_h);

  return ccsn;
}

// this task will change to something that generates the ccsn initial problem
TaskStatus InitializeCCSN(StateDescriptor *ccsnpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg) {
  PARTHENON_REQUIRE_THROWS(ccsnpkg->label() == "ccsn", "Requires the ccsn package");
  PARTHENON_REQUIRE_THROWS(monopolepkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  PARTHENON_REQUIRE_THROWS(eospkg->label() == "eos", "Requires the eos package");

  auto &params = ccsnpkg->AllParams();

  auto ccsn_enabled = ccsnpkg->Param<bool>("enabled");
  if (!ccsn_enabled) return TaskStatus::complete;
  auto monopole_enabled = monopolepkg->Param<bool>("enable_monopole_gr");
  if (!monopole_enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = monopolepkg->Param<MonopoleGR::Radius>("radius");

  auto alpha = monopolepkg->Param<MonopoleGR::Alpha_t>("lapse");
  auto alpha_h = monopolepkg->Param<MonopoleGR::Alpha_host_t>("lapse_h");
  Kokkos::deep_copy(alpha_h, alpha);

  auto matter_h = monopolepkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  auto state_h = params.Get<CCSN::State_host_t>("ccsn_state_h");
  auto intrinsic_h = params.Get<CCSN::State_host_t>("ccsn_intrinsic_h");

  // assumed model_1d loaded in hpp?
  const Real model_1d
  const Real dr = radius.dx();

  // set center value to BC if necessary
  // state_h(CCSN::M, 0) = 0;
  // state_h(CCSN::P, 0) = 1;
  // state_h(CCSN::PHI, 0) = std::log(alpha_h(0));

  // allocate state size of NCCSN which should equal num vars for CCSN ?
  Real state[NCCSN];
  Real k1[NCCSN];
  Real rhs[NCCSN];
  Real rhs_k[NCCSN];

  // cef: remove pressure loop because given in model_1d? or would this be new pressure after some solve?

  // first loop, to solve for pressure
  for (int i = 0; i < npoints - 1; ++i) {
    Real r = radius.x(i);
#pragma omp simd
    for (int v = 0; v < NCCSN; ++v) {
      state[v] = state_h(v, i);
    }
    TovRHS(r, state, K, Gamma, Pmin, rhs);
#pragma omp simd
    for (int v = 0; v < NCCSN; ++v) {
      k1[v] = state[v] + 0.5 * dr * rhs[v];
    }
    TovRHS(r, k1, K, Gamma, Pmin, rhs_k);
#pragma omp simd
    for (int v = 0; v < NCCSN; ++v) {
      state_h(v, i + 1) = state_h(v, i) + dr * rhs_k[v];
    }
  }

  // second loop, to set density, specific energy, and matter state
  for (int i = 0; i < npoints; ++i) {
    Real mass = state_h(CCSN::M, i);
    Real press = state_h(CCSN::P, i);
    Real rho, eps;
    // needed for analytic solution. Bad for actual code.
    if (press <= 1.1 * Pmin) {
      press = rho = eps = 0;
    } else {
      PolytropeThermoFromP(press, K, Gamma, rho, eps);
    }
    intrinsic_h(CCSN::RHO0, i) = rho;
    intrinsic_h(CCSN::EPS, i) = eps;
    matter_h(MonopoleGR::Matter::RHO, i) = rho * (1 + eps); // ADM mass
    matter_h(MonopoleGR::Matter::J_R, i) = 0;               // momentum
    matter_h(MonopoleGR::Matter::trcS, i) = 3 * press;      // in rest frame of fluid
    matter_h(MonopoleGR::Matter::Srr, i) = press;
  }
  // add appropraite print here for CCSN initialization
  // printf("TOV star constructed. Total mass = %.14e\n", state_h(TOV::M, npoints - 1));

  // Copy to device
  auto matter_d = monopolepkg->Param<MonopoleGR::Matter_t>("matter");
  auto state_d = params.Get<CCSN::State_t>("ccsn_state");
  auto intrinsic_d = params.Get<CCSN::State_t>("ccsn_intrinsic");
  Kokkos::deep_copy(matter_d, matter_h);
  Kokkos::deep_copy(state_d, state_h);
  Kokkos::deep_copy(intrinsic_d, intrinsic_h);

  return TaskStatus::complete;
}

} // namespace CCSN
