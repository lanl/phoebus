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
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <utility>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <parthenon_array_generic.hpp>
#include <utils/error_checking.hpp>

// Singularity
#include <singularity-eos/eos/eos.hpp>

// Phoebus
#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr/monopole_gr_base.hpp"
#include "monopole_gr/monopole_gr_interface.hpp"
#include "monopole_gr/monopole_gr_utils.hpp"

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

  std::string model_filename = pin->GetOrAddString("ccsn", "model_filename", "file.txt");
  params.Add("model_filename", model_filename);

  // Pressure floor
  Real pmin = pin->GetOrAddReal("ccsn", "Pmin", 1e-9);
  params.Add("pmin", pmin);

  // Mass density floor
  Real rhomin = pin->GetOrAddReal("ccsn", "rhomin", 1e-12);
  params.Add("rhomin", rhomin);

  // Internal energy floor
  Real epsmin = pin->GetOrAddReal("ccsn", "epsmin", 1e-12);
  params.Add("epsmin", epsmin);

  // read 1d model
  std::pair<int, int> p = Get1DProfileNumZones(model_filename);
  const int num_zones = p.first - 1;
  const int num_comments = p.second;
  const int num_vars = 9;

  params.Add("num_zones", num_zones);
  params.Add("num_comments", num_comments);
  params.Add("num_vars", num_vars);

  printf("1D model read in with %d number of zones. \n", num_zones);
  printf("1D model read in with %d number of comments. \n", num_comments);

  return ccsn;
}

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

  const std::string model_filename = params.Get<std::string>("model_filename");
  auto npoints = params.Get<int>("npoints");
  auto radius = monopolepkg->Param<MonopoleGR::Radius>("radius");
  const Real dr = radius.dx();

  auto num_zones = params.Get<int>("num_zones");
  auto num_comments = params.Get<int>("num_comments");
  auto num_vars = params.Get<int>("num_vars");

  auto Pmin = params.Get<Real>("pmin");

  // create raw array
  CCSN::State_t ccsn_state_raw_d("CCSN state raw", CCSN::NCCSN, num_zones);

  CCSN::State_host_t ccsn_state_raw_h = Kokkos::create_mirror_view(ccsn_state_raw_d);

  params.Add("ccsn_state_raw_d", ccsn_state_raw_d);
  params.Add("ccsn_state_raw_h", ccsn_state_raw_h);

  // create 1d array for interp data
  CCSN::State_t ccsn_state_interp_d("CCSN state interp", CCSN::NCCSN, npoints);

  CCSN::State_host_t ccsn_state_interp_h =
      Kokkos::create_mirror_view(ccsn_state_interp_d);

  params.Add("ccsn_state_interp_d", ccsn_state_interp_d);
  params.Add("ccsn_state_interp_h", ccsn_state_interp_h);

  // get matter info for monopole solver
  auto matter_h = monopolepkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  auto state_h = params.Get<CCSN::State_host_t>("ccsn_state_interp_h");

  // printf("Current working dir: %s\n", get_current_dir_name());

  printf("My name filename requested is %s.\n", model_filename.c_str());

  // pass 1d model array and fill from text file
  Get1DProfileData(model_filename, num_zones, num_comments, num_vars, ccsn_state_raw_d);

  /////

  Real input_radius[num_zones];

  // make array for MESA radius
  for (int i = 0; i < num_zones; ++i) {
    Real r = ccsn_state_raw_d(CCSN::R, i);
    input_radius[i] = r;
  }

  // set interp radial grid to monopole grid  - THIS should be a one liner??
  for (int i = 0; i < npoints; i++) {
    Real r = radius.x(i);
    ccsn_state_interp_d(CCSN::R, i) = r;
    ccsn_state_interp_h(CCSN::R, i) = r;
  }

  //// fill interpolated data onto device array
  for (int i = 0; i < npoints; i++) {

    // some of the vars below will be changed to EOS calls
    Real x = radius.x(i);
    int jlo = // call to some routine to get int if needed

    // add if then for if jlo = 0 or num_zones?
    for (int j = 1; j < num_vars; j++) {

      // xx[jlo::jlo+mm]
      auto xa = ccsn_state_raw_d.Slice(CCSN::R, std::make_pair(jlo, jlo + 3));

      // yy[jlo::jlo+mm]
      auto ya = ccsn_state_raw_d.Slice(j, std::make_pair(jlo, jlo + 3));

      Real yint = //  call to some interp routine  

      ccsn_state_interp_d(j, i) = yint;
    }
  }

  // add appropraite print here for CCSN initialization
  printf("MESA model interpolated to monopoleGR rgrid. Central density = %.14e (g/cc)\n",
         ccsn_state_interp_d(CCSN::RHO, 0));
  printf(
      "MESA model interpolated to monopoleGR rgrid. Central electron fraction = %.14e\n",
      ccsn_state_interp_d(CCSN::YE, 0));
  printf("MESA model interpolated to monopoleGR rgrid. Central pressure = %.14e "
         "(dyn/cm**2)\n",
         ccsn_state_interp_d(CCSN::P, 0));
  printf(
      "MESA model interpolated to monopoleGR rgrid. Central temperature = %.14e (GK)\n",
      ccsn_state_interp_d(CCSN::TEMP, 0));

  // Copy to device
  Kokkos::deep_copy(ccsn_state_interp_d, ccsn_state_interp_h);

  // save interpolated output for analysis
  DumpToTxtInterpModel1D(model_filename, ccsn_state_interp_h, npoints);

  // second loop, to set density, specific energy, and matter state
  for (int i = 0; i < npoints; ++i) {
    Real rho = ccsn_state_interp_h(CCSN::RHO, i);
    Real press = ccsn_state_interp_h(CCSN::P, i);
    Real eps = ccsn_state_interp_h(CCSN::EPS, i);

    // no floor for now, and using input thermo
    if (press <= 1.1 * Pmin) {
      press = rho = eps = 0;
    } else {
      // PolytropeThermoFromP(press, K, Gamma, rho, eps);
    }
    // these eqns will change?
    matter_h(MonopoleGR::Matter::RHO, i) = rho * (1 + eps); // ADM mass
    matter_h(MonopoleGR::Matter::J_R, i) = 0;               // momentum
    matter_h(MonopoleGR::Matter::trcS, i) = 3 * press;      // in rest frame of fluid
    matter_h(MonopoleGR::Matter::Srr, i) = press;
  }

  // create matter array on device
  auto matter_d = monopolepkg->Param<MonopoleGR::Matter_t>("matter");

  // copy ccsn state to device
  auto state_d = params.Get<CCSN::State_t>("ccsn_state_interp_d");

  // copy filled arrays to device
  Kokkos::deep_copy(matter_d, matter_h);
  Kokkos::deep_copy(state_d, state_h);

  return TaskStatus::complete;
}

} // namespace CCSN
