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
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <fstream>
#include <typeinfo>

// Parthenon
#include <globals.hpp>
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

  // Arrays for CCSN stuff
  CCSN::State_t ccsn_state("CCSN state", CCSN::NCCSN, npoints);
  auto ccsn_state_h = Kokkos::create_mirror_view(ccsn_state);

  params.Add("ccsn_state", ccsn_state);
  params.Add("ccsn_state_h", ccsn_state_h);

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

  auto model_filename = params.Get<std::string>("model_filename");
  auto npoints = params.Get<int>("npoints");
  auto radius = monopolepkg->Param<MonopoleGR::Radius>("radius");

  auto matter_h = monopolepkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  auto state_h = params.Get<CCSN::State_host_t>("ccsn_state_h");

  const Real dr = radius.dx();

  // read 1d model
  const int num_zones = Get1DProfileNumZones(model_filename);
  //printf("1D model read in with ",num_zones,"number of zones.");

  //printf("%s:%i model_filename = %s\n", __FILE__, __LINE__, model_filename.c_str());

  // const Real model_1d[9][num_zones] = {{CCSN::Get1DProfileData("model_filename",num_zones)}};

  // printf("1D model read in. Max radius is  = %.14e\n", model_1d[0][num_zones],"with ",0,"number of zones.");

  // allocate state size of NCCSN which should equal num vars for CCSN ?
  //  Real state[NCCSN];

  // interpolate to determine 1d model on monopole GR radial grid
  //Real model_1d_interp = CCSN::Interp1DProfile(model_1d,npoints,radius,dr);

  // second loop, to set density, specific energy, and matter state
  //for (int i = 0; i < npoints; ++i) {

     // some of the vars below will be changed to EOS calls

     // set i'th interpolated values from 1D model	  
     //Real rho = model_1d_interp(CCSN::RHO, i);
     //Real radvel = model_1d_interp(CCSN::V, i);
     //Real eps = model_1d_interp(CCSN::EPS, i);
     //Real ye = model_1d_interp(CCSN::YE, i);
     //Real pres = model_1d_interp(CCSN::P, i);
     //Real temp = model_1d_interp(CCSN::TEMP, i);     

     // set host state vector to interpolated values
     //state_h(CCSN::RHO, i) = rho;
     //state_h(CCSN::V, i) = radvel;
     //state_h(CCSN::EPS, i) = eps;
     //state_h(CCSN::YE, i) = ye;
     //state_h(CCSN::P, i) = pres;
     //state_h(CCSN::TEMP, i) = temp;

     // change to non-static values
     //matter_h(MonopoleGR::Matter::RHO, i) = rho * (1 + eps); // ADM mass
     //matter_h(MonopoleGR::Matter::J_R, i) = 0;               // momentum
     //matter_h(MonopoleGR::Matter::trcS, i) = 3 * press;      // in rest frame of fluid
     //matter_h(MonopoleGR::Matter::Srr, i) = press;
  // }

  // add appropraite print here for CCSN initialization
  // printf("TOV star constructed. Total mass = %.14e\n", state_h(TOV::M, npoints - 1));

  // Copy to device
  // auto matter_d = monopolepkg->Param<MonopoleGR::Matter_t>("matter");
  // auto state_d = params.Get<CCSN::State_t>("ccsn_state");
  // Kokkos::deep_copy(matter_d, matter_h);
  //  Kokkos::deep_copy(state_d, state_h);

  return TaskStatus::complete;
}

} // namespace CCSN
