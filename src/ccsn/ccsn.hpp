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

#ifndef CCSN_CCSN_HPP_
#define CCSN_CCSN_HPP_

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

namespace CCSN {
constexpr int NCCSN = 11;

// NVAR:NUM_ZONES
using parthenon::Real;
using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int R = 0;
constexpr int RHO = 1;
constexpr int TEMP = 2;
constexpr int YE = 3;
constexpr int EPS = 4;
constexpr int VEL_RAD = 5;
constexpr int PRES = 6;
constexpr int RHO_ADM = 7;
constexpr int J_ADM = 8;
constexpr int S_ADM = 9;
constexpr int Srr_ADM = 10;

KOKKOS_INLINE_FUNCTION
std::pair<int, int> Get1DProfileNumZones(const std::string model_filename) {

  // open file
  std::ifstream inputfile(model_filename);
  const std::string whitespace(" \t\n\r");

  // error check
  if (!inputfile.is_open()) {
    std::cout << model_filename << " not found :( \n.";
    exit(-1);
  }

  int nz = 0;
  int nc = 0;
  std::string line;

  // get number of zones from 1d file
  while (!inputfile.eof()) {

    getline(inputfile, line);

    std::size_t first_nonws = line.find_first_not_of(whitespace);

    // skip empty lines
    if (first_nonws == std::string::npos) {
      continue;
    }

    // skip c++ comments
    if (line.find("//") == first_nonws) {
      nc++;
      continue;
    }

    nz++;
  }

  inputfile.close();

  return std::make_pair(nz, nc);
}

template <typename ParArray2D>
KOKKOS_INLINE_FUNCTION void Get1DProfileData(const std::string model_filename,
                                             const int num_zones, const int num_comments,
                                             const int num_vars,
                                             ParArray2D &ccsn_state_raw_h) {

  std::ifstream inputfile(model_filename);

  Real val = 0;
  std::string line;

  // read file into model_1d
  for (int i = 0; i < num_zones + num_comments; i++) // number of zones
  {
    for (int j = 0; j < num_vars; j++) //  number of vars
    {
      inputfile >> val;
      ccsn_state_raw_h(j, i - num_comments) = val;
    }
  }

  std::cout << "Read 1D profile into state array.\n";

  inputfile.close();
}

template <typename ParArray2D>
KOKKOS_INLINE_FUNCTION void DumpToTxtInterpModel1D(const std::string model_filename,
                                                   ParArray2D &ccsn_state_interp_h,
                                                   const int npoints) {

  FILE *pf;
  std::string out = model_filename;
  out.std::string::append("_interp");
  pf = fopen(out.c_str(), "w");
  fprintf(pf, "# r \t rho \t vel_rad \t eint \t ye \t pres \t temp \t grav \t entr \n");
  for (int i = 0; i < npoints; ++i) {
    Real r = ccsn_state_interp_h(CCSN::R, i);
    fprintf(pf, "%.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e\n", r,
            ccsn_state_interp_h(CCSN::RHO, i), ccsn_state_interp_h(CCSN::V, i),
            ccsn_state_interp_h(CCSN::EPS, i), ccsn_state_interp_h(CCSN::YE, i),
            ccsn_state_interp_h(CCSN::P, i), ccsn_state_interp_h(CCSN::TEMP, i),
            ccsn_state_interp_h(CCSN::grav, i), ccsn_state_interp_h(CCSN::entr, i));
  }
  fclose(pf);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus InitializeCCSN(StateDescriptor *ccsnpkg, StateDescriptor *monopolepkg,
                          StateDescriptor *eospkg);

} // namespace CCSN

#endif // CCSN_CCSN_HPP_
