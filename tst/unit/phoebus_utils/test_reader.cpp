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

// stdlib includes
#include <cmath>
#include <memory>
#include <string>
#include <vector>
// external includes
#include "catch2/catch.hpp"
#include <Kokkos_Core.hpp>

// parthenon includes
#include <coordinates/coordinates.hpp>
#include <defs.hpp>
#include <kokkos_abstraction.hpp>
#include <parameter_input.hpp>

// Test utils
#include "test_utils.hpp"

// Relativity utils
#include "phoebus_utils/relativity_utils.hpp"

// Ascii Reader
#include "progenitor/ascii_reader.hpp"

using parthenon::Real;

TEST_CASE("READER", "[reader_utils]") {
  std::vector<Real> mydata[11];
  AsciiReader::readtable("../../tst/unit/phoebus_utils/fake_table.dat", mydata);
  const int npoints = mydata[0].size();
  int n_wrong = 0;
  double rho = mydata[AsciiReader::COLUMNS::rho][0];
  double sie = mydata[AsciiReader::COLUMNS::eps][0];
  double press = mydata[AsciiReader::COLUMNS::press][0];
  double temp = mydata[AsciiReader::COLUMNS::temp][0];
  double ye = mydata[AsciiReader::COLUMNS::ye][0];
  for (int i = 1; i < npoints; ++i) {
    if (mydata[AsciiReader::COLUMNS::temp][i] - temp > 1.e-5) {
      n_wrong += 1;
    }
    if (mydata[AsciiReader::COLUMNS::ye][i] - ye > 1.e-5) {
      n_wrong += 1;
    }
    if (mydata[AsciiReader::COLUMNS::rho][i] * mydata[AsciiReader::COLUMNS::eps][i] -
            press >
        1.e-5) {
      n_wrong += 1;
    }
    temp = mydata[AsciiReader::COLUMNS::temp][i];
    ye = mydata[AsciiReader::COLUMNS::ye][i];
    rho = mydata[AsciiReader::COLUMNS::rho][i];
    sie = mydata[AsciiReader::COLUMNS::eps][i];
  }
  REQUIRE(n_wrong == 0);
}
