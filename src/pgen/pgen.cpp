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

#include "pgen.hpp"

namespace phoebus {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  std::string name = pin->GetString("phoebus", "problem");

  if (name == "phoebus" || pgen_dict.count(name) == 0) {
    std::stringstream s;
    s << "Invalid problem name in input file.  Valid options include:" << std::endl;
    for (const auto &p : pgen_dict) {
      if (p.first != "phoebus") s << "   " << p.first << std::endl;
    }
    PARTHENON_THROW(s);
  }

  auto f = pgen_dict[name];
  f(pmb, pin);
}

void ProblemModifier(ParameterInput *pin) {
  std::string name = pin->GetString("phoebus", "problem");
  if (name == "phoebus" || pmod_dict.count(name) == 0) {
    return;
  }

  auto f = pmod_dict[name];
  f(pin);
}

void PostInitializationModifier(ParameterInput *pin, Mesh *pmesh) {
  std::string name = pin->GetString("phoebus", "problem");
  if (name == "phoebus" || pinitmod_dict.count(name) == 0) {
    return;
  }

  auto f = pinitmod_dict[name];
  f(pin, pmesh);
}

KOKKOS_FUNCTION
Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P) {
  PARTHENON_REQUIRE(P >= 0, "Pressure is negative!");

  Real eguessl = P/rho;
  Real Pguessl = eos.PressureFromDensityInternalEnergy(rho, eguessl);
  Real eguessr = eguessl;
  Real Pguessr = Pguessl;
  while (Pguessl > P) {
    eguessl /= 2.0;
    Pguessl = eos.PressureFromDensityInternalEnergy(rho, eguessl);
  }
  while (Pguessr < P) {
    eguessr *= 2.0;
    Pguessr = eos.PressureFromDensityInternalEnergy(rho, eguessr);
  }

  PARTHENON_REQUIRE(Pguessr>P && Pguessl<P, "Pressure not bracketed");

  while (Pguessr - Pguessl > 1.e-10*P) {
    Real emid = 0.5*(eguessl + eguessr);
    Real Pmid = eos.PressureFromDensityInternalEnergy(rho, emid);
    if (Pmid < P) {
      eguessl = emid;
      Pguessl = Pmid;
    } else {
      eguessr = emid;
      Pguessr = Pmid;
    }
  }
  return 0.5*rho*(eguessl + eguessr);
}

}
