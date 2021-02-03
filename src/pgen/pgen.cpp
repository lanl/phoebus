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

KOKKOS_FUNCTION
Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P) {
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
