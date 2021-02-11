//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include "radiation.hpp"
#include "phoebus_utils/variables.hpp"

namespace radiation {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
  auto physics = std::make_shared<StateDescriptor>("radiation");

  Params &params = physics->AllParams();

  const Real Gamma = pin->GetReal("eos", "Gamma");
  params.Add("Gamma", Gamma);

  std::string method = pin->GetString("radiation", "method");
  params.Add("method", method);

  return physics;
}

constexpr double mu = 0.5;
TaskStatus CalculateRadiationForce(MeshBlockData<Real> *rc, const double dt) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({p::density, p::temperature, c::energy});
  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int ptemp = imap[p::temperature].first;
  const int ceng = imap[c::energy].first;
//  const int gm1 = imap[p::gamma1].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  const Real gam = rad->Param<Real>("Gamma");

  double L = unit_conv.GetLengthCodeToCGS();
  double RHO = unit_conv.GetMassDensityCodeToCGS();
  printf("M: %e L: %e T: %e\n", unit_conv.GetMassCodeToCGS(), L, unit_conv.GetTimeCodeToCGS());
  printf("rho: %e\n", RHO);
  printf("E: %e T: %e\n", unit_conv.GetEnergyCodeToCGS(), unit_conv.GetTemperatureCodeToCGS());

  constexpr double N = 5.4e-39; // cgs

  double tf = 1.e8;
  double T0 = 1.e8;
  double ne = pc.h*sqrt(T0)/((gam - 1.)*M_PI*N*tf);
  double rho = ne*pc.mp;
  printf("ne: %e rho: %e\n", ne, rho);

  const Real TEMP = unit_conv.GetTemperatureCodeToCGS();

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "CalculateRadiationForce", DevExecSpace(),
    kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      double ndens = GetNumberDensity(v(prho, k, j, i)*RHO);
      double Lambda = 4.*M_PI*pc.kb*pow(ndens,2)*N/pc.h*pow(v(ptemp, k, j, i), 1./2.);
      //printf("gm1: %e\n", gam);
      double cv = pc.kb/(pc.mp*(gam - 1.));
      //printf("cv: %e\n", cv);
      printf("[%i] T: %e (%e) E: %e\n", i, v(ptemp, k, j, i), v(ptemp, k, j, i)*TEMP, v(ceng, k, j, i));
      v(ceng, k, j, i) -= v(ceng, k, j, i)*dt;
    });

  exit(-1);

  return TaskStatus::complete;
}

} // namespace radiation
