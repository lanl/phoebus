#include "pgen/pgen.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "radiation/radiation.hpp"
#include "utils/constants.hpp"

// Optically thin cooling.
// As descriged in the bhlight test suite
// Ryan, B. R., Dolence, J. C., & Gammie, C. F. 2015, ApJ, 807, 31. doi:10.1088/0004-637X/807/1/31

namespace thin_cooling {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({"p.density",
                              "p.velocity",
                              "p.energy",
                              "pressure",
                              "temperature",
                              "gamma1",
                              "cs"},
                              imap);

  const int irho = imap["p.density"].first;
  const int ivlo = imap["p.velocity"].first;
  const int ivhi = imap["p.velocity"].second;
  const int ieng = imap["p.energy"].first;
  const int iprs = imap["pressure"].first;
  const int itmp = imap["temperature"].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eospkg = pmb->packages.Get("eos");
  auto eos = eospkg->Param<singularity::EOS>("d.EOS");
  auto &unit_conv = eospkg.get()->Param<phoebus::UnitConversions>("unit_conv");
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  const Real gam = rad->Param<Real>("Gamma");

  const Real mu = 0.5;

  const Real ne_cgs =  pin->GetOrAddReal("thincooling", "ne0", 5.858732e+07);
  const Real rho_cgs = ne_cgs*pc.mp;
  const Real T_cgs = pin->GetOrAddReal("thincooling", "T0", 1.e8);
  const Real P_cgs = rho_cgs*pc.kb*T_cgs/(mu*pc.mp);
  const Real sie_cgs = (P_cgs/(gam - 1.))/rho_cgs;
  printf("CGS  ne: %e rho: %e T: %e P: %e sie: %e\n", ne_cgs, rho_cgs, T_cgs, P_cgs, sie_cgs);
  const Real cv_cgs =  pc.kb/((gam - 1)*(mu *pc.mp));
  double sie_cgs_eos = eos.InternalEnergyFromDensityTemperature(rho_cgs, T_cgs);
  double P_cgs_eos = eos.PressureFromDensityInternalEnergy(rho_cgs, sie_cgs_eos);
  printf(" cv: %e EOS sie: %e P: %e\n", cv_cgs, sie_cgs_eos, P_cgs_eos);

  const Real RHO = unit_conv.GetMassDensityCodeToCGS();
  const Real ENERGY = unit_conv.GetEnergyCodeToCGS();
  const Real DENSITY = unit_conv.GetNumberDensityCodeToCGS();
  const Real TEMP = unit_conv.GetTemperatureCodeToCGS();

  const Real ne_code = ne_cgs*unit_conv.GetNumberDensityCGSToCode();
  const Real rho_code = rho_cgs*unit_conv.GetMassDensityCGSToCode();
  const Real T_code = T_cgs*unit_conv.GetTemperatureCGSToCode();
  const Real P_code = P_cgs*unit_conv.GetEnergyCGSToCode()*unit_conv.GetNumberDensityCGSToCode();
  const Real sie_code = sie_cgs*unit_conv.GetEnergyCGSToCode()/unit_conv.GetMassCGSToCode();
  printf("CODE ne: %e rho: %e T: %e P: %e sie: %e\n", ne_code, rho_code, T_code, P_code, sie_code);
  const Real cv_code = cv_cgs*unit_conv.GetEnergyCGSToCode()/(unit_conv.GetMassCGSToCode()*unit_conv.GetTemperatureCGSToCode());
  double sie_code_eos = eos.InternalEnergyFromDensityTemperature(rho_code, T_code);
  double P_code_eos = eos.PressureFromDensityInternalEnergy(rho_code, sie_code_eos);
  double T_code_eos = eos.TemperatureFromDensityInternalEnergy(rho_code, sie_code_eos);
  printf(" cv: %e EOS sie: %e P: %e T: %e(%e)\n", cv_code, sie_code_eos, P_code_eos, T_code_eos,
    T_code_eos*TEMP);

  //exit(-1);

  const Real ne0 = pin->GetOrAddReal("thincooling", "ne0", 5.858732e+07);
  const Real rho0 = ne0*pc.mp*unit_conv.GetMassDensityCGSToCode();
  const Real T0 = pin->GetOrAddReal("thincooling", "T0", 1.e8)*unit_conv.GetTemperatureCGSToCode();
  double cv = pc.kb/((gam - 1)*(mu *pc.mp));
  printf("cv: %e\n", cv);
  cv *= unit_conv.GetEnergyCGSToCode()/unit_conv.GetTemperatureCGSToCode()/unit_conv.GetMassCGSToCode();
  printf("cv: %e\n", cv);

  printf("T0: %e\n", T0);
  printf("T0: %e K\n", T0*unit_conv.GetTemperatureCodeToCGS());

  double P0 = ne0*pc.kb*T0*unit_conv.GetTemperatureCodeToCGS();
  printf("P0: %e\n", P0);

  printf("Why is T != sie/Cv?\n");
//  exit(-1);

  pmb->par_for(
    "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      v(irho, k, j, i) = rho0;
      v(iprs, k, j, i) = eos.PressureFromDensityTemperature(rho0, T0);
      v(ieng, k, j, i) = phoebus::energy_from_rho_P(eos, rho0, v(iprs, k, j, i));
//Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P) {
      //v(ieng, k, j, i) = eos.InternalEnergyFromDensityTemperature(rho0, T0);

      //v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho0, v(ieng, k, j, i));
//Real energy_from_rho_P(const singularity::EOS &eos, const Real rho, const Real P) {
//Real IdealGas::PressureFromDensityTemperature(const Real rho, const Real temp,
      v(itmp, k, j, i) = T0;

      double rho1 = 3.0;
      double T1 = 2.5;
      double P1 = eos.PressureFromDensityTemperature(rho1, T1);
      double ug1 = phoebus::energy_from_rho_P(eos, rho1, P1);
      double sie1 = eos.InternalEnergyFromDensityTemperature(rho1, T1);
      printf("ug1 = %e rho1*sie1 = %e\n", ug1, rho1*sie1);


      printf("[%i] rho: %e ug: %e Pg: %e T0: %e\n", i, rho0, v(ieng,k,j,i), v(iprs,k,j,i),
        v(itmp,k,j,i));
      double sie = eos.InternalEnergyFromDensityTemperature(rho0, T0);
      printf("     u: %e rho*sie: %e\n", v(ieng,k,j,i), rho0*sie);

      /*Real lambda, rho1, sie1;
      eos.DensityEnergyFromPressureTemperature(v(iprs,k,j,i), v(itmp,k,j,i), &lambda, rho1, sie1);
      printf("rho1: %e sie1: %e\n", rho1, sie1);

      printf("new T: %e\n", eos.TemperatureFromDensityInternalEnergy(rho0, v(ieng, k, j, i)));

      printf("P: %e\n", v(iprs, k, j, i)*ENERGY*DENSITY);

      Real rho_cgs = rho0*RHO;
      Real ne_cgs = radiation::GetNumberDensity(rho_cgs);
      printf("ne_cgs: %e\n", ne_cgs);*/

      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
    });
//  exit(-1);

  fluid::PrimitiveToConserved(rc.get());

  fluid::ConservedToPrimitive(rc.get());
  printf("NAEIWUHDAISUHDS\n");
  pmb->par_for(
    "Phoebus::ProblemGenerator::ThinCooling", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      printf("[%i] rho: %e ug: %e Pg: %e T0: %e\n", i, rho0, v(ieng,k,j,i), v(iprs,k,j,i),
        v(itmp,k,j,i));

      printf("new T: %e\n", eos.TemperatureFromDensityInternalEnergy(rho0, v(ieng, k, j, i)));

      printf("P: %e\n", v(iprs, k, j, i)*ENERGY*DENSITY);

      Real rho_cgs = rho0*RHO;
      Real ne_cgs = radiation::GetNumberDensity(rho_cgs);
      printf("ne_cgs: %e\n", ne_cgs);
    });

}

}

//} // namespace phoebus
