#include <complex>
#include <sstream>

#include "pgen/pgen.hpp"

// Relativistic hydro linear modes.

using std::complex;

namespace LinearModes {

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

  const Real gam = pin->GetReal("eos", "Gamma");
  //const Real cv  = pin->GetReal("eos", "Cv");

  const std::string mode = pin->GetOrAddString("hydro_modes", "mode", "entropy");
  const Real amp = pin->GetReal("hydro_modes", "amplitude");

  double rho0 = 1.0;
  double ug0 = 1.e-2;

  // Wavevector
  double k1 = 2.*M_PI;

  complex<double> omega, drho, dug, du1;
  double u10 = 0.;
  if (mode == "entropy") {
    omega = complex<double>(0, 2.*M_PI/10.);
    //omega = complex<double>(0, 2.*M_PI/1.);
    drho = 1.;
    dug = 0.;
    du1 = 0.;
    u10 = 0.1; // Uniform advection
  } else if (mode == "sound") {
    omega = complex<double>(0., 0.6568547144496073);
    drho = 0.9944432913027026;
    dug = 0.0165740548550451;
    du1 = -0.10396076706483988;
  } else {
    std::stringstream msg;
    msg << "Mode " << mode << " not supported!" << std::endl;
    PARTHENON_FAIL(msg);
  }
  pin->SetReal("parthenon/time", "tlim", 2.*M_PI/omega.imag()); // Set final time to be one period

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
    "Phoebus::ProblemGenerator::Hydro_Modes", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);

      double mode = amp*cos(k1*x);

      double rho = rho0 + (drho*mode).real();
      v(irho, k, j, i) = rho;
      double ug = ug0 + (dug*mode).real();
      double Pg = (gam - 1.)*ug;
      v(iprs, k, j, i) = Pg;
      v(ieng, k, j, i) = ug;
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho); // this doesn't have to be exact, just a reasonable guess
      for (int d = 0; d < 3; d++) v(ivlo+d, k, j, i) = 0.0;
      v(ivlo, k, j, i) = u10 + (du1*mode).real();
    });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace LinearModes
