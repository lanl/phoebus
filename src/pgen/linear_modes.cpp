// #include <complex>
#include <Kokkos_Complex.hpp>
#include <sstream>

#include "pgen/pgen.hpp"

// Relativistic hydro linear modes.

using Kokkos::complex;

namespace linear_modes {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({"p.density",
                              "p.velocity",
                              "p.energy",
                              primitive_variables::bfield,
                              "pressure",
                              "temperature",
                              "gamma1",
                              "cs"},
                              imap);

  const int irho = imap["p.density"].first;
  const int ivlo = imap["p.velocity"].first;
  const int ivhi = imap["p.velocity"].second;
  const int ieng = imap["p.energy"].first;
  const int ib_lo = imap[primitive_variables::bfield].first;
  const int ib_hi = imap[primitive_variables::bfield].second;
  const int iprs = imap["pressure"].first;
  const int itmp = imap["temperature"].first;
  const int nv = ivhi - ivlo + 1;

  const Real gam = pin->GetReal("eos", "Gamma");
  //const Real cv  = pin->GetReal("eos", "Cv");

  PARTHENON_REQUIRE_THROWS(nv == 3, "3 have 3 velocities");

  const std::string mode = pin->GetOrAddString("hydro_modes", "mode", "entropy");
  const std::string physics = pin->GetOrAddString("hydro_modes", "physics", "mhd");
  const Real amp = pin->GetReal("hydro_modes", "amplitude");

  // Parameters
  double rho0 = 1.0;
  double ug0 = 1.e-2;
  double u10 = 0.;
  double u20 = 0.;
  double u30 = 0.;
  double B10 = 8.164966e-02;
  double B20 = 8.164966e-02;
  double B30 = 0.;

  // Wavevector
  double k1 = 2.*M_PI;

  complex<double> omega = 0.;
  complex<double> drho = 0.;
  complex<double> dug = 0.;
  complex<double> du1 = 0.;
  complex<double> du2 = 0.;
  complex<double> du3 = 0.;
  complex<double> dB1 = 0.;
  complex<double> dB2 = 0.;
  complex<double> dB3 = 0.;

  if (physics == "hydro") {
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
      msg << "Mode \"" << mode << "\" not recognized!";
      PARTHENON_FAIL(msg);
    }
  } else if (physics == "mhd") {
    if (mode == "alfven") {
      omega = complex<double>(0., 0.8957108706097654);
      drho = 0.9809599020287527;
      dug = 0.016349331700479263;
      du1 = 0.13984251695957056;
      du2 = -0.06463830154694966;
      dB2 = 0.11711673829229265;
    } else {
      std::stringstream msg;
      msg << "Mode \"" << mode << "\" not recognized!";
      PARTHENON_FAIL(msg);
    }
  } else {
    std::stringstream msg;
    msg << "Physics option \"" << physics << "\" not recognized!";
    PARTHENON_FAIL(msg);
  }
  pin->SetReal("parthenon/time", "tlim", 2.*M_PI/omega.imag()); // Set final time to be one period

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
    "Phoebus::ProblemGenerator::Linear_Modes", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);

      const double mode = amp*cos(k1*x);

      double rho = rho0 + (drho*mode).real();
      v(irho, k, j, i) = rho;
      double ug = ug0 + (dug*mode).real();
      double Pg = (gam - 1.)*ug;
      v(iprs, k, j, i) = Pg;
      v(ieng, k, j, i) = ug;
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho); // this doesn't have to be exact, just a reasonable guess
      if (ivhi > 0) {
        v(ivlo, k, j, i) = u10 + (du1*mode).real();
      }
      if (ivhi >= 2) {
        v(ivlo + 1, k, j, i) = u20 + (du2*mode).real();
      }
      //for (int d = 0; d < nv; d++) v(ivlo+d, k, j, i) = 0.0;
      if (ib_hi >= 1) {
        v(ib_lo, k, j, i) = B10 + (dB1*mode).real();
      }
      if (ib_hi >= 2) {
        v(ib_lo + 1, k, j, i) = B20 + (dB2*mode).real();
      }
    });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace linear_modes
