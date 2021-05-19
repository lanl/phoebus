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

// #include <complex>
#include <Kokkos_Complex.hpp>
#include <sstream>
#include <typeinfo>

#include "geometry/geometry.hpp"

#include "pgen/pgen.hpp"

// Relativistic hydro linear modes.
// TODO: Make this 3D instead of 2D.

using Kokkos::complex;
using Geometry::NDFULL;

namespace linear_modes {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();
  const int ndim = pmb->pmy_mesh->ndim;

  PackIndexMap imap;
  auto v = rc->PackVariables({"p.density",
                              "p.velocity",
                              "p.energy",
                              fluid_prim::bfield,
                              "pressure",
                              "temperature",
                              "gamma1",
                              "cs"},
                              imap);

  const int irho = imap["p.density"].first;
  const int ivlo = imap["p.velocity"].first;
  const int ivhi = imap["p.velocity"].second;
  const int ieng = imap["p.energy"].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iprs = imap["pressure"].first;
  const int itmp = imap["temperature"].first;
  const int nv = ivhi - ivlo + 1;

  const Real gam = pin->GetReal("eos", "Gamma");

  PARTHENON_REQUIRE_THROWS(nv == 3, "3 have 3 velocities");

  const std::string mode = pin->GetOrAddString("hydro_modes", "mode", "entropy");
  const std::string physics = pin->GetOrAddString("hydro_modes", "physics", "mhd");
  const Real amp = pin->GetReal("hydro_modes", "amplitude");

  // Parameters
  double rho0 = 1.;
  double ug0 = 1.;
  double u10 = 0.;
  double u20 = 0.;
  double u30 = 0.;
  double B10 = 1.;
  double B20 = 0.;
  double B30 = 0.;

  // Wavevector
  double k1 = 2.*M_PI;
  double k2 = 2.*M_PI;

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
      drho = 1.;
      dug = 0.;
      du1 = 0.;
      u10 = 0.1; // Uniform advection
    } else if (mode == "sound") {
      if (ndim == 1) {
        omega = complex<double>(0., 2.7422068833892093);
        drho = 0.5804294924639215;
        dug = 0.7739059899518946;
        du1 = -0.2533201985524494;
      } else if (ndim == 2) {
        omega = complex<double>(0., 3.8780661653218766);
        drho = 0.5804294924639213;
        dug = 0.7739059899518947;
        du1 = 0.1791244302079596;
        du2 = 0.1791244302079596;
      } else {
        PARTHENON_FAIL("ndim == 3 not supported!");
      }
    } else {
      std::stringstream msg;
      msg << "Mode \"" << mode << "\" not recognized!";
      PARTHENON_FAIL(msg);
    }
  } else if (physics == "mhd") {
    if (mode == "slow") {
      omega = complex<double>(0., 2.41024185339);
      drho = 0.558104461559;
      dug = 0.744139282078;
      du1 = -0.277124827421;
      du2 = 0.063034892770;
      dB1 = -0.164323721928;
      dB2 = 0.164323721928;
    } else if (mode == "alfven") {
      omega = complex<double>(0., 3.44144232573);
      du3 = 0.480384461415;
      dB3 = 0.877058019307;
    } else if (mode == "fast") {
      omega = complex<double>(0., 5.53726217331);
      drho = 0.476395427447;
      dug = 0.635193903263;
      du1 = -0.102965815319;
      du2 = -0.316873207561;
      dB1 = 0.359559114174;
      dB2 = -0.359559114174;
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
  auto gpkg = pmb->packages.Get("geometry");
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  Real a_snake, k_snake;
  if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake)) {
    a_snake = gpkg->Param<Real>("a");
    k_snake = gpkg->Param<Real>("k");
  }

  pmb->par_for(
    "Phoebus::ProblemGenerator::Linear_Modes", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real x = coords.x1v(i);
      Real y = coords.x2v(j);

      if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake)) {
        y = y - a_snake*sin(k_snake*x);
      }

      const double mode = amp*cos(k1*x + k2*y);

      double rho = rho0 + (drho*mode).real();
      v(irho, k, j, i) = rho;
      double ug = ug0 + (dug*mode).real();
      double Pg = (gam - 1.)*ug;
      v(ieng, k, j, i) = ug;
      v(iprs, k, j, i) = Pg;
      // This line causes NaNs and I don't know why
      //v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho);
      v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, v(ieng, k, j, i)/rho);
      if (ivhi > 0) {
        v(ivlo, k, j, i) = u10 + (du1*mode).real();
      }
      if (ivhi >= 2) {
        v(ivlo + 1, k, j, i) = u20 + (du2*mode).real();
      }
      if (ivhi >= 3) {
        v(ivlo + 2, k, j, i) = u30 + (du3*mode).real();
      }
      if (ib_hi >= 1) {
        v(ib_lo, k, j, i) = B10 + (dB1*mode).real();
      }
      if (ib_hi >= 2) {
        v(ib_lo + 1, k, j, i) = B20 + (dB2*mode).real();
      }
      if (ib_hi >= 3) {
        v(ib_lo + 2, k, j, i) = B30 + (dB3*mode).real();
      }

      if (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Snake)) {
        PARTHENON_REQUIRE(ivhi == 3, "Only works for 3D velocity!");
        // Transform velocity
        //Real Gamma = 1./sqrt(1. - pow(v(ivlo, k, j, i),2) + pow(v(ivlo+1, k, j, i),2)
        //  + pow(v(ivlo+2, k, j, i),2));
        Real vsq = 0.;
        Real gcov[NDFULL][NDFULL];
        Real vcon[3] = {v(ivlo, k, j, i), v(ivlo+1, k, j, i), v(ivlo+2, k, j, i)};
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 3; n++) {
            vsq += gcov[m+1][n+1]*vcon[m]*vcon[n];
          }
        }
        Real Gamma = 1./sqrt(1. - vsq);
        Real ucon[NDFULL] = {Gamma, // alpha = 1
                             Gamma*v(ivlo, k, j, i),
                             Gamma*v(ivlo+1, k, j, i),
                             Gamma*v(ivlo+2, k, j, i)};
        if (i == 64 && j == 64) {
          printf("Gamma: %e\n", Gamma);
        }
        Real J[NDFULL][NDFULL];
        J[0][0] = 1.;
        J[1][1] = 1.;
        J[2][2] = 1.;
        J[3][3] = 1.;
        //J[2][2] = -a_snake*sin(k_snake*x) + 1.;
        //J[2][1] = -a_snake*k_snake*cos(k_snake*x);
        J[2][1] = a_snake*k_snake*cos(k_snake*x);
        Real ucon_snake[NDFULL] = {0, 0, 0, 0};
        SPACETIMELOOP(mu) SPACETIMELOOP(nu){
          ucon_snake[mu] += J[mu][nu]*ucon[nu];
        }
        Gamma = ucon_snake[0]; // alpha = 1
        v(ivlo, k, j, i) = ucon_snake[1];
        v(ivlo+1, k, j, i) = ucon_snake[2];
        v(ivlo+2, k, j, i) = ucon_snake[3];
        //printf("ivlo: %i ivhi: %i u: %e %e %e\n", ivlo, ivhi, ucon_snake[1], ucon_snake[2],
        //  ucon_snake[3]);


        /*printf("Temporarily making constant value!\n");
        double rho = rho0;
        v(irho, k, j, i) = rho;
        double ug = ug0;
        double Pg = (gam - 1.)*ug;
        v(ieng, k, j, i) = ug;
        v(iprs, k, j, i) = Pg;
        v(ivlo, k, j, i) = 0.;
        v(ivlo + 1, k, j, i) = 0.;
        v(ivlo + 2, k, j, i) = 0.;
        */
        v(ib_lo, k, j, i) = 0.;
        v(ib_lo + 1, k, j, i) = 0.;
        v(ib_lo + 2, k, j, i) = 0.;
      }
    });

    //    exit(-1);

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace linear_modes
