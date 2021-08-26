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

#include "pgen/pgen.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/boyer_lindquist.hpp"
#include "utils/error_checking.hpp"

//namespace phoebus {

namespace bondi {

KOKKOS_FUNCTION
Real get_bondi_temp(const Real r, const Real n, const Real C1, const Real C2, const Real Tc, const Real rc) {
  Real rtol = 1.e-12;
  Real ftol = 1.e-14;
  // TODO(jcd): this hardcoded 0.8 is really only appropriate for gamma = 1.4.
  //            for other gamma, this Tguess may not fall between the two roots
  Real Tguess = Tc * std::pow(rc/r,0.8);
  Real Tmin = (r < rc ? 0.1*Tguess : Tguess);
  Real Tmax = (r < rc ? Tguess : 10.*Tguess);
  Real f0, f1, fh;
  Real T0, T1, Th;

  auto get_Tfunc = [&](const Real T) {
    return std::pow(1.+(1.+n)*T,2.)*(1.-2.0/r+std::pow(C1/(r*r*std::pow(T,n)),2.))-C2;
  };

  T0 = Tmin;
  f0 = get_Tfunc(T0);
  T1 = Tmax;
  f1 = get_Tfunc(T1);

  if (f0*f1 > 0.) {
    printf("Failed solving for T at r = %e C1 = %e C2 = %e\n", r, C1, C2);
    PARTHENON_FAIL("Bondi setup failed");
  }

  Th = 0.5*(T0 + T1);//(f1*T0 - f0*T1)/(f1 - f0);
  fh = get_Tfunc(Th);
  while ((T1 - T0)/Th > rtol && std::fabs(fh) > ftol) {
    if (fh*f0 < 0.) {
      T1 = Th;
      f1 = fh;
    } else {
      T0 = Th;
      f0 = fh;
    }

    Th = 0.5*(T0+T1);//(f1*T0 - f0*T1)/(f1 - f0);
    fh = get_Tfunc(Th);
  }

  return Th;
}

KOKKOS_INLINE_FUNCTION
  void bl_to_ks(const Real r, const Real a, Real *ucon_bl, Real *ucon_ks) {
    using namespace Geometry;
    Real trans[NDFULL][NDFULL];
    LinearAlgebra::SetZero(trans, NDFULL, NDFULL);
    const Real idenom = 1.0/(r*r - 2.0*r + a*a);
    trans[0][0] = 1.0;
    trans[0][1] = 2.0*r*idenom;
    trans[1][1] = 1.0;
    trans[2][2] = 1.0;
    trans[3][1] = a*idenom;
    trans[3][3] = 1.0;
    LinearAlgebra::SetZero(ucon_ks, NDFULL);
    SPACETIMELOOP2(mu,nu) {
      ucon_ks[mu] += trans[mu][nu]*ucon_bl[nu];
    }
  }

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
    "Problem \"bondi\" requires \"FMKS\" geometry!");

  auto rc = pmb->meshblock_data.Get().get();

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density,
                              fluid_prim::velocity,
                              fluid_prim::energy,
                              fluid_prim::bfield,
                              fluid_prim::ye,
                              fluid_prim::pressure,
                              fluid_prim::temperature},
                              imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iye  = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;

  // this only works with ideal gases
  const std::string eos_type = pin->GetString("eos","type");
  PARTHENON_REQUIRE_THROWS(eos_type=="IdealGas", "Bondi setup only works with ideal gas");
  const Real gam = pin->GetReal("eos","Gamma");
  const Real Cv = pin->GetReal("eos", "Cv");
  const Real n = 1.0/(gam - 1.0);
  PARTHENON_REQUIRE_THROWS(std::fabs(n-Cv) < 1.e-12, "Bondi requires Cv = 1/(Gamma-1)");
  PARTHENON_REQUIRE_THROWS(std::fabs(gam - 1.4) < 1.e-12, "Bondi requires gamma = 1.4");
  const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
  const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
  const Real Rhor = pin->GetOrAddReal("bondi", "Rhor", 2.0);

  // Solution constants
  const Real uc = std::sqrt(1.0/(2.*rs));
  const Real Vc = -std::sqrt(std::pow(uc,2)/(1. - 3.*std::pow(uc,2)));
  const Real Tc = -n*std::pow(Vc,2)/((n + 1.)*(n*std::pow(Vc,2) - 1.));
  const Real C1 = uc*std::pow(rs,2)*std::pow(Tc,n);
  const Real C2 = std::pow(1. + (1. + n)*Tc,2)*(1. - 2./rs + std::pow(C1,2)/
       (std::pow(rs,4)*std::pow(Tc,2*n)));

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  const Real a = pin->GetReal("geometry","a");
  auto bl = Geometry::BoyerLindquist(a);

  auto geom = Geometry::GetCoordinateSystem(rc);

  // set up transformation stuff
  auto gpkg = pmb->packages.Get("geometry");
  bool derefine_poles = gpkg->Param<bool>("derefine_poles");
  Real h = gpkg->Param<Real>("h");
  Real xt = gpkg->Param<Real>("xt");
  Real alpha = gpkg->Param<Real>("alpha");
  Real x0 = gpkg->Param<Real>("x0");
  Real smooth = gpkg->Param<Real>("smooth");
  auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

  pmb->par_for(
    "Phoebus::ProblemGenerator::Bondi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      Real x1 = coords.x1v(k,j,i);
      const Real x2 = coords.x2v(k,j,i);
      const Real x3 = coords.x3v(k,j,i);

      Real r = tr.bl_radius(x1);
      while (r < Rhor) {
        x1 += coords.dx1v(i);
        r = tr.bl_radius(x1);
      }

      v(itmp,k,j,i) = get_bondi_temp(r, n, C1, C2, Tc, rs);
      v(irho,k,j,i) = std::pow(v(itmp,k,j,i),n);
      v(ieng,k,j,i) = v(irho,k,j,i)*v(itmp,k,j,i)/(gam - 1.0);
      v(iprs,k,j,i) = eos.PressureFromDensityInternalEnergy(v(irho,k,j,i), v(ieng,k,j,i)/v(irho,k,j,i));
      Real ucon_bl[] = {0.0, 0.0, 0.0, 0.0};
      ucon_bl[1] = -C1/(std::pow(v(itmp,k,j,i),n)*std::pow(r,2));

      Real gcov[4][4];
      const Real th = tr.bl_theta(x1,x2);
      bl.SpacetimeMetric(0.0, r, th, x3, gcov);
      Real AA = gcov[0][0];
      Real BB = 2.*(gcov[0][1]*ucon_bl[1] +
                    gcov[0][2]*ucon_bl[2] +
                    gcov[0][3]*ucon_bl[3]);
      Real CC = 1. + gcov[1][1]*ucon_bl[1]*ucon_bl[1] +
                     gcov[2][2]*ucon_bl[2]*ucon_bl[2] +
                     gcov[3][3]*ucon_bl[3]*ucon_bl[3] +
                2. *(gcov[1][2]*ucon_bl[1]*ucon_bl[2] +
                     gcov[1][3]*ucon_bl[1]*ucon_bl[3] +
                     gcov[2][3]*ucon_bl[2]*ucon_bl[3]);
      Real discr = BB*BB - 4.*AA*CC;
      ucon_bl[0] = (-BB - std::sqrt(discr))/(2.*AA);
      const Real W_bl = ucon_bl[0]*bl.Lapse(0.0, r, th, x3);

      Real ucon[4];
      tr.bl_to_fmks(x1,x2,x3,a,ucon_bl, ucon);

      // ucon won't be properly normalized here if x1 is not consistent with i
      // so renormalize
      geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
      AA = gcov[0][0];
      BB = 2.*(gcov[0][1]*ucon[1] +
               gcov[0][2]*ucon[2] +
               gcov[0][3]*ucon[3]);
      CC = 1. + gcov[1][1]*ucon[1]*ucon[1] +
                gcov[2][2]*ucon[2]*ucon[2] +
                gcov[3][3]*ucon[3]*ucon[3] +
           2. *(gcov[1][2]*ucon[1]*ucon[2] +
                gcov[1][3]*ucon[1]*ucon[3] +
                gcov[2][3]*ucon[2]*ucon[3]);
      discr = BB*BB - 4.*AA*CC;
      PARTHENON_REQUIRE(discr > 0, "discr < 0");
      ucon[0] = (-BB - std::sqrt(discr))/(2.*AA);

      // now get three velocity
      const Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
      Real beta[3];
      geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
      Real W = lapse * ucon[0];
      for (int d = 0; d < 3; d++) {
        v(ivlo+d,k,j,i) = ucon[d+1]/W + beta[d]/lapse;
      }
    });

  fluid::PrimitiveToConserved(rc);
}

}
