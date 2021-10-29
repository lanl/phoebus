#include "pgen/pgen.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/boyer_lindquist.hpp"
#include "fluid/con2prim_robust.hpp"
#include "Kokkos_Random.hpp"

// Temporary
#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;
#include "fixup/fixup.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace torus {

Real lfish_calc(Real r, Real a) {
  return (((std::pow(a, 2) - 2. * a * std::sqrt(r) + std::pow(r, 2)) *
     ((-2. * a * r *
       (std::pow(a, 2) - 2. * a * std::sqrt(r) +
        std::pow(r,
      2))) / std::sqrt(2. * a * std::sqrt(r) + (-3. + r) * r) +
      ((a + (-2. + r) * std::sqrt(r)) * (std::pow(r, 3) + std::pow(a, 2) *
      (2. + r))) / std::sqrt(1 + (2. * a) / std::pow (r, 1.5) - 3. / r)))
    / (std::pow(r, 3) * std::sqrt(2. * a * std::sqrt(r) + (-3. + r) * r) *
       (std::pow(a, 2) + (-2. + r) * r))
      );
}

KOKKOS_FUNCTION
Real log_enthalpy(const Real r, const Real th, const Real a, const Real rin, const Real l, Real &uphi) {
  const Real sth = sin(th);
  const Real cth = cos(th);

  const Real DD = r * r - 2. * r + a * a;
  const Real AA = (r * r + a * a) * (r * r + a * a) -
             DD * a * a * sth * sth;
  const Real SS = r * r + a * a * cth * cth;

  const Real thin = M_PI / 2.;
  const Real sthin = sin(thin);
  const Real cthin = cos(thin);

  const Real DDin = rin * rin - 2. * rin + a * a;
  const Real AAin = (rin * rin + a * a) * (rin * rin + a * a)
             - DDin * a * a * sthin * sthin;
  const Real SSin = rin * rin + a * a * cthin * cthin;
  uphi = 0.0;
  const Real lnh =
            0.5 *
            std::log((1. +
           std::sqrt(1. +
                4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth)))
          / (SS * DD / AA))
            - 0.5 * std::sqrt(1. +
             4. * (l * l * SS * SS) * DD /
             (AA * AA * sth * sth))
            - 2. * a * r * l / AA -
            (0.5 *
             std::log((1. +
            std::sqrt(1. +
                 4. * (l * l * SSin * SSin) * DDin /
                 (AAin * AAin * sthin * sthin))) /
           (SSin * DDin / AAin))
             - 0.5 * std::sqrt(1. +
              4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin))
             - 2. * a * rin * l / AAin);
  if (lnh > 0.0) {
    Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
    Real up1 =
              std::sqrt((-1. +
              std::sqrt(1. + 4. * l * l * expm2chi)) / 2.);
    uphi = 2. * a * r * std::sqrt(1. +
                     up1 * up1) / std::sqrt(AA * SS *
                     DD) +
              std::sqrt(SS / AA) * up1 / sth;
  }
  return lnh;
}

KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]) {
  Real AA = gcov[0][0];
  Real BB = 2.*(gcov[0][1]*ucon[1] +
                gcov[0][2]*ucon[2] +
                gcov[0][3]*ucon[3]);
  Real CC = 1. + gcov[1][1]*ucon[1]*ucon[1] +
                 gcov[2][2]*ucon[2]*ucon[2] +
                 gcov[3][3]*ucon[3]*ucon[3] +
            2. *(gcov[1][2]*ucon[1]*ucon[2] +
                 gcov[1][3]*ucon[1]*ucon[3] +
                 gcov[2][3]*ucon[2]*ucon[3]);
  Real discr = BB*BB - 4.*AA*CC;
  if (discr < 0) printf("discr = %g   %g %g %g\n", discr, AA, BB, CC);
  PARTHENON_REQUIRE(discr >= 0, "discr < 0");
  return (-BB - std::sqrt(discr))/(2.*AA);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

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
  const int iblo = imap[fluid_prim::bfield].first;
  const int ibhi = imap[fluid_prim::bfield].second;
  const int iye  = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;

  // this only works with ideal gases
  const std::string eos_type = pin->GetString("eos","type");
  PARTHENON_REQUIRE_THROWS(eos_type=="IdealGas", "Bondi setup only works with ideal gas");
  const Real gam = pin->GetReal("eos","Gamma");
  const Real Cv = pin->GetReal("eos", "Cv");

  const Real rin = pin->GetOrAddReal("torus", "rin", 6.0);
  const Real rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
  const Real kappa = pin->GetOrAddReal("torus", "kappa", 1.e-2);
  const Real u_jitter = pin->GetOrAddReal("torus", "u_jitter", 1.e-2);
  const int seed = pin->GetOrAddInteger("torus", "seed", time(NULL));
  const Real bnorm = pin->GetOrAddReal("torus", "Bnorm", 1.e-2);

  const Real a = pin->GetReal("geometry","a");
  auto bl = Geometry::BoyerLindquist(a);

  // Solution constants
  const Real angular_mom = lfish_calc(rmax,a);

  //printf("rmax: %e a: %e angular_mom: %e\n", rmax, a, angular_mom);
  //exit(-1);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  auto floor = pmb->packages.Get("fixup")->Param<fixup::Floors>("floor");

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

  RNGPool rng_pool(pin->GetOrAddInteger("kelvin_helmholtz", "seed", seed));

  Real uphi_rmax;
  const Real hm1_rmax = std::exp(log_enthalpy(rmax,0.5*M_PI,a,rin,angular_mom,uphi_rmax)) - 1.0;
  const Real rho_rmax = std::pow(hm1_rmax * (gam - 1.) / (kappa * gam),
                 1. / (gam - 1.));
  const Real u_rmax = kappa * std::pow(rho_rmax, gam) / (gam - 1.) / rho_rmax;
  pmb->par_for(
    "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, 256/2 + jb.s, 256/2 + jb.s, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      auto rng_gen = rng_pool.get_state();
      const Real x1 = coords.x1v(k,j,i);
      const Real x2 = coords.x2v(k,j,i);
      const Real x3 = coords.x3v(k,j,i);

      Real r = tr.bl_radius(x1);
      Real th = tr.bl_theta(x1,x2);
      printf("[%i %i %i] X = %e %e %e r = %e th = %e\n", i,j,k,x1,x2,x3,r,th);
    });
  pmb->exec_space.fence();
  exit(-1);

  printf("rho_rmax: %28.18e\n", rho_rmax);
  printf("u_rmax: %28.18e\n", u_rmax);

  printf("xmin: %28.18e %28.18e %28.18e\n", coords.x1f(0,4,4), coords.x2f(0,4,4),
    coords.x3f(0,4,4));
  printf("dx: %28.18e %28.18e %28.18e\n", coords.dx1v(0,4,4), coords.dx1v(0,4,4),
    coords.dx1v(0,4,4));
  printf("dx: %28.18e %28.18e %28.18e\n", coords.dx1f(0,4,4), coords.dx1f(0,4,4),
    coords.dx1f(0,4,4));
  printf("[NG NG 0] X: %28.18e %28.18e %28.18e\n",
    coords.x1v(0,4,4), coords.x2v(0,4,4), coords.x3v(0,4,4));
  printf("[133 131 0] X: %28.18e %28.18e %28.18e\n",
    coords.x1v(0,131,133), coords.x2v(0,131,133), coords.x3v(0,131,133));

  pmb->par_for(
    "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      auto rng_gen = rng_pool.get_state();
      const Real x1 = coords.x1v(k,j,i);
      const Real x2 = coords.x2v(k,j,i);
      const Real x3 = coords.x3v(k,j,i);

      Real r = tr.bl_radius(x1);
      Real th = tr.bl_theta(x1,x2);

      Real lnh = -1.0;
      Real uphi;
      if (r > rin) lnh = log_enthalpy(r,th,a,rin,angular_mom,uphi);


      //if (j == 128) {
      //  printf("i: %i x1: %e r: %e lnh: %e\n", i, x1, r, lnh);
     // }
     /*{
       if (i == 120 && j == 120) {
         printf("[%i %i] X = %28.18e %28.18e %28.18e\n", i, j, x1, x2, x3);
         const Real lapse = geom.Lapse(CellLocation::Cent,k,j,i);
         printf("alpha: %e\n", lapse);
         Real dgcov[4][4][4] = {0};
         geom.MetricDerivative(CellLocation::Cent, k, j, i, dgcov);
         Real dlnalpha[4] = {0};
         geom.GradLnAlpha(CellLocation::Cent, k, j, i, dlnalpha);
         printf("dlnalpha: %e %e %e %e\n",
           dlnalpha[0], dlnalpha[1], dlnalpha[2], dlnalpha[3]);
         printf("dgcov:\n");
         SPACETIMELOOP3(mu, nu, lam) {
           printf("[%i %i %i] %e\n", mu, nu, lam, dgcov[mu][nu][lam]);
         }
       }
     }*/

      Real beta[3];
      Real gcov[4][4];
      // regions outside torus
      if (lnh < 0. || r < rin) {
        // Nominal values; real value set by fixup
        v(irho,k,j,i) = 0.0;
        v(ieng,k,j,i) = 0.0;
        // Actually start out with a hot atmosphere
        v(irho,k,j,i) = 1e-5*pow(r,-2.);
        //v(ieng,k,j,i) = 1.e-1*v(irho,k,j,i)*pow(r,-1.);
        v(ieng,k,j,i) = 1.e-7*pow(r,-(2.*5./3.));
        geom.SpacetimeMetric(CellLocation::Cent,k,j,i,gcov);
        geom.ContravariantShift(CellLocation::Cent,k,j,i,beta);
        const Real lapse = geom.Lapse(CellLocation::Cent,k,j,i);
        Real ucon[4] = {0.0, -beta[1]/lapse, 0.0, -beta[2]/lapse};
        //ucon[0] = ucon_norm(ucon,gcov);
        const Real W = ucon[0]*lapse;
        for (int d = 0; d < 3; d++) {
          v(ivlo+d,k,j,i) = 0.0;//beta[d]/lapse;
          //v(ivlo+d,k,j,i) = beta[d];
        }
        //if (i == 138 && j == 186) {
        //  printf("ATMOS ZONE! [%i %i] rho = %e u = %e\n", i, j, v(irho,k,j,i), v(ieng,k,j,i));
        //}
      }
      /* region inside magnetized torus; u^i is calculated in
       * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
       * so it needs to be transformed at the end */
      else {
        Real hm1 = std::exp(lnh) - 1.;
        Real rho = std::pow(hm1 * (gam - 1.) / (kappa * gam),
                 1. / (gam - 1.));
        Real u = kappa * std::pow(rho, gam) / (gam - 1.);

        v(irho,k,j,i) = rho / rho_rmax;
        v(ieng,k,j,i) = u / rho_rmax;
        v(ieng,k,j,i) *= (1. + u_jitter * (rng_gen.drand() - 0.5));

        Real ucon_bl[] = {0.0, 0.0, 0.0, uphi};
        bl.SpacetimeMetric(0.0, r, th, x3, gcov);
        ucon_bl[0] = ucon_norm(ucon_bl, gcov);
        Real ucon[4];
        tr.bl_to_fmks(x1,x2,x3,a,ucon_bl,ucon);
        const Real lapse = geom.Lapse(CellLocation::Cent,k,j,i);
        geom.ContravariantShift(CellLocation::Cent,k,j,i,beta);
        geom.SpacetimeMetric(CellLocation::Cent,k,j,i, gcov);
        ucon[0] = ucon_norm(ucon, gcov);
        const Real W = lapse*ucon[0];
        for (int d = 0; d < 3; d++) {
          v(ivlo+d,k,j,i) = ucon[d+1]/W + beta[d]/lapse;
        }
        //if (j == 127 + 4 && i > 180 && i < 190) {
        //  int d = 0;
        //  //printf("[%i] vtilde: %e beta: %e alpha: %e\n", i, lapse*ucon[d+1]/beta[d], beta[d], lapse);
        //  printf("[%i] v1: %e u^1: %e u^e: %e vtilde: %e beta/alpha = %e\n", i, v(ivlo+d,k,j,i),
        //    ucon[d+1], ucon[d+2], lapse*ucon[d+1]/beta[d], beta[d]/lapse);
        //}
        //v(ivlo,k,j,i) = beta[0]/lapse;
        ///v(ivlo+1,k,j,i) = beta[1]/lapse;
        //if (i == 138 && j == 186) {
        //  printf("DISK ZONE! [%i %i] rho = %e u = %e\n", i, j, v(irho,k,j,i), v(ieng,k,j,i));
        //}
        /*if (i == 133 && j == 83) {
          printf("init p: %e %e %e %e %e\n", v(irho,k,j,i), v(ieng,k,j,i), v(ivlo,k,j,i),
            v(ivlo+1,k,j,i), v(ivlo+2,k,j,i));
          printf("x: %28.18e %28.18e %28.18e\n", x1, x2, x3);
          printf("r: %28.18e th: %28.18e\n", r, th);
          printf("rho: %e hm1: %28.18e gam: %28.18e kappa: %28.18e\n", rho, hm1, gam, kappa);
          //exit(-1);
        }*/
      }
      // fixup
      Real rhoflr, epsflr;
      floor.GetFloors(x1, x2, x3, rhoflr, epsflr);
      v(irho,k,j,i) = v(irho,k,j,i) < rhoflr ? rhoflr : v(irho,k,j,i);
      v(ieng,k,j,i) = v(ieng,k,j,i)/v(irho,k,j,i) < epsflr ? v(irho,k,j,i)*epsflr : v(ieng,k,j,i);
      v(iprs,k,j,i) = eos.PressureFromDensityInternalEnergy(v(irho,k,j,i), v(ieng,k,j,i)/v(irho,k,j,i));
        if (i == 133 && j >= 81 && j <= 86) {
          printf("[%i %i] rho: %e uu: %e P: %e\n", i, j, v(irho,k,j,i),
            v(ieng,k,j,i), v(iprs,k,j,i));
        }

      v(itmp,k,j,i) = v(ieng,k,j,i)/v(irho,k,j,i)/Cv;
      //fprintf(stderr,"%g %g %g %g\n", r, th, v(irho,k,j,i), v(ieng,k,j,i));
      //if (i == ib.e) fprintf(stderr,"\n");
      rng_pool.free_state(rng_gen);
    });

  //exit(-1);

  //pmb->exec_space.fence();
  //exit(-1);

  // get vector potential
  ParArrayND<Real> A("vector potential", jb.e+1, ib.e+1);
  pmb->par_for(
    "Phoebus::ProblemGenerator::Torus2", jb.s+1, jb.e, ib.s+1, ib.e,
    KOKKOS_LAMBDA(const int j, const int i) {
      const Real rho_av = 0.25*(v(irho,kb.s,j,i) + v(irho,kb.s,j,i-1)
                              + v(irho,kb.s,j-1,i) + v(irho,kb.s,j-1,i-1));
      const Real q = rho_av / rho_rmax - 0.2;
      A(j,i) = (q > 0 ? q : 0.0);
    });

  //Real bsq_max;
  Real beta_min;
  if (ibhi > 0) {
  pmb->par_reduce(
    "Phoebus::ProblemGenerator::Torus3", kb.s, kb.e, jb.s, jb.e-1, ib.s, ib.e-1,
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &bmin) {
      const Real gdet = geom.DetGamma(CellLocation::Cent,k,j,i);
      v(iblo,k,j,i) = - ( A(j,i) - A(j+1,i)
                         + A(j,i+1) - A(j+1,i+1) )
                         / (2.0 * coords.Dx(X2DIR,k,j,i) * gdet);
      v(iblo+1,k,j,i) = ( A(j,i) + A(j+1,i)
                         - A(j,i+1) - A(j+1,i+1) )
                         / (2.0 * coords.Dx(X1DIR,k,j,i) * gdet);
      v(ibhi,k,j,i) = 0.0;

      v(iblo,k,j,i) *= bnorm;
      v(iblo+1,k,j,i) *= bnorm;


      // compute bsq for max
      Real vsq = 0.0;
      Real Bsq = 0.0;
      Real Bdotv = 0.0;
      Real gcov[3][3];
      geom.Metric(CellLocation::Cent,k,j,i,gcov);
      SPACELOOP2(m,n) {
        vsq += gcov[m][n] * v(ivlo+m,k,j,i) * v(ivlo+n,k,j,i);
        Bdotv += gcov[m][n] * v(ivlo+m,k,j,i) * v(iblo+n,k,j,i);
        Bsq += gcov[m][n] * v(iblo+m,k,j,i) * v(iblo+n,k,j,i);
      }
      const Real W = 1.0/std::sqrt(1.0 - vsq);
      if (i == 128 && j == 128) {
        printf("original W: %e gam: %e\n", W, gam);
      }
      Real b0 = W*Bdotv;
      Real bsq = (Bsq + b0*b0)/(W*W);
      Real beta = v(iprs,k,j,i)/(0.5*bsq + 1.e-10);
      if (v(irho,k,j,i) > 1.e-8 && beta < bmin) bmin = beta;
      //if (bsq > b2max) b2max = bsq;
    }, Kokkos::Min<Real>(beta_min));
    if (parthenon::Globals::my_rank == 0)
      std::cout << "beta_min = " << beta_min << std::endl;
  }

  // Lets smooth the fluid out and see what happens
  const int nsmooth = 0;
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  for (int n = 0; n < nsmooth; n++) {
    //auto rho_copy =
    auto pkg = pmb->packages.Get("fluid");
    auto rho = pmb->meshblock_data.Get()->Get(fluid_prim::density).data;
    auto rho_copy = rho.GetMirrorAndCopy(Kokkos::DefaultExecutionSpace());
    auto ug = pmb->meshblock_data.Get()->Get(fluid_prim::energy).data;
    auto ug_copy = ug.GetMirrorAndCopy(Kokkos::DefaultExecutionSpace());
    auto v = pmb->meshblock_data.Get()->Get(fluid_prim::velocity).data;
    auto v_copy = v.GetMirrorAndCopy(Kokkos::DefaultExecutionSpace());
    auto t = pmb->meshblock_data.Get()->Get(fluid_prim::temperature).data;
    auto t_copy = t.GetMirrorAndCopy(Kokkos::DefaultExecutionSpace());
    auto p = pmb->meshblock_data.Get()->Get(fluid_prim::pressure).data;
    auto p_copy = p.GetMirrorAndCopy(Kokkos::DefaultExecutionSpace());

    pmb->par_for(
      "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        double ref = rho(k,j,i);
        if (rho_copy(0,0,0,k,j,i+1) > 10.*ref || rho_copy(0,0,0,k,j,i-1) > 10.*ref ||
            rho_copy(0,0,0,k,j+1,i) > 10.*ref || rho_copy(0,0,0,k,j-1,i) > 10.*ref) {
          rho(k,j,i) = (1./4.)*(rho_copy(0,0,0,k,j,i+1) + rho_copy(0,0,0,k,j,i-1) +
                                rho_copy(0,0,0,k,j+1,i) + rho_copy(0,0,0,k,j-1,i));
          ug(k,j,i) = (1./4.)*(ug_copy(0,0,0,k,j,i+1) + ug_copy(0,0,0,k,j,i-1) +
                                ug_copy(0,0,0,k,j+1,i) + ug_copy(0,0,0,k,j-1,i));
          for (int ii = 0; ii < 3; ii++) {
            v(ii,k,j,i) = (1./4.)*(v_copy(0,0,ii,k,j,i+1) + v_copy(0,0,ii,k,j,i-1) +
                                  v_copy(0,0,ii,k,j+1,i) + v_copy(0,0,ii,k,j-1,i));
          }
          t(k,j,i) = (1./4.)*(t_copy(0,0,0,k,j,i+1) + t_copy(0,0,0,k,j,i-1) +
                                t_copy(0,0,0,k,j+1,i) + t_copy(0,0,0,k,j-1,i));
          p(k,j,i) = (1./4.)*(p_copy(0,0,0,k,j,i+1) + p_copy(0,0,0,k,j,i-1) +
                                p_copy(0,0,0,k,j+1,i) + p_copy(0,0,0,k,j-1,i));
          //t(k,j,i) = ug(k,j,i)/rho(k,j,i)/Cv;
          //p(k,j,i) = ug(k,j,i)/rho(k,j,i)/Cv;
        }
      });

    //parthenon::ApplyBoundaryConditions(pmb->meshblock_data.Get());
    // Boundary conditions
  }

  // now normalize the b-field
  //for (int i = 0; i < 100; i++) {
  fluid::PrimitiveToConserved(rc);

  //fluid::ConservedToPrimitive(rc);
  //fixup::ConservedToPrimitiveFixup(rc);
  //fluid::PrimitiveToConserved(rc);



  /*auto fail = rc->Get(internal_variables::fail).data;

    printf("Before inversion:\n");
    pmb->par_for(
      "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (i == 128 && j == 128) {
          printf("p: %e %e %e %e %e %e %e\n", v(irho,k,j,i), v(ieng,k,j,i), v(iprs,k,j,i), v(itmp,k,j,i),
            v(ivlo,k,j,i), v(ivlo+1,k,j,i), v(ivlo+2,k,j,i));
          printf("fail: %i\n", fail(k,j,i));
        }
      });
  pmb->exec_space.fence();

  fluid::ConservedToPrimitive(rc);


    printf("After inversion:\n");
    pmb->par_for(
      "Phoebus::ProblemGenerator::Torus", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (i == 128 && j == 128) {
          printf("p: %e %e %e %e %e %e %e\n", v(irho,k,j,i), v(ieng,k,j,i), v(iprs,k,j,i), v(itmp,k,j,i),
            v(ivlo,k,j,i), v(ivlo+1,k,j,i), v(ivlo+2,k,j,i));
          printf("fail: %i\n", fail(k,j,i));
        }
      });

  pmb->exec_space.fence();
  exit(-1);*/
  //}
/*
  Real beta[3];
  geom.ContravariantShift(CellLocation::Cent,kb.s,jb.s+256,ib.s+440, beta);
  alpha = geom.Lapse(CellLocation::Cent,kb.s,jb.s+256,ib.s+440);
  std::cout << "rho, u = " << v(irho,kb.s,256,440) << " " << v(ieng, kb.s, 256, 440) << std::endl;
  std::cout << "v1_0 = " << v(ivlo,kb.s,256,440) - beta[0]/alpha << std::endl;
  fluid::PrimitiveToConserved(rc);
  std::cout << "v1_1 = " << v(ivlo,kb.s,256,440) - beta[0]/alpha << std::endl;
  fluid::ConservedToPrimitive(rc);
  std::cout << "rho, u = " << v(irho,kb.s,256,440) << " " << v(ieng, kb.s, 256, 440) << std::endl;
  std::cout << "v1_2 = " << v(ivlo,kb.s,256,440) - beta[0]/alpha << std::endl;
  */
//exit(-1);
}

void ProblemModifier(ParameterInput *pin) {
  Real router = pin->GetOrAddReal("coordinates", "r_outer", 40.0);
  Real x1max = log(router);
  //pin->SetReal("parthenon/mesh", "x1max", x1max);
  //printf("x1max: %28.18e\n", x1max);

  Real a = pin->GetReal("geometry","a");
  Real Rh = 1.0 + sqrt(1.0 - a*a);
  Real xh = log(Rh);
  int ninside = pin->GetOrAddInteger("torus", "n_inside_horizon", 4);
  int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
  Real dx = (x1max - xh)/(nx1 - ninside);
  Real x1min = xh - ninside*dx;
  // TODO(BRR) harm val:
  x1min = 2.147441732474514309e-01;
  //Real x1max = log(40.);

  /*pin->SetReal("parthenon/mesh", "x1min", x1min);\*/
  //pin->SetReal("parthenon/mesh", "x1min", x1min);
  //pin->SetReal("parthenon/mesh", "x1max", x1max);
}


}

