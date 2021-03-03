
#include "riemann.hpp"

namespace riemann {

std::vector<std::string> FluxState::recon_vars;
std::vector<std::string> FluxState::flux_vars;

KOKKOS_FUNCTION
void FluxState::prim_to_flux(const int d, const int k, const int j, const int i,
                  const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F) const {
  const int dir = d-1;
  const Real &rho = q(dir,prho,k,j,i);
  const Real vcon[] = {q(dir,pvel_lo,k,j,i), q(dir,pvel_lo+1,k,j,i), q(dir,pvel_hi,k,j,i)};
  const Real &vel = vcon[dir];
  Real Bcon[] = {0.0, 0.0, 0.0};
  const Real &u = q(dir,peng,k,j,i);
  const Real &P = q(dir,prs,k,j,i);
  const Real &gamma1 = q(dir,gm1,k,j,i);

  for (int m = pb_lo; m <= pb_hi; m++) {
    Bcon[m-pb_lo] = q(dir, m, k, j, i);
  }

  CellLocation loc = DirectionToFaceID(d);

  Real gcov[3][3];
  geom.Metric(loc, k, j, i, gcov);

  Real vcov[3];
  Real BdotB = 0.0;
  Real Bdotv = 0.0;
  for (int m = 0; m < 3; m++) {
    vcov[m] = 0.0;
    for (int n = 0; n < 3; n++) {
      vcov[m] += gcov[m][n]*vcon[n];
    }
    Bdotv += Bcon[m]*vcov[m];
    // TODO(JCD): should we just always execute this loop from 0 <= n < 3
    for (int n = pb_lo; n <= pb_hi; n++) {
      BdotB += gcov[m][n-pb_lo] * Bcon[m] * Bcon[n-pb_lo];
    }
  }

  // get Lorentz factor
  Real vsq = 0.0;
  for (int m = 0; m < 3; m++) {
    vsq += vcon[m]*vcov[m];
  }
  const Real W = 1.0/sqrt(1.0 - vsq);

  const Real alpha = geom.Lapse(d, k, j, i);
  Real beta[Geometry::NDSPACE];
  geom.ContravariantShift(loc, k, j, i, beta);
  const Real vt[3] = {vcon[0] - beta[0]/alpha,
                      vcon[1] - beta[1]/alpha,
                      vcon[2] - beta[2]/alpha};
  const Real &vtil = vt[dir];

  Real gcov4[Geometry::NDFULL][Geometry::NDFULL];
  geom.SpacetimeMetric(loc, k, j, i, gcov4);

  Real gcon[Geometry::NDSPACE][Geometry::NDSPACE];
  geom.MetricInverse(loc, k, j, i, gcon);

  Real b[4] = {W*Bdotv/alpha, 0.0, 0.0, 0.0};
  for (int m = pb_lo; m <= pb_hi; m++) {
    b[m-pb_lo+1] = Bcon[m-pb_lo]/W + alpha*b[0]*vt[m-pb_lo];
  }
  const Real bsq = (BdotB + alpha*alpha*b[0]*b[0])/(W*W);
  Real bcov[] = {0.0, 0.0, 0.0};
  if (pb_hi > 0) {
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 4; n++) {
        bcov[m] += gcov4[m+1][n] * b[n];
      }
    }
  }

  // conserved density
  U[crho] = rho*W;
  if (cye>0) U[cye] = U[crho]*q(dir,pye,k,j,i);

  // conserved momentum
  const Real rhohWsq = (rho + u + P)*W*W;
  for (int m = 0; m < 3; m++) {
    U[cmom_lo+m] = (rhohWsq+bsq)*vcov[m] - alpha*b[0]*bcov[m];
  }

  // conserved energy
  U[ceng] = (rhohWsq+bsq) - (P + 0.5*bsq) - alpha*alpha*b[0]*b[0] - U[crho];

  // magnetic fields
  for (int m = cb_lo; m <= cb_hi; m++) {
    U[m] = Bcon[m-cb_lo];
  }

  // Get fluxes
  // mass flux
  F[crho] = U[crho]*vtil;
  if (cye>0) F[cye] = U[cye]*vtil;

  // momentum flux
  for (int n = 0; n < 3; n++) {
    F[cmom_lo+n] = U[cmom_lo+n]*vtil + (P + 0.5*bsq)*Delta(dir,n) - bcov[n]*Bcon[dir]/W;
  }

  // energy flux
  F[ceng] = U[ceng]*vtil + (P + 0.5*bsq)*vel - alpha*b[0]*Bcon[dir]/W;

  for (int n = cb_lo; n <= cb_hi; n++) {
    F[n] = U[n]*vtil - Bcon[dir]*vt[n-cb_lo];
  }
 
  const Real vasq = bsq*W*W/(rhohWsq+bsq);
  const Real cssq = gamma1*P*W*W/rhohWsq;
  Real cmsq = cssq + vasq - cssq*vasq;
  cmsq = (cmsq > 0.0 ? cmsq : 1.e-16); // TODO(JCD): what should this 1.e-16 be?
  cmsq = (cmsq > 1.0 ? 1.0 : cmsq);

  const Real gdd = gcon[d][d];
  const Real vcoff = alpha/(1. - vsq*cmsq);
  const Real v0 = vcon[dir]*(1.0 - cmsq);
  const Real vpm = sqrt(cmsq*(1.0  - vsq)*(gdd*(1.0 - vsq*cmsq) - vel*vel*(1.0 - cmsq)));
  vp = vcoff*(v0 + vpm) - beta[dir];
  vm = vcoff*(v0 - vpm) - beta[dir];

  //vp = vel + sqrt(cmsq);
 //vm = vel - sqrt(cmsq);
}

KOKKOS_FUNCTION
Real llf(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  fs.prim_to_flux(d, k, j, i, fs.ql, vml, vpl, Ul, Fl);
  fs.prim_to_flux(d, k, j, i, fs.qr, vmr, vpr, Ur, Fr);

  const Real cmax = std::max(std::max(-vml,vpl), std::max(-vmr,vpr));

  CellLocation loc = DirectionToFaceID(d);
  const Real gdet = fs.geom.DetGamma(loc, k, j, i);
  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = 0.5*(Fl[m] + Fr[m] - cmax*(Ur[m] - Ul[m])) * gdet;
  }
  return cmax;
}

KOKKOS_FUNCTION
Real hll(const FluxState &fs, const int d, const int k, const int j, const int i) {
  Real Ul[NCONS_MAX], Ur[NCONS_MAX];
  Real Fl[NCONS_MAX], Fr[NCONS_MAX];
  Real vml, vpl, vmr, vpr;

  fs.prim_to_flux(d, k, j, i, fs.ql, vml, vpl, Ul, Fl);
  fs.prim_to_flux(d, k, j, i, fs.qr, vmr, vpr, Ur, Fr);

  const Real cl = std::min(std::min(vml, vmr), 0.0);
  const Real cr = std::max(std::max(vpl, vpr), 0.0);

  CellLocation loc = DirectionToFaceID(d);
  const Real gdet = fs.geom.DetGamma(loc, k, j, i);
  for (int m = 0; m < fs.NumConserved(); m++) {
    fs.v.flux(d,m,k,j,i) = (cr*Fl[m] - cl*Fr[m] + cr*cl*(Ur[m] - Ul[m]))/(cr - cl) * gdet;
  }
  return std::max(-cl,cr);
}

} // namespace riemann
