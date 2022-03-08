#ifndef RADIATION_KD_GRID_HPP_
#define RADIATION_KD_GRID_HPP_

namespace radiation {

KOKKOS_INLINE_FUNCTION
void integrate_ninj_domega_quad(const Real xi0, const Real xi1, const Real phi0, const Real phi1, Real *wgts) {
  // expressions validated through numerical quadrature in python
  // xi = 1.0 - mu
  const Real cth0 = 1.0 - xi0;
  const Real sth0 = sqrt(1.0 - cth0*cth0);
  const Real cth1 = 1.0 - xi1;
  const Real sth1 = sqrt(1.0 - cth1*cth1);
  const Real sphi0 = sin(phi0);
  const Real cphi0 = cos(phi0);
  const Real sphi1 = sin(phi1);
  const Real cphi1 = cos(phi1);
  
  const Real sth03 = sth0*sth0*sth0;
  const Real sth13 = sth1*sth1*sth1;
  const Real cth03 = cth0*cth0*cth0;
  const Real cth13 = cth1*cth1*cth1;

  // cos(3 th) = 4 cos^3(th) - 3 cos(th)
  const Real c3th0 = 4.0*cth03 - 3.0*cth0;
  const Real c3th1 = 4.0*cth13 - 3.0*cth1;

  // cos(2 phi) = 2 cos^2(phi) - 1
  const Real c2phi0 = 2.0*cphi0*cphi0 - 1.0;
  const Real c2phi1 = 2.0*cphi1*cphi1 - 1.0;
  // sin(2 phi) = 2 sin(phi) cos(phi)
  const Real s2phi0 = 2.0*sphi0*cphi0;
  const Real s2phi1 = 2.0*sphi1*cphi1;

  Real term = 9.0*(cth0 - cth1) - c3th0 + c3th1;
  ind = geometry_utils::Flatten(0,0,3);
  wgts[ind] = (phi1 - phi0)*(3.0*cth0 + c3th0 - 4.0*cth13)/12.0;
  ind = geometry_utils::Flatten(0,1,3);
  wgts[ind] = (sphi0 - sphi1) * (sth03 - sth13)/3.0;
  ind = geometry_utils::Flatten(0,2,3);
  wgts[ind] = (cphi1 - cphi0) * (sth03 - sth13)/3.0;
  int ind = geometry_utils::Flatten2(1,1,3);
  wgts[ind] = -term*(2.0*(phi0 - phi1) + s2phi0 - s2phi1)/48.0;
  ind = geometry_utils::Flatten(1,2,3);
  wgts[ind] = (9.0*(cth0 - cth1) - c3th0 + c3th1) * (c2phi0 - c2phi1)/48.0;
  ind = geometry_utils::Flatten(2,2,3);
  wgts[ind] = term*(-2.0*(phi0 - phi1) + s2phi0 - s2phi1)/48.0;

}

}

#endif // RADIATION_KD_GRID_HPP_