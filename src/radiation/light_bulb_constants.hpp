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

#ifndef LIGHT_BULB_CONSTANTS_HPP_
#define LIGHT_BULB_CONSTANTS_HPP_

namespace LightBulb {
namespace Liebendorfer {
constexpr Real A0 = 2.120282875020e+02;
constexpr Real A1 = -1.279083733386e+02;
constexpr Real A2 = 3.199984467460e+01;
constexpr Real A3 = -4.238440349145e+00;
constexpr Real A4 = 3.133904196302e-01;
constexpr Real A5 = -1.226365543366e-02;
constexpr Real A6 = 1.983947360151e-04;
constexpr Real LRHOMIN = 8.;
constexpr Real LRHOMAX = 13.;

  
// 14th order polynomial fit for T(rho); Useful to construct PNS.
// piecewise fit: high-density
constexpr Real BH0 = 1.05507273308e13;
constexpr Real BH1 = -6.99255736359e11;

//pievewise fit: low-density
constexpr Real BL0 = 2.66396927e9;
constexpr Real BL1 = 3.59155951e10; 
constexpr Real BL2 = 4.21460709e9; 
constexpr Real BL3 = -3.30158741e9;
constexpr Real BL4 = 2.10721665e7;
constexpr Real BL5 = 9.24399769e7; 
constexpr Real BL6 = -9.99999997e6;
constexpr Real BL7 = 3.10736981e5;

constexpr Real LOGRHO_TMAX = 14.437384726278943;
// 4th order polynomial fit for Ye(rho); Constructs PNS at t=1s after bounce. 
constexpr Real C0 = -10.67;
constexpr Real C1 = 4.158;
constexpr Real C2 = -0.5399;
constexpr Real C3 = 0.02862;
constexpr Real C4 = -0.0005247; 

} // namespace Liebendorfer

namespace HeatAndCool {

// to calculate tau
constexpr Real XL1 = 8.35;
constexpr Real XL2 = 8.8;
constexpr Real XL3 = 14.15;
constexpr Real XL4 = 14.5;
constexpr Real YL1 = -4.0;
constexpr Real YL2 = -1.8;
constexpr Real YL3 = 2.4;
constexpr Real YL4 = 3.4;

constexpr Real CFAC = 1.399e20;
constexpr Real HFAC = 1.544e20;
constexpr Real RNORM = 1.e7;
} // namespace HeatAndCool
} // namespace LightBulb

#endif
