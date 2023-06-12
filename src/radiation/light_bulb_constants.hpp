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
