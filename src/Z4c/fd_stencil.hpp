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

/*
Brief : Various finite difference stencil values
Date : Jul.19.2023
Author : Hyun Lim
*/

#ifndef FD_STENCIL_HPP_
#define FD_STENCIL_HPP_

//Define finite stencil template
// Centered finite difference stencil
// degree : Degree of derivative. E.g. 1 for the first derivative
// nghost : number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDCenteredStencil{
    public:
    //Degree of the derivative
    enum {degree = degree_};
    // Number of ghost points required for the differencing
    enum {nghost = nghost_};
    // Position at which the derivative is computed w.r.t the beginningof the stencil
    enum {offset = nghost_};
    // Width of the stencil
    enum {width = 2*nghost_ + 1{};
    // Finite difference coefficients
    static Real constexpr coeff[width];
}

// Define finite difference coefficients
// Centered finite differecing for first derivative
template<>
Real constexpr FDCenteredStencil<1, 1>::coeff[]= {
    -1./2., 0., 1./2.,
};

template<>
Real constexpr FDCenteredStencil<1, 2>::coeff[]= {
    -1./12., -2./3., 0., 2./3., -1./12.,
};

template<>
Real constexpr FDCenteredStencil<1, 3>::coeff[]= {
    -1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.,
};

template<>
Real const FDCenteredStencil<1, 4>::coeff[] = {
  1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.,
};

template<>
Real const FDCenteredStencil<1, 5>::coeff[] = {
  -1./1260., 5./504., -5./84., 5./21., -5./6., 0., 5./6., -5./21., 5./84., -5./504., 1./1260.,
};

template<>
Real const FDCenteredStencil<1, 6>::coeff[] = {
  1./5544.,-1./385.,1./56.,-5./63.,15./56.,-6./7.,0.,6./7.,-15./56.,5./63.,-1./56.,1./385.,-1./5544.
};

// Centered finite differencing for second derivative

template<>
Real const FDCenteredStencil<2, 1>::coeff[] = {
  1., -2., 1.,
};

template<>
Real const FDCenteredStencil<2, 2>::coeff[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.,
};

template<>
Real const FDCenteredStencil<2, 3>::coeff[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.,
};

template<>
Real const FDCenteredStencil<2, 4>::coeff[] = {
  -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.,
};

// add for testing
template<>
Real const FDCenteredStencil<2, 5>::coeff[] = {
  1./3150., -5./1008., 5./126., -5./21., 5./3., -5269./1800., 5./3., -5./21., 5./126., -5./1008., 1./3150.,
};

template<>
Real const FDCenteredStencil<2, 6>::coeff[] = {
  -1./16632., 2./1925., -1./112., 10./189., -15./56., 12./7., -5369./1800., 12./7., -15./56., 10./189., -1./112., 2./1925., -1./16632.,
};

template<>
Real const FDCenteredStencil<2, 7>::coeff[] = {
  1./84084., -7./30888., 7./3300., -7./528., 7./108., -7./24., 7./4., -266681./88200., 7./4., -7./24., 7./108., -7./528., 7./3300., -7./30888., 1./84084.,
};

template<>
Real const FDCenteredStencil<2, 8>::coeff[] = {
 -1./411840., 16./315315., -2./3861., 112./32175., -7./396., 112./1485., -14./45., 16./9., -1077749./352800., 16./9., -14./45., 112./1485., -7./396., 112./32175., -2./3861., 16./315315., -1./411840.,
};

// Higher orrder derivative operators for Kreiss-Oliger dissipation
template<>
Real const FDCenteredStencil<4, 2>::coeff[] = {
  1., -4., 6., -4., 1.,
};

template<>
Real const FDCenteredStencil<6, 3>::coeff[] = {
  1., -6., 15., -20., 15., -6., 1.,
};

template<>
Real const FDCenteredStencil<8, 4>::coeff[] = {
  1., -8., 28., -56., 70., -56., 28., -8., 1.,
};

template<>
Real const FDCenteredStencil<10, 5>::coeff[] = {
  1., -10., 45., -120., 210., -252., 210., -120., 45., -10., 1.,
};

template<>
Real const FDCenteredStencil<12, 6>::coeff[] = {
  1., -12., 66., -220., 495., -792., 924., -792., 495., -220., 66., -12., 1.
};

template<>
Real const FDCenteredStencil<14, 7>::coeff[] = {
  1.,-14.,91.,-364.,1001.,-2002.,3003.,-3432.,3003.,-2002.,1001.,-364.,91.,-14.,1.
};



#endif // FD_STENCIL_HPP_
