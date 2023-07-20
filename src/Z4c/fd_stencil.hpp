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
Real constexpr FDCenteredStencil<1, 4>::coeff[]= {
    -1./12., -2./3., 0., 2./3., -1./12.,
};

template<>
Real constexpr FDCenteredStencil<1, 5>::coeff[]= {
    -1./12., -2./3., 0., 2./3., -1./12.,
};

template<>
Real constexpr FDCenteredStencil<1, 5>::coeff[]= {
    -1./12., -2./3., 0., 2./3., -1./12.,
};

// Centered finite differencing for second derivative


// Higher orrder derivative operators for Kreiss-Oliger dissipation
#endif // FD_STENCIL_HPP_
