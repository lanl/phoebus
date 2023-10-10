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
Brief : Various finite difference computations for handling
        spatial derivative.
Date : Jul.19.2023
Author : Hyun Lim
*/

#ifndef FD_COMPUTE_HPP_
#define FD_COMPUTE_HPP_

#include "fd_stencil.hpp"

//TODO : Coupling parthenon looping i.e. Kokkos
private:
  struct {
    // 1st derivative stecil
    typedef FDCenteredStencil<1, NGHOST-1> s1;
    // 2nd derivative stencil
    typedef FDCenteredStencil<2, NGHOST-1> s2;
    // dissipation operator
    typedef FDCenteredStencil<
      FDDissChoice<NGHOST-1>::degree,
      FDDissChoice<NGHOST-1>::nghost
      > sd;
    // left-biased derivative
    typedef FDLeftBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sl;
    // right-biased derivative
    typedef FDRightBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sr;

    int stride[3];
    Real idx[3];
    Real diss;

    // 1st derivative (high order centered)
    inline Real Dx(int dir, Real & u) {
      Real * pu = &u - s1::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s1::nghost; ++n1) {
        int const n2  = s1::width - n1 - 1;
        Real const c1 = s1::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s1::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s1::coeff[s1::nghost] * pu[s1::nghost*stride[dir]];
      return out * idx[dir];
    }
    // 1st derivative 2nd order centered
    inline Real Ds(int dir, Real & u) {
      Real * pu = &u;
      return 0.5 * idx[dir] * (pu[stride[dir]] - pu[-stride[dir]]);
    }
    // Advective derivative
    // The advective derivative is for an equation in the form
    //    d_t u = vx d_x u
    // So negative vx means advection from the *left* to the *right*, so we use
    // *left* biased FD stencils
    inline Real Lx(int dir, Real & vx, Real & u) {
      Real * pu = &u;

      Real dl(0.);
      for(int n = 0; n < sl::width; ++n) {
        dl += sl::coeff[n] * pu[(n - sl::offset)*stride[dir]];
      }

      Real dr(0.);
      for(int n = sr::width-1; n >= 0; --n) {
        dr += sr::coeff[n] * pu[(n - sr::offset)*stride[dir]];
      }
      return ((vx < 0) ? (vx * dl) : (vx * dr)) * idx[dir];
    }
    // Homogeneous 2nd derivative
    inline Real Dxx(int dir, Real & u) {
      Real * pu = &u - s2::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s2::nghost; ++n1) {
        int const n2  = s2::width - n1 - 1;
        Real const c1 = s2::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s2::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s2::coeff[s2::nghost] * pu[s2::nghost*stride[dir]];
      return out * SQR(idx[dir]);
    }
    // Mixed 2nd derivative
    inline Real Dxy(int dirx, int diry, Real & u) {
      Real * pu = &u - s1::offset*(stride[dirx] + stride[diry]);
      Real out(0.);

      for(int nx1 = 0; nx1 < s1::nghost; ++nx1) {
        int const nx2 = s1::width - nx1 - 1;
        for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
          int const ny2 = s1::width - ny1 - 1;

          Real const c11 = s1::coeff[nx1] * s1::coeff[ny1] * pu[nx1*stride[dirx] + ny1*stride[diry]];
          Real const c12 = s1::coeff[nx1] * s1::coeff[ny2] * pu[nx1*stride[dirx] + ny2*stride[diry]];
          Real const c21 = s1::coeff[nx2] * s1::coeff[ny1] * pu[nx2*stride[dirx] + ny1*stride[diry]];
          Real const c22 = s1::coeff[nx2] * s1::coeff[ny2] * pu[nx2*stride[dirx] + ny2*stride[diry]];

          Real const ca = (1./6.)*((c11 + c12) + (c21 + c22));
          Real const cb = (1./6.)*((c11 + c21) + (c12 + c22));
          Real const cc = (1./6.)*((c11 + c22) + (c12 + c21));

          out += ((ca + cb) + cc) + ((ca + cc) + cb);
        }
        int const ny = s1::nghost;

        Real const c1 = s1::coeff[nx1] * s1::coeff[ny] * pu[nx1*stride[dirx] + ny*stride[diry]];
        Real const c2 = s1::coeff[nx2] * s1::coeff[ny] * pu[nx2*stride[dirx] + ny*stride[diry]];

        out += (c1 + c2);
      }
      int const nx = s1::nghost;
      for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
        int const ny2 = s1::width - ny1 - 1;

        Real const c1 = s1::coeff[nx] * s1::coeff[ny1] * pu[nx*stride[dirx] + ny1*stride[diry]];
        Real const c2 = s1::coeff[nx] * s1::coeff[ny2] * pu[nx*stride[dirx] + ny2*stride[diry]];

        out += (c1 + c2);
      }
            int const nx = s1::nghost;
      for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
        int const ny2 = s1::width - ny1 - 1;

        Real const c1 = s1::coeff[nx] * s1::coeff[ny1] * pu[nx*stride[dirx] + ny1*stride[diry]];
        Real const c2 = s1::coeff[nx] * s1::coeff[ny2] * pu[nx*stride[dirx] + ny2*stride[diry]];

        out += (c1 + c2);
      }
      int const ny = s1::nghost;
      out += s1::coeff[nx] * s1::coeff[ny] * pu[nx*stride[dirx] + ny*stride[diry]];

      return out * idx[dirx] * idx[diry];
    }
    // Kreiss-Oliger dissipation operator
    inline Real Diss(int dir, Real & u) {
      Real * pu = &u - sd::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < sd::nghost; ++n1) {
        int const n2  = sd::width - n1 - 1;
        Real const c1 = sd::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = sd::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += sd::coeff[sd::nghost] * pu[sd::nghost*stride[dir]];

      return out * idx[dir] * diss;
    }
  } FD;

#endif // FD_COMPUTE_HPP_
