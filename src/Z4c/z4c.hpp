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
Breif: General header for Z4c formulation
       In this impelementation, we follow the Z4c formulation that is reported in
       Bernuzzi & Hilditch (2010) and Hilditch et al. (2013).
       The Z4c system is comprised of dynamical variable:
       chi: Conformal factor (scalar)
       g : Induced 3-metric (rank-2)
       Khat : Trace of extrinsic curvature (scalar)
       A : Traceless of extrinsic curvature (rank-2)
       Theta : Temporal projection of Z4 vector (scalar)
       Gam : Conformaly-related function (3-vector)

       In addtion, we follow typical gauge choice:
       Bona-Masso for lapse, alpha (scalar) \TODO :maybe 1+log?
       Gamma-driver for shift, beta (3-vector) and B (3-vector)       
Date : Jul.19.2023
Author : Hyun Lim
*/

#ifndef Z4C_HPP_
#define Z4C_HPP_

#include "fd_compute.hpp"
#include "z4c_eqn.cpp"

//TODO : Link to Phoebus
#include <parthenon/package.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace z4c { //TODO: check consistencty with namespace everywhere
  namespace constraint{
  VARIABLE(z4c.c, H); // Hamiltoninan constraint
  VARIABLE(z4c.c, M); // Momentum constraint
  VARIABLE(z4c.c, Z); // Z-vector constraint
  } // constraint
  namespace evolution_z4c{
  VARIABLE(z4c.c, chi); // Conformal factor
  VARIABLE(z4c.c, g); // Induced 3 metric
  VARIABLE(z4c.c, Khat); // Trace of extrinsic curvature
  VARIABLE(z4c.c, A); // Traceless of extrinsic curvature
  VARIABLE(z4c.c, Gam); // Conformally related function 
  VARIABLE(z4c.c, Theta); // Temporal projection of Z4 vector
  VARIABLE(z4c.c, alpha); // Lapse
  VARIABLE(z4c.c, beta); // Shift
  } // evolution_z4c
  namespace evolution_adm{
  VARIABLE(z4c.c, g); // Induced 3 metric
  VARIABLE(z4c.c, K); // Extrinsic curvature
  VARIABLE(z4c.c, psi); // Psi scalar function
  } // evolution_adm
  namespace matter_source{ // TODO : this is not really the varialbe. We may not want them as like this
  VARIABLE(z4c.c, rho); 
  VARIABLE(z4c.c, Si); 
  VARIABLE(z4c.c, Sij);
  } // matter_source

  // TODO: add psi4 for GW extraction

  // Prototype for package init
  std::shared_ptr<parthenon::StateDescriptor> Initialize(ParameterInput *pin);

} // z4c

// Some precomputation
// compute spatial determinant of a 3x3  matrix
inline Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
    Real const gyy, Real const gyz, Real const gzz) {
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) - SQR(gxy)*gzz + gxx*gyy*gzz;
}
inline Real SpatialDet(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & g,
                int const k, int const j, int const i) {
  return SpatialDet(g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
                    g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i));
}
// compute inverse of a 3x3 matrix
 inline void SpatialInv(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real * uxx, Real * uxy, Real * uxz,
                Real * uyy, Real * uyz, Real * uzz) {
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
}
// compute trace of a rank 2 covariant spatial tensor
inline Real Trace(Real const detginv,
           Real const gxx, Real const gxy, Real const gxz,
           Real const gyy, Real const gyz, Real const gzz,
           Real const Axx, Real const Axy, Real const Axz,
           Real const Ayy, Real const Ayz, Real const Azz) {
  return (detginv*(
     - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
     + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
     - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
     ));
}



#endif // Z4C_HPP_
