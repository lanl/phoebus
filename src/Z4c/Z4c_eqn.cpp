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
Breif : Implementation for Z4c class with related constraint
Date : Jul.25.2023
Author : Hyun Lim

*/

#include "Z4c.hpp"

void Z4c::ADMConstraints(
  //Athena-GR way
  #if 0		
  AthenaArray<Real> & u_con, AthenaArray<Real> & u_adm,
  AthenaArray<Real> & u_mat, AthenaArray<Real> & u_z4c) {
  u_con.ZeroClear();

  Constraint_vars con;
  SetConstraintAliases(u_con, con);

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  Z4c_vars z4c;
  SetZ4cAliases(u_z4c, z4c);
  #endif

  ILOOP2(k,j) {
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = FD.Dx(c, adm.g_dd(a,b,k,j,i));
        dK_ddd(c,a,b,i) = FD.Dx(c, adm.K_dd(a,b,k,j,i));
      }
    }
    // second derivatives of g
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d) {
      if(a == b) {
        ILOOP1(i) {
          ddg_dddd(a,a,c,d,i) = FD.Dxx(a, adm.g_dd(c,d,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = FD.Dxy(a, b, adm.g_dd(c,d,k,j,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    ILOOP1(i) {
      detg(i) = SpatialDet(adm.g_dd,k,j,i);
      SpatialInv(1./detg(i),
          adm.g_dd(0,0,k,j,i), adm.g_dd(0,1,k,j,i), adm.g_dd(0,2,k,j,i),
          adm.g_dd(1,1,k,j,i), adm.g_dd(1,2,k,j,i), adm.g_dd(2,2,k,j,i),
          &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
          &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    }

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
      }
    }

    Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }

    Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    R.ZeroClear();
    R_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < NDIM; ++e) {
          ILOOP1(i) {
            R_dd(a,b,i) += g_uu(c,d,i) * Gamma_udd(e,a,c,i) * Gamma_ddd(e,b,d,i);
            R_dd(a,b,i) -= g_uu(c,d,i) * Gamma_udd(e,a,b,i) * Gamma_ddd(e,c,d,i);
          }
        }
        // Wave operator part of the Ricci
        ILOOP1(i) {
          R_dd(a,b,i) += 0.5*g_uu(c,d,i)*(
              - ddg_dddd(c,d,a,b,i) - ddg_dddd(a,b,c,d,i) +
                ddg_dddd(a,c,b,d,i) + ddg_dddd(b,c,a,d,i));
        }
      }
      ILOOP1(i) {
        R(i) += g_uu(a,b,i) * R_dd(a,b,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
        for(int c = 0; c < NDIM; ++c) {
          ILOOP1(i) {
            K_ud(a,b,i) += g_uu(a,c,i) * adm.K_dd(c,b,k,j,i);
          }
        }
      }
      ILOOP1(i) {
        K(i) += K_ud(a,a,i);
      }
    }
    // K^a_b K^b_a
    KK.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        KK(i) += K_ud(a,b,i) * K_ud(b,a,i);
      }
    }
    // Covariant derivative of K
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c) {
      ILOOP1(i) {
        DK_ddd(a,b,c,i) = dK_ddd(a,b,c,i);
      }
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,b,i) * adm.K_dd(d,c,k,j,i);
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,c,i) * adm.K_dd(b,d,k,j,i);
        }
      }
    }
    DK_udd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    ILOOP1(i) {
      con.H(k,j,i) = R(i) + SQR(K(i)) - KK(i) - 16*M_PI * mat.rho(k,j,i);
    }
    // Momentum constraint (contravariant)
    //
    M_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        M_u(a,i) -= 8*M_PI * g_uu(a,b,i) * mat.S_d(b,k,j,i);
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          M_u(a,i) += g_uu(a,b,i) * DK_udd(c,b,c,i);
          M_u(a,i) -= g_uu(b,c,i) * DK_udd(a,b,c,i);
        }
      }
    }
    // Momentum constraint (covariant)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.M_d(a,k,j,i) += adm.g_dd(a,b,k,j,i) * M_u(b,i);
      }
    }
    // Momentum constraint (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.M(k,j,i) += adm.g_dd(a,b,k,j,i) * M_u(a,i) * M_u(b,i);
      }
    }
    // Constraint violation Z (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.Z(k,j,i) += 0.25*adm.g_dd(a,b,k,j,i)*(z4c.Gam_u(a,k,j,i) - Gamma_u(a,i))
                                                *(z4c.Gam_u(b,k,j,i) - Gamma_u(b,i));
      }
    }
    // Constraint violation monitor C^2
    ILOOP1(i) {
      con.C(k,j,i) = SQR(con.H(k,j,i)) + con.M(k,j,i) + SQR(z4c.Theta(k,j,i)) + 4.0*con.Z(k,j,i);
    }
  }
}
