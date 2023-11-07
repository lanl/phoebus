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

  //TODO : make better
constexpr size_t scratch_level = 1; // can be 0, 1, or 2. Depends on hardware. 0 is lower level cache usually. 2 is highest level (maybe main memory?)
// gets allocation size for scratch memory for rank 2 tensor along pencil in i
const auto scratch_size = parthenon::ScratchPad3D<Real>::shmem_size(NDIM, NDIM, Ni) /* + more sizes if desired */;

// make loop...
parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "loop name", DevExecSpace(),
  // scratch declarations
  scratch_size, scratch_level,
  // loop bounds
  0, nblocks-1, kb.s, kb.e,
  // lambda capture
  KOKKOS_LAMBDA(parthenon::team_member_t member, const int b, const int k) {
  // loop body

  // request scratch memory.  Can now be used as a multid array.
  ScratchPad3D<Real> ginverse(member.team_scratch(scratch_level), NDIM, NDIM, Ni);
  // can keep declaring scratch arrays up to as much memory as requested in scratch_size above

  // do stuff...
});


  ILOOP2(md, scratch_szie, scratch_level,b, k, j) {
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

    //Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }

    //Gamma_u.ZeroClear();
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
    //R.ZeroClear();
    //R_dd.ZeroClear();
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
    //K.ZeroClear();
    //K_ud.ZeroClear();
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
    //KK.ZeroClear();
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
    //DK_udd.ZeroClear();
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
    //M_u.ZeroClear();
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
  }); //end of ILOOP2 //TODO: check
}

TaskStatus ComputeRHS(MeshData<Real> *md_state, MeshData<Real> *md_rhs) {
  //TODO: boiler plate crap above
  auto state = desc.GetPack(md_state);
  auto rhs = desc.GetPack(md_rhs);

  // in loop...
  ILOOP2(md_state, scratch_size, scratch_level, b, k, j) {
        // -----------------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dalpha_d(a,i) = FD.Dx(a, z4c.alpha(k,j,i));
        dchi_d(a,i)   = FD.Dx(a, z4c.chi(k,j,i));
        dKhat_d(a,i)  = FD.Dx(a, z4c.Khat(k,j,i));
        dTheta_d(a,i) = FD.Dx(a, z4c.Theta(k,j,i));
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        dbeta_du(b,a,i) = FD.Dx(b, z4c.beta_u(a,k,j,i));
        dGam_du(b,a,i)  = FD.Dx(b, z4c.Gam_u(a,k,j,i));
      }
    }
    // Tensors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = FD.Dx(c, z4c.g_dd(a,b,k,j,i));
        dA_ddd(c,a,b,i) = FD.Dx(c, z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // 2nd derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        ddalpha_dd(a,a,i) = FD.Dxx(a, z4c.alpha(k,j,i));
        ddchi_dd(a,a,i) = FD.Dxx(a, z4c.chi(k,j,i));
      }
      for(int b = a + 1; b < NDIM; ++b) {
        ILOOP1(i) {
          ddalpha_dd(a,b,i) = FD.Dxy(a, b, z4c.alpha(k,j,i));
          ddchi_dd(a,b,i) = FD.Dxy(a, b, z4c.chi(k,j,i));
        }
      }
    }
        // Vectors
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      if(a == b) {
        ILOOP1(i) {
          ddbeta_ddu(a,b,c,i) = FD.Dxx(a, z4c.beta_u(c,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddbeta_ddu(a,b,c,i) = FD.Dxy(a, b, z4c.beta_u(c,k,j,i));
        }
      }
    }
    // Tensors
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      if(a == b) {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = FD.Dxx(a, z4c.g_dd(c,d,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = FD.Dxy(a, b, z4c.g_dd(c,d,k,j,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Advective derivatives
    //
    // Scalars
    //Lalpha.ZeroClear();
    //Lchi.ZeroClear();
    //LKhat.ZeroClear();
    //LTheta.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        Lalpha(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.alpha(k,j,i));
        Lchi(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.chi(k,j,i));
        LKhat(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Khat(k,j,i));
        LTheta(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Theta(k,j,i));
      }
    }
    // Vectors
    //Lbeta_u.ZeroClear();
    //LGam_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Lbeta_u(b,i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.beta_u(b,k,j,i));
        LGam_u(b,i)  += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Gam_u(b,k,j,i));
      }
    }
    // Tensors
    //Lg_dd.ZeroClear();
    //LA_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Lg_dd(a,b,i) += FD.Lx(c, z4c.beta_u(c,k,j,i), z4c.g_dd(a,b,k,j,i));
        LA_dd(a,b,i) += FD.Lx(c, z4c.beta_u(c,k,j,i), z4c.A_dd(a,b,k,j,i));
      }
    }
    
        // -----------------------------------------------------------------------------------
    // Get K from Khat
    //
    ILOOP1(i) {
      K(i) = z4c.Khat(k,j,i) + 2.*z4c.Theta(k,j,i);
    }

    // -----------------------------------------------------------------------------------
    // Inverse metric
    //
    ILOOP1(i) {
      detg(i) = SpatialDet(z4c.g_dd, k, j, i);
      SpatialInv(1.0/detg(i),
          z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
          z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
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
    //Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }
    // Gamma's computed from the conformal metric (not evolved)
    //Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }
    // -----------------------------------------------------------------------------------
    // Curvature of conformal metric
    //
    //R_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          R_dd(a,b,i) += 0.5*(z4c.g_dd(c,a,k,j,i)*dGam_du(b,c,i) +
                              z4c.g_dd(c,b,k,j,i)*dGam_du(a,c,i) +
                              Gamma_u(c,i)*(Gamma_ddd(a,b,c,i) + Gamma_ddd(b,a,c,i)));
        }
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          R_dd(a,b,i) -= 0.5*g_uu(c,d,i)*ddg_dddd(c,d,a,b,i);
        }
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d)
      for(int e = 0; e < NDIM; ++e) {
        ILOOP1(i) {
          R_dd(a,b,i) += g_uu(c,d,i)*(
              Gamma_udd(e,c,a,i)*Gamma_ddd(b,e,d,i) +
              Gamma_udd(e,c,b,i)*Gamma_ddd(a,e,d,i) +
              Gamma_udd(e,a,d,i)*Gamma_ddd(e,c,b,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Derivatives of conformal factor phi
    //
    ILOOP1(i) {
      chi_guarded(i) = std::max(z4c.chi(k,j,i), opt.chi_div_floor);
      oopsi4(i) = pow(chi_guarded(i), -4./opt.chi_psi_power);
    }

    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dphi_d(a,i) = dchi_d(a,i)/(chi_guarded(i) * opt.chi_psi_power);
      }
    }
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Real const ddphi_ab = ddchi_dd(a,b,i)/(chi_guarded(i) * opt.chi_psi_power) -
          opt.chi_psi_power * dphi_d(a,i) * dphi_d(b,i);
        Ddphi_dd(a,b,i) = ddphi_ab;
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Ddphi_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dphi_d(c,i);
        }
      }
    }
        // -----------------------------------------------------------------------------------
    // Curvature contribution from conformal factor
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Rphi_dd(a,b,i) = 4.*dphi_d(a,i)*dphi_d(b,i) - 2.*Ddphi_dd(a,b,i);
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          Rphi_dd(a,b,i) -= 2.*z4c.g_dd(a,b,k,j,i) * g_uu(c,d,i)*(Ddphi_dd(c,d,i) +
              2.*dphi_d(c,i)*dphi_d(d,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Trace of the matter stress tensor
    //
    //S.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        S(i) += oopsi4(i) * g_uu(a,b,i) * mat.S_dd(a,b,k,j,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // 2nd covariant derivative of the lapse
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Ddalpha_dd(a,b,i) = ddalpha_dd(a,b,i)
                          - 2.*(dphi_d(a,i)*dalpha_d(b,i) + dphi_d(b,i)*dalpha_d(a,i));
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Ddalpha_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dalpha_d(c,i);
        }
        for(int d = 0; d < NDIM; ++d) {
          ILOOP1(i) {
            Ddalpha_dd(a,b,i) += 2.*z4c.g_dd(a,b,k,j,i) * g_uu(c,d,i) * dphi_d(c,i) * dalpha_d(d,i);
          }
        }
      }
    }

    //Ddalpha.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Ddalpha(i) += oopsi4(i) * g_uu(a,b,i) * Ddalpha_dd(a,b,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Contractions of A_ab, inverse, and derivatives
    //
    //AA_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        AA_dd(a,b,i) += g_uu(c,d,i) * z4c.A_dd(a,c,k,j,i) * z4c.A_dd(d,b,k,j,i);
      }
    }
    //AA.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        AA(i) += g_uu(a,b,i) * AA_dd(a,b,i);
      }
    }
    //A_uu.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        A_uu(a,b,i) += g_uu(a,c,i) * g_uu(b,d,i) * z4c.A_dd(c,d,k,j,i);
      }
    }
    //DA_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          DA_u(a,i) -= (3./2.) * A_uu(a,b,i) * dchi_d(b,i) / chi_guarded(i);
          DA_u(a,i) -= (1./3.) * g_uu(a,b,i) * (2.*dKhat_d(b,i) + dTheta_d(b,i));
        }
      }
      for(int b = 0; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          DA_u(a,i) += Gamma_udd(a,b,c,i) * A_uu(b,c,i);
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci scalar
    //
    //R.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        R(i) += oopsi4(i) * g_uu(a,b,i) * (R_dd(a,b,i) + Rphi_dd(a,b,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // Hamiltonian constraint
    //
    ILOOP1(i) {
      Ht(i) = R(i) + (2./3.)*SQR(K(i)) - AA(i);
    }

    // -----------------------------------------------------------------------------------
    // Finalize advective (Lie) derivatives
    //
    // Shift vector contractions
    //dbeta.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dbeta(i) += dbeta_du(a,a,i);
      }
    }
    //ddbeta_d.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        ddbeta_d(a,i) += (1./3.) * ddbeta_ddu(a,b,b,i);
      }
    }
    // Finalize Lchi
    ILOOP1(i) {
      Lchi(i) += (1./6.) * opt.chi_psi_power * chi_guarded(i) * dbeta(i);
    }
    // Finalize LGam_u (note that this is not a real Lie derivative)
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        LGam_u(a,i) += (2./3.) * Gamma_u(a,i) * dbeta(i);
      }
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          LGam_u(a,i) += g_uu(a,b,i) * ddbeta_d(b,i) - Gamma_u(b,i) * dbeta_du(b,a,i);
        }
        for(int c = 0; c < NDIM; ++c) {
          ILOOP1(i) {
            LGam_u(a,i) += g_uu(b,c,i) * ddbeta_ddu(b,c,a,i);
          }
        }
      }
    }
    // Finalize Lg_dd and LA_dd
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Lg_dd(a,b,i) -= (2./3.) * z4c.g_dd(a,b,k,j,i) * dbeta(i);
        LA_dd(a,b,i) -= (2./3.) * z4c.A_dd(a,b,k,j,i) * dbeta(i);
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Lg_dd(a,b,i) += dbeta_du(a,c,i) * z4c.g_dd(b,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(b,c,i) * z4c.A_dd(a,c,k,j,i);
          Lg_dd(a,b,i) += dbeta_du(b,c,i) * z4c.g_dd(a,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(a,c,i) * z4c.A_dd(b,c,k,j,i);
        }
      }
    }

    // RHS asseble goes here
    SPACETIMELOOP2(mu, nu) {
      Real *At_rhs = &rhs(b, At(flatten(mu, nu), k, j);
         parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e,
         [&](const int i) {
             At_rhs[i] = /* some nonsense */;
         });
    }
  })//end of ILOOP2 //TODO: check;

  return TaskStatus::complete;
}
