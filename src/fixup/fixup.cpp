#include "fixup.hpp"

#include <singularity-eos/eos/eos.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

// TODO(BRR) temp
#include "fluid/fluid.hpp"

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto fix = std::make_shared<StateDescriptor>("fixup");
  Params &params = fix->AllParams();

  bool enable_flux_fixup = pin->GetOrAddBoolean("fixup", "enable_flux_fixup", false);
  params.Add("enable_flux_fixup", enable_flux_fixup);
  bool enable_fixup = pin->GetOrAddBoolean("fixup", "enable_fixup", false);
  params.Add("enable_fixup", enable_fixup);
  bool enable_floors = pin->GetOrAddBoolean("fixup", "enable_floors", false);
  params.Add("enable_floors", enable_floors);
  bool enable_c2p_fixup = pin->GetOrAddBoolean("fixup", "enable_c2p_fixup", false);
  params.Add("enable_c2p_fixup", enable_c2p_fixup);
  bool enable_ceilings = pin->GetOrAddBoolean("fixup", "enable_ceilings", false);
  params.Add("enable_ceilings", enable_ceilings);

  if (enable_floors) {
    const std::string floor_type = pin->GetString("fixup", "floor_type");
    if (floor_type == "ConstantRhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_floor", 0.0);
      params.Add("floor", Floors(constant_rho_sie_floor_tag, rho0, sie0));
    } else if (floor_type == "ExpX1RhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "sie_exp_floor", -1.0);
      params.Add("floor", Floors(exp_x1_rho_sie_floor_tag, rho0, sie0, rp, sp));
    } else {
      PARTHENON_FAIL("invalid <fixup>/floor_type input");
    }
  } else {
    params.Add("floor", Floors());
  }

  if (enable_ceilings) {
    const std::string ceiling_type = pin->GetOrAddString("fixup", "ceiling_type", "ConstantGamSie");
    if (ceiling_type == "ConstantGamSie") {
      Real gam0 = pin->GetOrAddReal("fixup", "gam0_ceiling", 1000.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0_ceiling", 100.0);
      params.Add("ceiling", Ceilings(constant_gam_sie_ceiling_tag, gam0, sie0));
    } else {
      PARTHENON_FAIL("invalid <fixup>/ceiling_type input");
    }
  } else {
    params.Add("ceiling", Ceilings());
  }

  params.Add("bounds", Bounds(params.Get<Floors>("floor"), params.Get<Ceilings>("ceiling")));

  return fix;
}

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  if (!enable_c2p_fixup) return TaskStatus::complete;

  const std::vector<std::string> vars({p::density, c::density, p::velocity,
                                       c::momentum, p::energy, c::energy, p::bfield,
                                       p::ye, c::ye, p::pressure, p::temperature,
                                       p::gamma1, impl::cell_signal_speed, impl::fail});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int tmp = imap[p::temperature].first;
  const int gm1 = imap[p::gamma1].first;
  const int slo = imap[impl::cell_signal_speed].first;
  const int shi = imap[impl::cell_signal_speed].second;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  int pye = imap[p::ye].second; // negative if not present
  int cye = imap[c::ye].second;
  int ifail = imap[impl::fail].first;

  const int ndim = pmb->pmy_mesh->ndim;

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  
  auto bounds = fix_pkg->Param<Bounds>("bounds");
        
  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  // Ben's second attempt
  
  int nfail_total;
  parthenon::par_reduce(
      //DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
        if (v(b,ifail,k,j,i) == con2prim_robust::FailFlags::fail) {
          nf++;
          
        }
      }, Kokkos::Sum<int>(nfail_total));
  printf("total nfail: %i\n", nfail_total);
  
  int nfixed_total = 0;
  auto fail = rc->Get(impl::fail).data;
  auto tmp_fail = fail.GetDeviceMirror();
  while (nfixed_total < nfail_total) {
    // Copy ifail array
    tmp_fail.DeepCopy(fail);


    int nfixed = 0;
    parthenon::par_reduce(
        //DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
        parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup", DevExecSpace(),
        0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
        auto fixup = [&](const int iv) {
          //v(b,iv,k,j,i)  = v(b,ifail,k,j,i-1)*v(b,iv,k,j,i-1)
          //               + v(b,ifail,k,j,i+1)*v(b,iv,k,j,i+1);
          //v(b,iv,k,j,i) += v(b,ifail,k,j-1,i)*v(b,iv,k,j-1,i)
          //               + v(b,ifail,k,j+1,i)*v(b,iv,k,j+1,i);
          Real wsum = 0.;
          Real sum = 0.;
          for (int l = -1; l < 2; l++) {
            for (int m = -1; m < 2; m++) {
              for (int n = 0; n < 1; n++) {
                Real w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                wsum += w;
                sum += w*v(b,iv,k+n,j+m,i+l);
              }
            }
          }
          if (isnan(sum)) {
            printf("SUM IS NAN! [%i] [%i %i %i]\n", iv, k, j, i);
            for (int l = -1; l < 2; l++) {
              for (int m = -1; m < 2; m++) {
                for (int n = 0; n < 1; n++) {
                  Real w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                  //wsum += w;
                  Real dsum = w*v(b,iv,k+n,j+m,i+l);
                  printf("[%i %i %i] [%i %i %i %i] w = %e dsum = %e\n",
                    l, m, n, iv,k+n,j+m,i+l,w, dsum);
                }
              }
            }
            PARTHENON_FAIL("nan sum");
          }
          return sum/wsum;
        };
        if (v(b,ifail,k,j,i) == con2prim_robust::FailFlags::fail) {
          Real wsum = 0.;
          for (int l = -1; l < 2; l++) {
            for (int m = -1; m < 2; m++) {
              for (int n = 0; n < 1; n++) {
                Real w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                wsum += w;
              }
            }
          }
          
          if (wsum < 1.e-10) {
            // No usable neighbors this iteration

          } else {

            v(b,prho,k,j,i) = fixup(prho);
            v(b,peng,k,j,i) = fixup(peng);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b,pv,k,j,i) = fixup(pv);
            }

            // Apply floors
            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i), rho_floor, sie_floor);
            v(b,prho,k,j,i) = v(b,prho,k,j,i) > rho_floor ? v(b,prho,k,j,i) : rho_floor;
            double u_floor = v(b,prho,k,j,i)*sie_floor;
            v(b,peng,k,j,i) = v(b,peng,k,j,i) > u_floor ? v(b,peng,k,j,i) : u_floor; 

            // Apply ceilings
            
            // Limit gammasq
            double gammacov[3][3];
            geom.Metric(CellLocation::Cent, k, j, i, gammacov);
            double vsq = 0.; 
            // TODO(BRR) what about pvel_hi
            SPACELOOP(m) {
              SPACELOOP(n) {
                vsq += gammacov[m][n] * v(b,pvel_lo+m,k,j,i) * v(b,pvel_lo+n,k,j,i);
              }
            }
            double gammamax = 50.;
            double vsqmax = 1. - 1./(gammamax*gammamax);
            if (vsq > vsqmax) {
              double vfac = sqrt(vsqmax/vsq); 
              v(b,pvel_lo,k,j,i) *= vfac;
              v(b,pvel_lo+1,k,j,i) *= vfac;
              v(b,pvel_lo+2,k,j,i) *= vfac;
            }

              double vfac = sqrt(vsqmax/vsq); 
            //if (vsq > vsqmax || isnan(vfac)) {
            //  printf("v: %e %e %e vfac: %e vsqmax: %e vsqL %e\n", v(b,pvel_lo,k,j,i),
            //    v(b,pvel_lo+1,k,j,i), v(b,pvel_lo+2,k,j,i), vfac, vsqmax, vsq);
            //}
            
            // Set other primitives
            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i); 
              
            // Reset conserved variables
            const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
            const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
            Real beta[3];
            geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
            Real gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
            Real gcon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
            Real S[3];
            const Real vel[] = {v(b, pvel_lo, k, j, i),
                                v(b, pvel_lo+1, k, j, i),
                                v(b, pvel_hi, k, j, i)};
            Real bcons[3];
            Real bp[3] = {0.0, 0.0, 0.0};
            if (pb_hi > 0) {
              bp[0] = v(b, pb_lo, k, j, i);
              bp[1] = v(b, pb_lo+1, k, j, i);
              bp[2] = v(b, pb_hi, k, j, i);
            }
            Real ye_cons;
            Real ye_prim = 0.0;
            if (pye > 0) {
              ye_prim = v(b, pye, k, j, i);
            }
            Real sig[3];
            prim2con::p2c(v(b,prho,k,j,i), vel, bp, v(b,peng,k,j,i), ye_prim, v(b,prs,k,j,i), 
                  v(b,gm1,k,j,i), gcov, gcon, beta, alpha, gdet,
                  v(b,crho,k,j,i), S, bcons, v(b,ceng,k,j,i), ye_cons, sig);
              v(b, cmom_lo, k, j, i) = S[0];
              v(b, cmom_lo+1, k, j, i) = S[1];
              v(b, cmom_hi, k, j, i) = S[2];
              if (pye > 0) v(b, cye, k, j, i) = ye_cons;
              for (int m = slo; m <= shi; m++) {
                v(b,m,k,j,i) = sig[m-slo];
              }

            tmp_fail(k, j, i) = con2prim_robust::FailFlags::success;
            nf++;
          }
        }
      }, Kokkos::Sum<int>(nfixed));
    //printf("nfixed: %i\n", nfixed);
    nfixed_total += nfixed;
    fail.DeepCopy(tmp_fail);
  }
  
  //parthenon::par_for(
  //    DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
  //    0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
  //    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
  //      PARTHENON_REQUIRE(!isnan(v(b,crho,k,j,i)), "crho still nan!");
  //    });
  
  int nnan;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nn) {
        if (isnan(v(b,prho,k,j,i))) {
          nn++;
          
        }
      }, Kokkos::Sum<int>(nnan));
  pmb->exec_space.fence();
  if (nnan > 0) {
    printf("%i nans!\n", nnan);
    exit(-1);
  }
  //exit(-1);
            
            
  // Apply floors everywhere
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            // Apply floors
            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i), rho_floor, sie_floor);
            
            v(b,prho,k,j,i) = v(b,prho,k,j,i) > rho_floor ? v(b,prho,k,j,i) : rho_floor;
            double u_floor = v(b,prho,k,j,i)*sie_floor;
            v(b,peng,k,j,i) = v(b,peng,k,j,i) > u_floor ? v(b,peng,k,j,i) : u_floor; 
            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i); 
            
            // P -> C after floors
            const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
            const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
            Real beta[3];
            geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
            Real gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
            Real gcon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
            Real S[3];
            const Real vel[] = {v(b, pvel_lo, k, j, i),
                                v(b, pvel_lo+1, k, j, i),
                                v(b, pvel_hi, k, j, i)};
            Real bcons[3];
            Real bp[3] = {0.0, 0.0, 0.0};
            if (pb_hi > 0) {
              bp[0] = v(b, pb_lo, k, j, i);
              bp[1] = v(b, pb_lo+1, k, j, i);
              bp[2] = v(b, pb_hi, k, j, i);
            }
            Real ye_cons;
            Real ye_prim = 0.0;
            if (pye > 0) {
              ye_prim = v(b, pye, k, j, i);
            }
            Real sig[3];
            prim2con::p2c(v(b,prho,k,j,i), vel, bp, v(b,peng,k,j,i), ye_prim, v(b,prs,k,j,i), v(b,gm1,k,j,i),
                gcov, gcon, beta, alpha, gdet,
                v(b,crho,k,j,i), S, bcons, v(b,ceng,k,j,i), ye_cons, sig);
            v(b, cmom_lo, k, j, i) = S[0];
            v(b, cmom_lo+1, k, j, i) = S[1];
            v(b, cmom_hi, k, j, i) = S[2];
            if (pye > 0) v(b, cye, k, j, i) = ye_cons;
            for (int m = slo; m <= shi; m++) {
              v(b,m,k,j,i) = sig[m-slo];
            }
      });

  // TODO(BRR) only do this where necessary

  return TaskStatus::complete;

  // Ben's attempt
  /*parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto fixup = [&](const int iv, const bool weighted) {
          v(b,iv,k,j,i)  = v(b,ifail,k,j,i-1)*v(b,iv,k,j,i-1)
                         + v(b,ifail,k,j,i+1)*v(b,iv,k,j,i+1);
            v(b,iv,k,j,i) += v(b,ifail,k,j-1,i)*v(b,iv,k,j-1,i)
                           + v(b,ifail,k,j+1,i)*v(b,iv,k,j+1,i);
          Real wsum = 0.;
          Real sum = 0.;
          for (int l = -1; l < 2; l++) {
            for (int m = -1; m < 2; m++) {
              for (int n = 0; n < 1; n++) {
                Real w;
                if (weighted) {
                  w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                } else {
                  w = 1./(abs(l) + abs(m) + abs(n) + 1);
                }
                wsum += w;
                sum += w*v(b,iv,k+n,j+m,i+l);
              }
            }
          }
          return sum/wsum;
        };
        if (v(b,ifail,k,j,i) == con2prim_robust::FailFlags::fail) {
          Real wsum = 0.;
          for (int l = -1; l < 2; l++) {
            for (int m = -1; m < 2; m++) {
              for (int n = 0; n < 1; n++) {
                Real w = 1./(abs(l) + abs(m) + abs(n) + 1)*v(b,ifail,k+n,j+m,i+l);
                wsum += w;
              }
            }
          }
          if (wsum > 1.e-10) {
            v(b,prho,k,j,i) = fixup(prho, true);
            v(b,peng,k,j,i) = fixup(peng, true);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b,pv,k,j,i) = fixup(pv, true);
            }
          } else {
            // No good neighbors
            v(b,prho,k,j,i) = fixup(prho, false);
            v(b,peng,k,j,i) = fixup(peng, false);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b,pv,k,j,i) = 0.;
            }
          }

          // Apply floors
          double rho_floor, sie_floor;
          bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i), rho_floor, sie_floor);
          v(b,prho,k,j,i) = v(b,prho,k,j,i) > rho_floor ? v(b,prho,k,j,i) : rho_floor;
          double u_floor = v(b,prho,k,j,i)*sie_floor;
          v(b,peng,k,j,i) = v(b,peng,k,j,i) > u_floor ? v(b,peng,k,j,i) : u_floor; 
          
          // Limit gammasq
          double gammacov[3][3];
          geom.Metric(CellLocation::Cent, k, j, i, gammacov);
          double vsq = 0.; 
          // TODO(BRR) what about pvel_hi
          SPACELOOP(m) {
            SPACELOOP(n) {
              vsq += gammacov[m][n] * v(b,pvel_lo+m,k,j,i) * v(b,pvel_lo+n,k,j,i);
            }
          }
          double gammamax = 50.;
          double vsqmax = 1. - 1./(gammamax*gammamax);
          if (vsq > vsqmax) {
            double vfac = sqrt(vsqmax/vsq); 
            v(b,pvel_lo,k,j,i) *= vfac;
            v(b,pvel_lo+1,k,j,i) *= vfac;
            v(b,pvel_lo+2,k,j,i) *= vfac;
          }
          
          // Set other primitives
          v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                              v(b,peng,k,j,i)/v(b,prho,k,j,i));
          v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
          v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                v(b,tmp,k,j,i))/v(b,prs,k,j,i); 

          PARTHENON_REQUIRE(!isnan(v(b,prho,k,j,i)), "A");
          PARTHENON_REQUIRE(!isnan(v(b,peng,k,j,i)), "A");
          PARTHENON_REQUIRE(!isnan(v(b,pvel_lo,k,j,i)), "A");
          PARTHENON_REQUIRE(!isnan(v(b,pvel_lo+1,k,j,i)), "A");
          PARTHENON_REQUIRE(!isnan(v(b,pvel_lo+2,k,j,i)), "A");
        }
      });

  return TaskStatus::complete;
  

  //parthenon::par_for(
  int nfixed_total = 0;
  while (nfixed_total < nfail_total) {
  int nfixed = 0;
  parthenon::par_reduce(
      //DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
        auto fixup = [&](const int iv, const Real inv_mask_sum) {
          v(b,iv,k,j,i)  = v(b,ifail,k,j,i-1)*v(b,iv,k,j,i-1)
                         + v(b,ifail,k,j,i+1)*v(b,iv,k,j,i+1);
          if (ndim > 1) {
            v(b,iv,k,j,i) += v(b,ifail,k,j-1,i)*v(b,iv,k,j-1,i)
                           + v(b,ifail,k,j+1,i)*v(b,iv,k,j+1,i);
            if (ndim == 3) {
              v(b,iv,k,j,i) += v(b,ifail,k-1,j,i)*v(b,iv,k-1,j,i)
                             + v(b,ifail,k+1,j,i)*v(b,iv,k+1,j,i);
            }
          }
          return inv_mask_sum*v(b,iv,k,j,i);
        };
        if (v(b,ifail,k,j,i) == con2prim_robust::FailFlags::fail) {
          Real num_valid = v(b,ifail,k,j,i-1) + v(b,ifail,k,j,i+1);
          if (ndim > 1) num_valid += v(b,ifail,k,j-1,i) + v(b,ifail,k,j+1,i);
          if (ndim == 3) num_valid += v(b,ifail,k-1,j,i)  + v(b,ifail,k+1,j,i);
          if (num_valid > 0.5) {
            const Real norm = 1.0/num_valid;
            v(b,prho,k,j,i) = fixup(prho, norm);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b,pv,k,j,i) = fixup(pv, norm);
            }
            double gammacov[3][3];
            //geom.Metric(X0, coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i), gamma);
            geom.Metric(CellLocation::Cent, k, j, i, gammacov);
            double vsq = 0.; 
            // TODO(BRR) what about pvel_hi
            SPACELOOP(m) {
              SPACELOOP(n) {
                vsq += gammacov[m][n] * v(b,pvel_lo+m,k,j,i) * v(b,pvel_lo+n,k,j,i);
              }
            }
            double gammamax = 50.;
            double vsqmax = 1. - 1./(gammamax*gammamax);
            if (vsq > vsqmax) {
              double vfac = sqrt(vsqmax/vsq); 
              v(b,pvel_lo,k,j,i) *= vfac;
              v(b,pvel_lo+1,k,j,i) *= vfac;
              v(b,pvel_lo+2,k,j,i) *= vfac;
            }

            //v(b,tmp,k,j,i) = fixup(tmp, norm);
            //v(b,peng,k,j,i) = v(b,prho,k,j,i)*eos.InternalEnergyFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            //TODO(BRR) apply floors to rho, ug here? Getting rho = 0, T = 0...
            v(b,peng,k,j,i) = fixup(peng, norm);
            if (pye > 0) v(b, pye,k,j,i) = fixup(pye, norm);
  
            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i), rho_floor, sie_floor);
            v(b,prho,k,j,i) = v(b,prho,k,j,i) > rho_floor ? v(b,prho,k,j,i) : rho_floor;
            double u_floor = v(b,prho,k,j,i)*sie_floor;
            v(b,peng,k,j,i) = v(b,peng,k,j,i) > u_floor ? v(b,peng,k,j,i) : u_floor; 
            
            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i);
            if (isnan(v(b,gm1,k,j,i))) {
              printf("gm1 is nan! %e %e %e %e\n", v(b,gm1,k,j,i), v(b,prho,k,j,i), v(b,tmp,k,j,i), v(b,prs,k,j,i));
              PARTHENON_FAIL("gm1 is nan!");
            }

            // TODO(jcd): make this work with MeshBlockPacks
            // TODOO(jcd): don't forget Ye!!!
            const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
            const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
            Real beta[3];
            geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
            Real gcov[4][4];
            geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
            Real gcon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
            Real S[3];
            const Real vel[] = {v(b, pvel_lo, k, j, i),
                                v(b, pvel_lo+1, k, j, i),
                                v(b, pvel_hi, k, j, i)};
            Real bcons[3];
            Real bp[3] = {0.0, 0.0, 0.0};
            if (pb_hi > 0) {
              bp[0] = v(b, pb_lo, k, j, i);
              bp[1] = v(b, pb_lo+1, k, j, i);
              bp[2] = v(b, pb_hi, k, j, i);
            }
            Real ye_cons;
            Real ye_prim = 0.0;
            if (pye > 0) {
              ye_prim = v(b, pye, k, j, i);
            }
            Real sig[3];
            prim2con::p2c(v(b,prho,k,j,i), vel, bp, v(b,peng,k,j,i), ye_prim, v(b,prs,k,j,i), v(b,gm1,k,j,i),
                gcov, gcon, beta, alpha, gdet,
                v(b,crho,k,j,i), S, bcons, v(b,ceng,k,j,i), ye_cons, sig);
            v(b, cmom_lo, k, j, i) = S[0];
            v(b, cmom_lo+1, k, j, i) = S[1];
            v(b, cmom_hi, k, j, i) = S[2];
            if (pye > 0) v(b, cye, k, j, i) = ye_cons;
            for (int m = slo; m <= shi; m++) {
              v(b,m,k,j,i) = sig[m-slo];
            }
            nf++;
            v(b,ifail,k,j,i) = con2prim_robust::FailFlags::success;
            if (isnan(v(b,crho,k,j,i))) {
              printf("p: %e %e %e %e %e\n", v(b,prho,k,j,i), v(b,peng,k,j,i),
                v(b,pvel_lo,k,j,i), v(b,pvel_lo+1,k,j,i), v(b,pvel_lo+2,k,j,i));
            }
            PARTHENON_REQUIRE(!isnan(v(b,crho,k,j,i)), "crho still nan!");
            PARTHENON_REQUIRE(!isnan(v(b,gm1,k,j,i)), "gm1 still nan!");
          } else {
            //printf("Found no valid neighbors\n");
          }
        }
      //});
      }, Kokkos::Sum<int>(nfixed));
  printf("nfixed: %i\n", nfixed);
  nfixed_total += nfixed;
  }*/

  //pmb->exec_space.fence();
  //exit(-1);


  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        PARTHENON_REQUIRE(!isnan(v(b,crho,k,j,i)), "crho still nan!");
      });
  return TaskStatus::complete;
}

TaskStatus FixFailures(MeshBlockData<Real> *rc) {

  namespace p = fluid_prim;
  namespace c = fluid_cons;
  auto *pmb = rc->GetParentPointer().get();
  return TaskStatus::complete;

  std::vector<std::string> vars({p::density, c::density, p::velocity,
                                 c::momentum, p::energy, c::energy});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;
  //  const Real Rh = std::log(2.0);

  auto gpkg = pmb->packages.Get("geometry");
  const Real a = gpkg->Param<Real>("a");
  const Real reh = 1. + sqrt(1. - a * a);
  const Real x1eh = std::log(reh); // TODO(BRR) still coordinate dependent

  parthenon::par_for(
    DEFAULT_LOOP_PATTERN, "Fix velocity inside horizon", DevExecSpace(),
    kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      if (coords.x1v(i) <= x1eh) {
        v(crho,k,j,i) *= 0.99;//std::min(v(cmom_lo,k,j,i), 0.0);
      }
    }
  );

  return TaskStatus::complete;
}

TaskStatus NothingEscapes(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fixup")->Param<bool>("enable_flux_fixup")) return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  //  const Real Rh = std::log(2.0);

  auto gpkg = pmb->packages.Get("geometry");
  const Real a = gpkg->Param<Real>("a");
  const Real reh = 1. + sqrt(1. - a * a);
  const Real x1eh = std::log(reh); // TODO(BRR) still coordinate dependent

  PackIndexMap imap;
  auto flux = rc->PackVariablesAndFluxes({fluid_cons::density, fluid_cons::energy, fluid_cons::momentum, fluid_prim::pressure},
                                         {fluid_cons::density, fluid_cons::energy, fluid_cons::momentum},
                                         imap);
  const int crho = imap[fluid_cons::density].first;
  
  int d = 1;
  //int crho = flux.crho;
  int NumConserved = 5;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.s,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(d,crho,k,j,i) = flux.flux(d,crho,k,j,i) > 0 ? 0. : flux.flux(d,crho,k,j,i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.e+1, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(d,crho,k,j,i) = flux.flux(d,crho,k,j,i) < 0 ? 0. : flux.flux(d,crho,k,j,i);
      });
  
  d = 2;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
      0, NumConserved-1, kb.s, kb.e, jb.s, jb.s, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        flux.flux(d,m,k,j,i) = 0.;
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
      0, NumConserved-1, kb.s, kb.e, jb.e+1, jb.e+1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        flux.flux(d,m,k,j,i) = 0.;//flux.v.flux(d,crho,k,j,i) < 0 ? 0. : flux.v.flux(d,crho,k,j,i);
      });
  /*
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FixFluxes", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,                                                      \
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (coords.x1f(i) <= x1eh+1.e-8) {
          for (int l = 0; l < 2; l++) {
            flux.flux(1,l,k,j,i) = std::min(flux.flux(1,l,k,j,i), 0.0);
          }
        }
      });*/

  return TaskStatus::complete;
}

template TaskStatus ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace fixup
