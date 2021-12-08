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

#include "fixup.hpp"

#include <bvals/bvals_interfaces.hpp>
#include <defs.hpp>
#include <singularity-eos/eos/eos.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"
#include "phoebus_utils/relativity_utils.hpp"

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
  bool report_c2p_fails = pin->GetOrAddBoolean("fixup", "report_c2p_fails", false);
  params.Add("report_c2p_fails", report_c2p_fails);
  bool enable_mhd_floors = pin->GetOrAddBoolean("fixup", "enable_mhd_floors", false);
  params.Add("enable_mhd_floors", enable_mhd_floors);

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
    } else if (floor_type == "ExpX1RhoU") {
      std::cout << "FOUND FLOOOR TYPE" << std::endl;
      Real rho0 = pin->GetOrAddReal("fixup", "rho0_floor", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "u0_floor", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp_floor", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "u_exp_floor", -3.0);
      params.Add("floor", Floors(exp_x1_rho_u_floor_tag, rho0, sie0, rp, sp));
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

  if (enable_mhd_floors) {
    Real bsqorho_max = pin->GetOrAddReal("fixup", "bsqorho_max", 50.);
    Real bsqou_max = pin->GetOrAddReal("fixup", "bsqou_max", 2500.);
    Real uorho_max = pin->GetOrAddReal("fixup", "uorho_max", 50.);
    params.Add("bsqorho_max", bsqorho_max);
    params.Add("bsqou_max", bsqou_max);
    params.Add("uorho_max", uorho_max);
  }

  params.Add("bounds", Bounds(params.Get<Floors>("floor"), params.Get<Ceilings>("ceiling")));

  return fix;
}

template <typename T>
TaskStatus ApplyFloors(T *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  
  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();

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
 
  bool enable_floors = fix_pkg->Param<bool>("enable_floors");
  if (!enable_floors) return TaskStatus::complete;
  bool enable_mhd_floors = fix_pkg->Param<bool>("enable_mhd_floors");

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;
  
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            
          double rho_floor, sie_floor;
          bounds.GetFloors(coords.x1v(k,j,i), coords.x2v(k,j,i), coords.x3v(k,j,i), rho_floor, sie_floor);
          
          bool floor_applied = false;
          if (v(b,prho,k,j,i) < rho_floor) {
            floor_applied = true;
            v(b,prho,k,j,i) = rho_floor;
          }
          if (v(b,peng,k,j,i)/v(b,prho,k,j,i) < sie_floor) {
            floor_applied = true;
            v(b,peng,k,j,i) = sie_floor*v(b,prho,k,j,i);
          }
          
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);

          if (enable_mhd_floors) {
            Real Bsq = 0.0;
            Real Bdotv = 0.0;
            const Real vp[3] = {v(b,pvel_lo,k,j,i), v(b,pvel_lo+1,k,j,i), v(b,pvel_lo+2,k,j,i)};
            const Real bp[3] = {v(b,pb_lo,k,j,i), v(pb_lo+1,k,j,i), v(pb_lo+2,k,j,i)};
            const Real W = phoebus::GetLorentzFactor(vp, gcov);
            const Real iW = 1.0/W;
            SPACELOOP2(ii, jj) {
              Bsq += gcov[ii+1][jj+1] * bp[ii] * bp[jj];
              Bdotv += gcov[ii+1][jj+1] * bp[ii] * vp[jj];
            }
            Real bcon0 = W * Bdotv / alpha;
            const Real bsq = (Bsq + alpha*alpha * bcon0*bcon0)*iW*iW;

            if (bsq/v(b,prho,k,j,i) > 50.) {
              floor_applied = true;
              v(b,prho,k,j,i) = bsq/50.;
            }
            if (bsq/v(b,peng,k,j,i) > 2500.) {
              floor_applied = true;
              v(b,peng,k,j,i) = bsq/2500.;
            }
            if (v(b,peng,k,j,i)/v(b,prho,k,j,i) > 50.) {
              floor_applied = true;
              v(b,peng,k,j,i) = 50.*v(b,prho,k,j,i);
            }
          }

          if (floor_applied) {
            // Update dependent primitives
            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i);

            // Update conserved variables
            const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
            Real gcon[3][3];
            geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
            Real beta[3];
            geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
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
          }
        });

  return TaskStatus::complete;
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
  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();

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

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup failures", DevExecSpace(),
        0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b,ifail,k,j,i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        }, Kokkos::Sum<int>(nfail_total));
    printf("total nfail: %i\n", nfail_total);
  }

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  if (!enable_c2p_fixup) return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;

  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto bounds = fix_pkg->Param<Bounds>("bounds");

  Coordinates_t coords = rc->GetParentPointer().get()->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(),
      0, v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
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
          printf("fail! %i %i %i\n", k, j, i);
          Real num_valid = v(b,ifail,k,j,i-1) + v(b,ifail,k,j,i+1);
          if (ndim > 1) num_valid += v(b,ifail,k,j-1,i) + v(b,ifail,k,j+1,i);
          if (ndim == 3) num_valid += v(b,ifail,k-1,j,i)  + v(b,ifail,k+1,j,i);
          if (num_valid > 0.5) {
            const Real norm = 1.0/num_valid;
            v(b,prho,k,j,i) = fixup(prho, norm);
            for (int pv = pvel_lo; pv <= pvel_hi; pv++) {
              v(b,pv,k,j,i) = fixup(pv, norm);
            }
            v(b,peng,k,j,i) = fixup(peng, norm);

            // Apply floors
            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k,j,i), coords.x2v(k,j,i), coords.x3v(k,j,i), rho_floor, sie_floor);
            v(b,prho,k,j,i) = v(b,prho,k,j,i) > rho_floor ? v(b,prho,k,j,i) : rho_floor;
            double u_floor = v(b,prho,k,j,i)*sie_floor;
            v(b,peng,k,j,i) = v(b,peng,k,j,i) > u_floor ? v(b,peng,k,j,i) : u_floor; 

            if (pye > 0) v(b, pye,k,j,i) = fixup(pye, norm);
            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i);
          } else {
            // No valid neighbors; set fluid mass/energy to floors and set primitive velocities to zero
            
            // Apply floors
            double rho_floor, sie_floor;
            bounds.GetFloors(coords.x1v(k,j,i), coords.x2v(k,j,i), coords.x3v(k,j,i), rho_floor, sie_floor);
            v(b,prho,k,j,i) = rho_floor;
            double u_floor = v(b,prho,k,j,i)*sie_floor;
            v(b,peng,k,j,i) = u_floor;

            // Safe value for ye
            if (pye > 0) {
              v(b, pye, k, j, i) = 0.5;
            }

            v(b,tmp,k,j,i) = eos.TemperatureFromDensityInternalEnergy(v(b,prho,k,j,i),
                                v(b,peng,k,j,i)/v(b,prho,k,j,i));
            v(b,prs,k,j,i) = eos.PressureFromDensityTemperature(v(b,prho,k,j,i),v(b,tmp,k,j,i));
            v(b,gm1,k,j,i) = eos.BulkModulusFromDensityTemperature(v(b,prho,k,j,i),
                                  v(b,tmp,k,j,i))/v(b,prs,k,j,i);

            // Zero primitive velocities
            SPACELOOP(ii) {
              v(b, pvel_lo+ii, k, j, i) = 0.;
            }
          }
          // Update conserved variables
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
        }
      });

  return TaskStatus::complete;
}

TaskStatus FixFluxes(MeshBlockData<Real> *rc) {
  using parthenon::BoundaryFace;
  using parthenon::BoundaryFlag;
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fixup")->Param<bool>("enable_flux_fixup")) return TaskStatus::complete;
  
  auto fluid = pmb->packages.Get("fluid");
  const std::string bc_ix1 = fluid->Param<std::string>("bc_ix1");
  const std::string bc_ox1 = fluid->Param<std::string>("bc_ox1");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int ndim = pmb->pmy_mesh->ndim;

  namespace p = fluid_prim;
  namespace c = fluid_cons;

  // x1-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
    if (bc_ix1 == "outflow") {
      auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                             std::vector<std::string>({fluid_cons::density}));
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(),
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X1DIR,0,k,j,i) = std::min(flux.flux(X1DIR,0,k,j,i), 0.0);
        });
    } else if (bc_ix1 == "reflect") {
      auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
                                             std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(),
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.s, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X1DIR,0,k,j,i) = 0.0;
          flux.flux(X1DIR,1,k,j,i) = 0.0;
        });
    }
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    if (bc_ox1 == "outflow") {
      auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                             std::vector<std::string>({fluid_cons::density}));
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(),
        kb.s, kb.e, jb.s, jb.e, ib.e+1, ib.e+1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X1DIR,0,k,j,i) = std::max(flux.flux(X1DIR,0,k,j,i), 0.0);
        });
    } else if (bc_ox1 == "reflect") {
      auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
                                             std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x1", DevExecSpace(),
        kb.s, kb.e, jb.s, jb.e, ib.e+1, ib.e+1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          flux.flux(X1DIR,0,k,j,i) = 0.0;
          flux.flux(X1DIR,1,k,j,i) = 0.0;
        });
    }
  }
  if (ndim == 1) return TaskStatus::complete;

  // x2-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::outflow) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                           std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.s, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X2DIR,0,k,j,i) = std::min(flux.flux(X2DIR,0,k,j,i), 0.0);
      });
  } else if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, 
                                             fluid_cons::energy, fluid_cons::momentum}),
                                           std::vector<std::string>({fluid_cons::density,
                                             fluid_cons::energy, fluid_cons::momentum}), imap);
    const int cmom_lo = imap[c::momentum].first;
    const int cmom_hi = imap[c::momentum].second;
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.s, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X2DIR,0,k,j,i) = 0.0;
        flux.flux(X2DIR,1,k,j,i) = 0.0;
        flux.flux(X2DIR,cmom_lo,k,j,i) = 0.0;
        flux.flux(X2DIR,cmom_lo+2,k,j,i) = 0.0;
      });
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::outflow) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                           std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(),
      kb.s, kb.e, jb.e+1, jb.e+1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X2DIR,0,k,j,i) = std::max(flux.flux(X2DIR,0,k,j,i), 0.0);
      });
  } else if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
    PackIndexMap imap;
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, 
                                             fluid_cons::energy, fluid_cons::momentum}),
                                           std::vector<std::string>({fluid_cons::density, 
                                             fluid_cons::energy, fluid_cons::momentum}), imap);
    const int cmom_lo = imap[c::momentum].first;
    const int cmom_hi = imap[c::momentum].second;
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x2", DevExecSpace(),
      kb.s, kb.e, jb.e+1, jb.e+1, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X2DIR,0,k,j,i) = 0.0;
        flux.flux(X2DIR,1,k,j,i) = 0.0;
        flux.flux(X2DIR,cmom_lo,k,j,i) = 0.0;
        flux.flux(X2DIR,cmom_lo+2,k,j,i) = 0.0;
      });
  }

  if (ndim == 2) return TaskStatus::complete;

  // x3-direction
  if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::outflow) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                           std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(),
      kb.s, kb.s, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X3DIR,0,k,j,i) = std::min(flux.flux(X3DIR,0,k,j,i), 0.0);
      });
  } else if (pmb->boundary_flag[BoundaryFace::inner_x3] == BoundaryFlag::reflect) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
                                           std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(),
      kb.s, kb.s, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X3DIR,0,k,j,i) = 0.0;
        flux.flux(X3DIR,1,k,j,i) = 0.0;
      });
  }
  if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::outflow) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density}),
                                           std::vector<std::string>({fluid_cons::density}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(),
      kb.e+1, kb.e+1, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X3DIR,0,k,j,i) = std::max(flux.flux(X3DIR,0,k,j,i), 0.0);
      });
  } else if (pmb->boundary_flag[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    auto flux = rc->PackVariablesAndFluxes(std::vector<std::string>({fluid_cons::density, fluid_cons::energy}),
                                           std::vector<std::string>({fluid_cons::density, fluid_cons::energy}));
    parthenon::par_for(DEFAULT_LOOP_PATTERN, "FixFluxes::x3", DevExecSpace(),
      kb.e+1, kb.e+1, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        flux.flux(X3DIR,0,k,j,i) = 0.0;
        flux.flux(X3DIR,1,k,j,i) = 0.0;
      });
  }

  return TaskStatus::complete;
}

template TaskStatus ApplyFloors<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

template TaskStatus ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);

} // namespace fixup
