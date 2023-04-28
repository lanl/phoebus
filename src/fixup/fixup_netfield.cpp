// © 2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

#include <cmath>

#include "fixup.hpp"

#include <bvals/bvals_interfaces.hpp>
#include <defs.hpp>

#include "analysis/history.hpp"
#include "fluid/con2prim_robust.hpp"
#include "fluid/fluid.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/reduction.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

using Microphysics::EOS::EOS;
using robust::ratio;

namespace fixup {

TaskStatus SumMdotPhiForNetFieldScaling(MeshData<Real> *md, const Real t, const int stage,
                                        std::vector<Real> *sums) {
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (stage == 1) {
    if (enable_phi_enforcement) {
      const Real enforced_phi_start_time =
          fix_pkg->Param<Real>("enforced_phi_start_time");
      const Real next_dphi_dt_update_time =
          fix_pkg->Param<Real>("next_dphi_dt_update_time");
      if (t >= next_dphi_dt_update_time && t >= enforced_phi_start_time) {
        const Real Mdot = History::ReduceMassAccretionRate(md);
        const Real Phi = History::ReduceMagneticFluxPhi(md);
        printf("Per-MD sum: Mdot: %e Phi: %e\n", Mdot, Phi);

        (*sums)[0] += Mdot;
        (*sums)[1] += Phi;
      }
    }
  }
  printf("%s:%i\n", __FILE__, __LINE__);

  return TaskStatus::complete;
}

TaskStatus NetFieldStartReduce(MeshData<Real> *md, const Real t, const int stage,
                               AllReduce<std::vector<Real>> *net_field_totals) {
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {
    const Real enforced_phi_start_time = fix_pkg->Param<Real>("enforced_phi_start_time");
    const Real next_dphi_dt_update_time =
        fix_pkg->Param<Real>("next_dphi_dt_update_time");
    if (t >= next_dphi_dt_update_time && t >= enforced_phi_start_time) {
      TaskStatus status = net_field_totals->StartReduce(MPI_SUM);
      return status;
    }
  }

  return TaskStatus::complete;
}

TaskStatus NetFieldCheckReduce(MeshData<Real> *md, const Real t, const int stage,
                               AllReduce<std::vector<Real>> *net_field_totals) {
  if (stage != 1) {
    return TaskStatus::complete;
  }

  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {
    const Real enforced_phi_start_time = fix_pkg->Param<Real>("enforced_phi_start_time");
    const Real next_dphi_dt_update_time =
        fix_pkg->Param<Real>("next_dphi_dt_update_time");
    if (t >= next_dphi_dt_update_time && t >= enforced_phi_start_time) {
      TaskStatus status = net_field_totals->CheckReduce();
      return status;
    }
  }

  return TaskStatus::complete;
}

TaskStatus UpdateNetFieldScaleControls(MeshData<Real> *md, const Real t, const Real dt,
                                       const int stage, std::vector<Real> *vals0,
                                       std::vector<Real> *vals1) {
  if (stage != 1) {
    return TaskStatus::complete;
  }
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {
    const Real next_dphi_dt_update_time =
        fix_pkg->Param<Real>("next_dphi_dt_update_time");
    if (t >= next_dphi_dt_update_time) {
      // Real phi_factor = fix_pkg->Param<Real>("phi_factor");
      Real dphi_dt = fix_pkg->Param<Real>("dphi_dt");
      const Real enforced_phi = fix_pkg->Param<Real>("enforced_phi");
      const Real enforced_phi_timescale = fix_pkg->Param<Real>("enforced_phi_timescale");

      Real phi_factor = (*vals1)[1] - (*vals0)[1];
      // Limit change of dphi_dt
      if (dphi_dt > 0.) {
        dphi_dt = std::clamp((enforced_phi - (*vals0)[1]) / enforced_phi_timescale,
                             3. / 4. * dphi_dt, 4. / 3. * dphi_dt);
      } else {
        dphi_dt = (enforced_phi - (*vals0)[1]) / enforced_phi_timescale;
      }
      printf("dphi_dt: %e phi_factor: %e\n", dphi_dt, phi_factor);

      fix_pkg->UpdateParam("phi_factor", phi_factor);
      fix_pkg->UpdateParam("dphi_dt", dphi_dt);

      // Update timing values
      const Real enforced_phi_cadence = fix_pkg->Param<Real>("enforced_phi_cadence");
      fix_pkg->UpdateParam("next_dphi_dt_update_time",
                           next_dphi_dt_update_time + enforced_phi_cadence);
      printf("next_dphi_dt_update_time = %e\n", next_dphi_dt_update_time);
    }
  }
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("vals0: %e %e\n", (*vals0)[0], (*vals0)[1]);
  printf("vals1: %e %e\n", (*vals1)[0], (*vals1)[1]);
  return TaskStatus::complete;
}

TaskStatus ModifyNetField(MeshData<Real> *md, const Real t, const Real dt,
                          const int stage, const bool fiducial,
                          const Real fiducial_factor) {
  if (stage != 1 && fiducial) {
    return TaskStatus::complete;
  }
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {

    const Real next_dphi_dt_update_time =
        fix_pkg->Param<Real>("next_dphi_dt_update_time");

    if (fiducial && !(t >= next_dphi_dt_update_time)) {
      return TaskStatus::complete;
    }

    namespace p = fluid_prim;
    namespace c = fluid_cons;
    const std::vector<std::string> vars({p::bfield, c::bfield});
    PackIndexMap imap;
    auto pack = md->PackVariables(vars, imap);

    const int pblo = imap[p::bfield].first;
    const int cblo = imap[c::bfield].first;

    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const auto jb = md->GetBoundsJ(IndexDomain::interior);
    const auto kb = md->GetBoundsK(IndexDomain::interior);

    auto geom = Geometry::GetCoordinateSystem(md);
    auto gpkg = pm->packages.Get("geometry");
    bool derefine_poles = gpkg->Param<bool>("derefine_poles");
    Real h = gpkg->Param<Real>("h");
    Real xt = gpkg->Param<Real>("xt");
    Real alpha = gpkg->Param<Real>("alpha");
    Real x0 = gpkg->Param<Real>("x0");
    Real smooth = gpkg->Param<Real>("smooth");
    const Real Rh = gpkg->Param<Real>("Rh");
    auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

    Real phi_factor =
        fix_pkg->Param<Real>("phi_factor") * fix_pkg->Param<Real>("dphi_dt") * dt;
    if (fiducial) {
      phi_factor = fiducial_factor;
    }
    printf("phi_factor: %e\n", phi_factor);

    // Calculate hyperbola-based magnetic field configuration inside the event horizon
    ParArrayND<Real> A("vector potential", pack.GetDim(5), jb.e + 2, ib.e + 2);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Phoebus::Fixup::EndOfStepModify::EvaluateQ",
        DevExecSpace(), 0, pack.GetDim(5) - 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int j, const int i) {
          const auto &coords = pack.GetCoords(b);
          const Real x1 = coords.Xf<1>(0, j, i);
          const Real x2 = coords.Xf<2>(0, j, i);
          const Real r = tr.bl_radius(x1);
          const Real th = tr.bl_theta(x1, x2);

          const Real x = r * std::sin(th);
          const Real z = r * std::cos(th);
          const Real a_hyp = Rh;
          const Real b_hyp = 3. * a_hyp;
          const Real x_hyp = a_hyp * std::sqrt(1. + std::pow(z / b_hyp, 2.));
          const Real q = (std::pow(x, 2) - std::pow(x_hyp, 2)) / std::pow(x_hyp, 2.);
          if (x < x_hyp) {
            A(b, j, i) = q;
          }
        });

    // Modify B field
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Phoebus::Fixup::EndOfStepModify::EvaluateBField",
        DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoords(b);
          const Real gammadet = geom.DetGamma(CellLocation::Cent, b, k, j, i);

          pack(b, pblo, k, j, i) +=
              phi_factor *
              (-(A(b, j, i) - A(b, j + 1, i) + A(b, j, i + 1) - A(b, j + 1, i + 1)) /
               (2.0 * coords.CellWidthFA(X2DIR, k, j, i) * gammadet));
          pack(b, pblo + 1, k, j, i) +=
              phi_factor *
              ((A(b, j, i) + A(b, j + 1, i) - A(b, j, i + 1) - A(b, j + 1, i + 1)) /
               (2.0 * coords.CellWidthFA(X1DIR, k, j, i) * gammadet));

          SPACELOOP(ii) {
            pack(b, cblo + ii, k, j, i) = pack(b, pblo + ii, k, j, i) * gammadet;
          }
        });

    if (!fiducial) {
      // P2C
    }
  } // enable_phi_enforcement

  return TaskStatus::complete;
}

} // namespace fixup