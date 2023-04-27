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

TaskStatus SumMdotPhiForNetFieldScaling(MeshData<Real> *md, const Real t,
                                        std::vector<Real> *sums) {
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {
    const Real enforced_phi_start_time = fix_pkg->Param<Real>("enforced_phi_start_time");
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
  printf("%s:%i\n", __FILE__, __LINE__);

  return TaskStatus::complete;
}

TaskStatus UpdateNetFieldScaleControls(MeshData<Real> *md, const Real t, const Real dt,
                                       const Real Mdot, const Real Phi) {
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pm = md->GetParentPointer();
  printf("%s:%i\n", __FILE__, __LINE__);
  return TaskStatus::complete;
}

TaskStatus ModifyNetField(MeshData<Real> *md, const Real t, const Real dt,
                          std::vector<Real> *vals, const bool fiducial,
                          const Real fiducial_factor) {
  const Real Mdot = (*vals)[0];
  const Real Phi = (*vals)[1];
  const Real phi = Phi / std::sqrt(Mdot);
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("Mdot: %e Phi: %e\n", Mdot, Phi);
  auto *pm = md->GetParentPointer();
  StateDescriptor *fix_pkg = pm->packages.Get("fixup").get();

  const bool enable_phi_enforcement = fix_pkg->Param<bool>("enable_phi_enforcement");

  if (enable_phi_enforcement) {

    const Real next_dphi_dt_update_time =
        fix_pkg->Param<Real>("next_dphi_dt_update_time");

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

    Real phi_factor = fix_pkg->Param<Real>("phi_factor");
    if (fiducial) {
      phi_factor = fiducial_factor;
    }

    if (t >= next_dphi_dt_update_time) {
      //  next_dphi_dt_update_time += enforced_phi_cadence;

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
                phi_factor * dt *
                (-(A(b, j, i) - A(b, j + 1, i) + A(b, j, i + 1) - A(b, j + 1, i + 1)) /
                 (2.0 * coords.CellWidthFA(X2DIR, k, j, i) * gammadet));
            pack(b, pblo + 1, k, j, i) +=
                phi_factor * dt *
                ((A(b, j, i) + A(b, j + 1, i) - A(b, j, i + 1) - A(b, j + 1, i + 1)) /
                 (2.0 * coords.CellWidthFA(X1DIR, k, j, i) * gammadet));
          });

      // TODO(BRR) optionally call p2c here?
    } // t >= next_dphi_dt_update_time
  }   // enable_phie_enforcement

  exit(-1);
  return TaskStatus::complete;
}

} // namespace fixup