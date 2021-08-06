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

#include <cmath>
#include <memory>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

// Phoebus
#include "geometry/geometry_utils.hpp"

#include "gr1d.hpp"

using namespace parthenon::package::prelude;

namespace GR1D {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto gr1d = std::make_shared<StateDescriptor>("GR1D");
  Params &params = gr1d->AllParams();

  bool enable_gr1d = pin->GetOrAddBoolean("GR1D", "enabled", false);
  params.Add("enable_gr1d", enable_gr1d);
  if (!enable_gr1d) return gr1d; // Short-circuit with nothing

  // TODO(JMM): Ghost zones or one-sided differences/BCs?
  int npoints = pin->GetOrAddInteger("GR1D", "npoints", 100);
  params.Add("npoints", npoints);

  // Number of iterations per cycle, before checking error
  int niters_check = pin->GetOrAddInteger("GR1D", "niters_check", 10);
  params.Add("niters_check", niters_check);

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("GR1D", "rout", 100);
  params.Add("rin", rin);
  params.Add("rout", rout);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Grids grids;
  grids.a = Grids::Grid_t("GR1D a", npoints);
  grids.dadr = Grids::Grid_t("GR1D da/dr", npoints);
  grids.K_rr = Grids::Grid_t("GR1D K^r_r", npoints);
  grids.dKdr = Grids::Grid_t("GR1D dK/dr", npoints);
  grids.alpha = Grids::Grid_t("GR1D alpha", npoints);
  grids.dalphadr = Grids::Grid_t("GR1D dalpha/dr", npoints);
  grids.aleph = Grids::Grid_t("GR1D aleph", npoints);
  grids.dalephdr = Grids::Grid_t("GR1D daleph/dr", npoints);
  grids.rho = Grids::Grid_t("GR1D rho", npoints);
  grids.j_r = Grids::Grid_t("GR1D j^r", npoints);
  grids.trcS = Grids::Grid_t("GR1D S", npoints);

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        grids.a(i) = 1;
        grids.dadr(i) = 0;
        grids.K_rr(i) = 0;
        grids.dKdr(i) = 0;
        grids.alpha(i) = 1;
        grids.dalphadr(i) = 0;
        grids.aleph(i) = 0;
        grids.dalephdr(i) = 0;
        grids.rho(i) = 0;
        grids.j_r(i) = 0;
        grids.trcS(i) = 0;
      });

  // TODO(JMM): Initialize all grids to zero
  params.Add("grids", grids);

  // The radius object, returns radius vs index and index vs radius
  Radius radius(rin, rout, npoints);
  params.Add("radius", radius);

  return gr1d;
}

TaskStatus IterativeSolve(StateDescriptor *pkg) {
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  auto npoints = params.Get<int>("npoints");
  auto niters_check = params.Get<int>("niters_check");
  auto rin = params.Get<Real>("rin");
  auto rout = params.Get<Real>("rout");
  auto radius = params.Get<GR1D::Radius>("radius");
  auto grids = params.Get<GR1D::Grids>("grids");

  auto a = grids.a;
  auto dadr = grids.dadr;
  auto K = grids.K_rr;
  auto dKdr = grids.dKdr;
  auto alpha = grids.alpha;
  auto dalphadr = grids.dalphadr;
  auto aleph = grids.aleph;
  auto dalephdr = grids.dalephdr;
  auto rho = grids.rho;
  auto j = grids.j_r;
  auto S = grids.trcS;
  const Real dr = radius.dx();

  // Do we actually need scratch?
  int scratch_level = 1;
  size_t scratch_size_in_bytes = parthenon::ScratchPad1D<Real>::shmem_size(1);

  // little trick to do sequential iterations on the GPU without launch overhead
  auto lp_outer = parthenon::outer_loop_pattern_teams_tag;
  parthenon::par_for_outer(
      lp_outer, "GR1D iterative solve for metric", parthenon::DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 1, 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int team) {
        for (int iter = 0; iter < niters_check; ++iter) {
          // BCs on vars
          aleph(0) = 0;
	  //alpha(npoints - 1) = 1;
          aleph(npoints - 1) =
	    Geometry::Utils::ratio(1 - a(npoints - 1), radius.max() * a(npoints - 1));
          K(npoints - 1) = 0;
	  //a(npoints - 1) = 1;
          member.team_barrier();

          // FD
          SummationByPartsFD4(member, a, dadr, npoints, dr);
          SummationByPartsFD4(member, K, dKdr, npoints, dr);
          SummationByPartsFD4(member, alpha, dalphadr, npoints, dr);
          SummationByPartsFD4(member, aleph, dalephdr, npoints, dr);
          member.team_barrier();

          // BCS on derivatives
          dadr(0) = 0;
          dKdr(0) = 0;
          dadr(npoints - 1) =
	    a(npoints - 1) * (1 - a(npoints - 1) * a(npoints - 1)) / (2 * radius.max());
          member.team_barrier();

          // Iterations
          par_for_inner(member, 0, npoints - 1, [&](const int i) {
            Real r = radius.x(i);

            Real al = a(i);
            Real Kl = K(i);
            Real alphal = alpha(i);
            Real alephl = aleph(i);
            a(i) +=
                (al / 2) * (r * r * (16 * M_PI * rho(i) - (3 / 2) * Kl * Kl) - al + 1) -
                dadr(i);

            K(i) += r * Kl * M_PI * al * al * j(i) - 3 * Kl - r * dKdr(i);

            alpha(i) += alephl - dalphadr(i);

            aleph(i) +=
                al * al * al * alphal * ((3 / 2) * Kl * Kl + 4 * M_PI * (rho(i) + S(i))) -
                dadr(i) * alephl - dalephdr(i) * al;
          });
          member.team_barrier();
        }
      });

  return TaskStatus::complete;
}

} // namespace GR1D
