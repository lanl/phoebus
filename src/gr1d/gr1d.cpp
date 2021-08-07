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

// stdlib
#include <cmath>
#include <memory>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

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
  PARTHENON_REQUIRE_THROWS(npoints > 0, "npoints must be strictly positive");
  params.Add("npoints", npoints);

  // Number of iterations per cycle, before checking error
  int niters_check = pin->GetOrAddInteger("GR1D", "niters_check", 10);
  PARTHENON_REQUIRE_THROWS(niters_check > 0, "niters must be strictly positive");
  params.Add("niters_check", niters_check);

  // Error tolerance
  Real error_tolerance = pin->GetOrAddReal("GR1D", "error_tolerance", 1e-8);
  PARTHENON_REQUIRE_THROWS(error_tolerance > 0,
                           "Error tolerance must be strictly positive");
  params.Add("error_tolerance", error_tolerance);

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("GR1D", "rout", 100);
  params.Add("rin", rin);
  params.Add("rout", rout);

  // Jacobi Step size
  Real jacobi_step_size = pin->GetOrAddReal("GR1D", "jacobi_step_size", 1e-1);
  params.Add("jacobi_step_size", jacobi_step_size);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Grids grids;
  grids.a = Grids::Grid_t("GR1D a", npoints);
  grids.dadr = Grids::Grid_t("GR1D da/dr", npoints);
  grids.delta_a = Grids::Grid_t("GR1D delta a", npoints);
  grids.K_rr = Grids::Grid_t("GR1D K^r_r", npoints);
  grids.dKdr = Grids::Grid_t("GR1D dK/dr", npoints);
  grids.delta_K = Grids::Grid_t("GR1D delta K", npoints);
  grids.alpha = Grids::Grid_t("GR1D alpha", npoints);
  grids.dalphadr = Grids::Grid_t("GR1D dalpha/dr", npoints);
  grids.delta_alpha = Grids::Grid_t("GR1D delta alpha", npoints);
  grids.aleph = Grids::Grid_t("GR1D aleph", npoints);
  grids.dalephdr = Grids::Grid_t("GR1D daleph/dr", npoints);
  grids.delta_aleph = Grids::Grid_t("GR1D delta aleph", npoints);
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
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto niters_check = params.Get<int>("niters_check");
  auto radius = params.Get<GR1D::Radius>("radius");
  auto jacobi_step_size = params.Get<Real>("jacobi_step_size");
  auto grids = params.Get<GR1D::Grids>("grids");

  auto a = grids.a;
  auto dadr = grids.dadr;
  auto delta_a = grids.delta_a;
  auto K = grids.K_rr;
  auto dKdr = grids.dKdr;
  auto delta_K = grids.delta_K;
  auto alpha = grids.alpha;
  auto dalphadr = grids.dalphadr;
  auto delta_alpha = grids.delta_alpha;
  auto aleph = grids.aleph;
  auto delta_aleph = grids.delta_aleph;
  auto dalephdr = grids.dalephdr;
  auto rho = grids.rho;
  auto j = grids.j_r;
  auto S = grids.trcS;
  const Real dr = radius.dx();

  Real tau = std::min(jacobi_step_size*dr*dr, 1./(npoints*npoints));

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
          // FD
          SummationByPartsFD4(member, a, dadr, npoints, dr);
          SummationByPartsFD4(member, K, dKdr, npoints, dr);
          //SummationByPartsFD4(member, alpha, dalphadr, npoints, dr);
          //SummationByPartsFD4(member, aleph, dalephdr, npoints, dr);
          member.team_barrier();

          // Deltas
          par_for_inner(member, 0, npoints - 1, [&](const int i) {
            Real r = radius.x(i);

            delta_a(i) =
	      (a(i) / (2 * r)) *
                    (r * r * (16 * M_PI * rho(i) - (3 / 2) * K(i) * K(i)) - a(i) + 1) -
                dadr(i);

            delta_K(i) = K(i) * M_PI * a(i) * a(i) * j(i) - (3 / r) * K(i) - dKdr(i);

            delta_alpha(i) = aleph(i) - dalphadr(i);
            delta_aleph(i) = a(i) * a(i) * alpha(i) *
                                 ((3 / 2) * K(i) * K(i) + 4 * M_PI * (rho(i) + S(i))) -
	      dadr(i) * aleph(i)/a(i) - dalephdr(i);
          });
          member.team_barrier();
	  
	  // Boundary conditions
	  // r = 0
	  delta_a(0) = -dadr(0);
	  delta_K(0) = -dKdr(0);
	  delta_aleph(0) = -aleph(0);
	  // r outer
	  Real aout = a(npoints - 1);
	  Real rmax = radius.max();
	  delta_a(npoints - 1) = (aout - aout*aout*aout)/(2* rmax) - dadr(npoints - 1);
	  delta_K(npoints - 1) = -K(npoints - 1);
	  delta_aleph(npoints - 1) = (1 - aout)/(aout*rmax) - aleph(npoints-1);
	  member.team_barrier();

          // Update
          par_for_inner(member, 0, npoints - 1, [&](const int i) {
            a(i) -= tau * delta_a(i);
            K(i) -= tau * delta_K(i);
            alpha(i) -= tau * delta_alpha(i);
            aleph(i) -= tau * delta_aleph(i);
          });
	  member.team_barrier();
        }
      });

  return TaskStatus::complete;
}

bool Converged(StateDescriptor *pkg) {
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  auto npoints = params.Get<int>("npoints");
  auto error_tolerance = params.Get<Real>("error_tolerance");
  auto grids = params.Get<GR1D::Grids>("grids");

  Real max_err = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "GR1D: CheckConvergence",
      parthenon::DevExecSpace(), 2, npoints - 2,
      KOKKOS_LAMBDA(const int i, Real &eps) {
        eps = std::max(eps, GetError(grids.a, grids.delta_a, i));
        eps = std::max(eps, GetError(grids.K_rr, grids.delta_K, i));
        eps = std::max(eps, GetError(grids.alpha, grids.delta_alpha, i));
        eps = std::max(eps, GetError(grids.aleph, grids.delta_aleph, i));
        //printf("\terr = %14g\n", eps);
      },
      Kokkos::Max<Real>(max_err));
  max_err = std::abs(max_err);
  //printf("max error = %.14e\n", max_err);
  return max_err < error_tolerance;
}

} // namespace GR1D
