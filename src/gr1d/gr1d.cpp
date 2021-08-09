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
#include <cstdio>
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
  int niters_check = pin->GetOrAddInteger("GR1D", "niters_check", 10000);
  PARTHENON_REQUIRE_THROWS(niters_check > 0, "niters must be strictly positive");
  params.Add("niters_check", niters_check);

  // Error tolerance
  Real error_tolerance = pin->GetOrAddReal("GR1D", "error_tolerance", 1e-4);
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
  Real jacobi_step_size = pin->GetOrAddReal("GR1D", "jacobi_step_size", 1e-6);
  params.Add("jacobi_step_size", jacobi_step_size);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Grids grids;
  grids.a = Grids::Grid_t("GR1D a", npoints);
  grids.lna = Grids::Grid_t("GR1D ln(a)", npoints);
  grids.dlnadr = Grids::Grid_t("GR1D dln(a)/dr", npoints);
  grids.delta_a = Grids::Grid_t("GR1D delta a", npoints);
  grids.K_rr = Grids::Grid_t("GR1D K^r_r", npoints);
  grids.dKdr = Grids::Grid_t("GR1D dK/dr", npoints);
  grids.delta_K = Grids::Grid_t("GR1D delta K", npoints);
  grids.alpha = Grids::Grid_t("GR1D alpha", npoints);
  grids.dalphadr = Grids::Grid_t("GR1D dalpha/dr", npoints);
  grids.delta_alpha = Grids::Grid_t("GR1D delta alpha", npoints);
  grids.rho = Grids::Grid_t("GR1D rho", npoints);
  grids.j_r = Grids::Grid_t("GR1D j^r", npoints);
  grids.trcS = Grids::Grid_t("GR1D S", npoints);

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        grids.a(i) = 1;
        grids.lna(i) = 0;
        grids.dlnadr(i) = 0;
        grids.K_rr(i) = 0;
        grids.dKdr(i) = 0;
        grids.alpha(i) = 1;
        grids.dalphadr(i) = 0;
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

  auto error_tolerance = params.Get<Real>("error_tolerance");

  auto a = grids.a;
  auto lna = grids.lna;
  auto dlnadr = grids.dlnadr;
  auto delta_a = grids.delta_a;
  auto K = grids.K_rr;
  auto dKdr = grids.dKdr;
  auto delta_K = grids.delta_K;
  auto alpha = grids.alpha;
  auto dalphadr = grids.dalphadr;
  auto delta_alpha = grids.delta_alpha;
  auto rho = grids.rho;
  auto j = grids.j_r;
  auto S = grids.trcS;
  const Real dr = radius.dx();

  Real tau = jacobi_step_size*dr*dr;

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
          // logs, temps
          par_for_inner(member, 0, npoints - 1,
                        [&](const int i) { lna(i) = std::log(a(i)); });
          member.team_barrier();

          // FD
          SummationByPartsFD4(member, lna, dlnadr, npoints, dr);
          SummationByPartsFD4(member, K, dKdr, npoints, dr);
          // SummationByPartsFD4(member, alpha, dalphadr, npoints, dr);
          member.team_barrier();

          // Deltas
          par_for_inner(member, 0, npoints - 1, [&](const int i) {
            Real r = radius.x(i);

            delta_a(i) = r * r * (16 * M_PI * rho(i) - (3. / 2.) * K(i) * K(i)) -
                         2 * r * dlnadr(i) + 1;
            delta_a(i) -= a(i);

            delta_K(i) = (r / 3.) * (8 * M_PI * a(i) * a(i) * j(i) - dKdr(i));
            delta_K(i) -= K(i);

            // delta_alpha(i) = aleph(i) - dalphadr(i);
            // delta_aleph(i) = a(i) * a(i) * alpha(i) *
            //                      ((3 / 2) * K(i) * K(i) + 4 * M_PI * (rho(i) + S(i))) -
            //   dadr(i) * aleph(i)/ (std::abs(a(i)) + 1e-20) - dalephdr(i);
          });
          member.team_barrier();

          // Boundary conditions
          // r = 0
          delta_a(0) = a(1) - a(0);
          delta_K(0) = K(1) - K(0);
          // r outer
          Real aout = a(npoints - 1);
          Real aout3 = aout * aout * aout;
          Real rmax = radius.max();
          delta_a(npoints - 1) = dr * (aout - aout3) / (2. * rmax) + a(npoints - 2);
          delta_a(npoints - 1) -= a(npoints - 1);
          delta_K(npoints - 1) = -K(npoints - 1); // K -> 0
          member.team_barrier();

          // Update
          //printf("%d\n", iter);
          par_for_inner(member, 0, npoints - 1, [&](const int i) {
            // printf("\t%d: %14e %14e %14e %14e %14e %14e %14e\n", i, a(i), K(i), lna(i),
            //        dlnadr(i), dKdr(i), delta_a(i), delta_K(i));
            a(i) += tau*delta_a(i);
            K(i) += tau*delta_K(i);
	    // ensure a is strictly positive
	    a(i) = std::max(error_tolerance, a(i));
            // alpha(i) -= tau * delta_alpha(i);
          });
          member.team_barrier();
        }
      });

  return TaskStatus::complete;
}

bool Converged(StateDescriptor *pkg) {
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return true;

  auto npoints = params.Get<int>("npoints");
  auto error_tolerance = params.Get<Real>("error_tolerance");
  auto grids = params.Get<GR1D::Grids>("grids");

  Real max_err = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "GR1D: CheckConvergence",
      parthenon::DevExecSpace(), 0, npoints - 1,
      KOKKOS_LAMBDA(const int i, Real &eps) {
        eps = std::max(eps, GetError(grids.a, grids.delta_a, i));
        eps = std::max(eps, GetError(grids.K_rr, grids.delta_K, i));
        //printf("\terr = %14g\n", eps);
      },
      Kokkos::Max<Real>(max_err));
  max_err = std::abs(max_err);
  printf("max error = %.14e\n", max_err);
  return max_err < error_tolerance;
}

void DumpToTxt(const std::string &filename, StateDescriptor *pkg) {
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<GR1D::Radius>("radius");
  auto grids = params.Get<GR1D::Grids>("grids");

  auto a = grids.a;
  auto K = grids.K_rr;
  auto alpha = grids.alpha;
  auto rho = grids.rho;
  auto j = grids.j_r;
  auto S = grids.trcS;

  auto a_h = Kokkos::create_mirror_view(a);
  auto K_h = Kokkos::create_mirror_view(K);
  auto alpha_h = Kokkos::create_mirror_view(alpha);
  auto rho_h = Kokkos::create_mirror_view(rho);
  auto j_h = Kokkos::create_mirror_view(j);
  auto S_h = Kokkos::create_mirror_view(S);

  Kokkos::deep_copy(a_h, a);
  Kokkos::deep_copy(K_h, K);
  Kokkos::deep_copy(alpha_h, alpha);
  Kokkos::deep_copy(rho_h, rho);
  Kokkos::deep_copy(j_h, j);
  Kokkos::deep_copy(S_h, S);

  FILE *pf;
  pf = fopen(filename.c_str(), "w");
  fprintf(pf, "#r\ta\tK\talpha\trho\tj\tS\n");
  for (int i = 0; i < npoints; ++i) {
    Real r = radius.x(i);
    fprintf(pf, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e\n", r, a_h(i), K_h(i), alpha_h(i),
            rho_h(i), j_h(i), S_h(i));
  }
  fclose(pf);
}

} // namespace GR1D
