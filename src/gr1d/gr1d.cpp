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

namespace ShootingMethod {
KOKKOS_INLINE_FUNCTION
Real get_arhs(Real a, Real K, Real r, Real rho) {
  return (r <= 0)
             ? 0
             : (a / (2. * r)) * (r * r * (16 * M_PI * rho - (3. / 2.) * K * K) - a + 1);
}
KOKKOS_INLINE_FUNCTION
Real get_Krhs(Real a, Real K, Real r, Real j) {
  return (r <= 0) ? 0 : 8 * M_PI * a * a * j - (3. / r) * K;
}
KOKKOS_INLINE_FUNCTION
void hypersurface_rhs(Real r, const Real in[NHYPER], const Real matter[NMAT],
                      Real out[NHYPER]) {
  Real a = in[Hypersurface::A];
  Real K = in[Hypersurface::K];
  Real rho = matter[Matter::RHO];
  Real j = matter[Matter::J_R];
  out[Hypersurface::A] = get_arhs(a, K, r, rho);
  out[Hypersurface::K] = get_Krhs(a, K, r, j);
}

template <typename H, typename M>
KOKKOS_INLINE_FUNCTION void get_residual(const H &h, const M &m, Real r, int npoints,
                                         Real R[NHYPER]) {
  Real a = h(Hypersurface::A, npoints - 1);
  Real K = h(Hypersurface::K, npoints - 1);
  Real rho = m(Matter::RHO, npoints - 1);
  Real dadr = get_arhs(a, K, r, rho);
  R[Hypersurface::A] = (a - a * a * a) / (2 * r) - dadr;
  R[Hypersurface::K] = K;
}
} // namespace ShootingMethod

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
  int niters_check = pin->GetOrAddInteger("GR1D", "niters_check", 1000);
  PARTHENON_REQUIRE_THROWS(niters_check > 0, "niters must be strictly positive");
  params.Add("niters_check", niters_check);

  // Error tolerance
  Real error_tolerance = pin->GetOrAddReal("GR1D", "error_tolerance", 1e-6);
  PARTHENON_REQUIRE_THROWS(error_tolerance > 0,
                           "Error tolerance must be strictly positive");
  params.Add("error_tolerance", error_tolerance);

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("GR1D", "rout", 100);
  params.Add("rin", rin);
  params.Add("rout", rout);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Matter_t matter("GR1D matter grid", NMAT, npoints);
  Matter_host_t matter_h = Kokkos::create_mirror_view(matter);
  Hypersurface_t hypersurface("GR1D hypersurface grid", NHYPER, npoints);
  Hypersurface_host_t hypersurface_h = Kokkos::create_mirror_view(hypersurface);
  Alpha_t alpha("GR1D lapse grid", npoints);
  Alpha_t dalpha("GR1D delta lapse grid", npoints);

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        for (int v = 0; v < NMAT; ++v) {
          matter(v, i) = 0;
        }
        hypersurface(Hypersurface::A, i) = 1;
        hypersurface(Hypersurface::K, i) = 0;
        alpha(i) = 100;
        dalpha(i) = 0;
      });
  Kokkos::deep_copy(matter_h, matter);
  Kokkos::deep_copy(hypersurface_h, hypersurface);

  params.Add("matter", matter);
  params.Add("matter_h", matter_h);
  params.Add("hypersurface", hypersurface);
  params.Add("hypersurface_h", hypersurface_h);
  params.Add("lapse", alpha);
  params.Add("delta_lapse", dalpha);

  // The radius object, returns radius vs index and index vs radius
  Radius radius(rin, rout, npoints);
  params.Add("radius", radius);

  return gr1d;
}

TaskStatus IntegrateHypersurface(StateDescriptor *pkg) {
  using namespace ShootingMethod;

  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");

  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<GR1D::Radius>("radius");
  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");
  auto matter = params.Get<Matter_t>("matter");
  auto matter_h = params.Get<Matter_host_t>("matter_h");

  Kokkos::deep_copy(matter_h, matter);

  int iA = Hypersurface::A;
  int iK = Hypersurface::K;
  int irho = Matter::RHO;
  int iJ = Matter::J_R;
  Real dr = radius.dx();

  Real state[NHYPER];
  Real k1[NHYPER];
  Real src[NMAT_H];
  Real src_k[NMAT_H];
  Real rhs[NHYPER];
  Real rhs_k[NHYPER];

  hypersurface_h(iA, 0) = 1.0;
  hypersurface_h(iK, 0) = 0.0;
  for (int i = 0; i < npoints - 1; ++i) {
    Real r = radius.x(i);
#pragma omp simd
    for (int v = 0; v < NHYPER; ++v) {
      state[v] = hypersurface_h(v, i);
    }
#pragma omp simd
    for (int v = 0; v < NMAT_H; ++v) {
      src[v] = matter_h(v, i);
    }

    hypersurface_rhs(r, state, src, rhs);
#pragma omp simd
    for (int v = 0; v < NHYPER; ++v) {
      k1[v] = state[v] + 0.5 * dr * rhs[v];
    }
#pragma omp simd
    for (int v = 0; v < NMAT_H; ++v) {
      src_k[v] = 0.5 * (matter_h(v, i) + matter_h(v, i + 1));
    }
    hypersurface_rhs(r, k1, src_k, rhs_k);
#pragma omp simd
    for (int v = 0; v < NHYPER; ++v) {
      hypersurface_h(v, i + 1) = hypersurface_h(v, i) + dr * rhs_k[v];
    }
  }

  Kokkos::deep_copy(hypersurface, hypersurface_h);

  return TaskStatus::complete;
}

TaskStatus JacobiStepForLapse(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto niters_check = params.Get<int>("niters_check");
  auto error_tolerance = params.Get<Real>("error_tolerance");
  auto radius = params.Get<GR1D::Radius>("radius");

  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto matter = params.Get<Matter_t>("matter");
  auto alpha = params.Get<Alpha_t>("lapse");
  auto dalpha = params.Get<Alpha_t>("delta_lapse");

  const Real dr = radius.dx();

  const int iA = GR1D::Hypersurface::A;
  const int iK = GR1D::Hypersurface::K;
  const int iRHO = GR1D::Matter::RHO;
  const int iJ = GR1D::Matter::J_R;
  const int iS = GR1D::Matter::trcS;

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

          // Deltas
          par_for_inner(member, 1, npoints - 2, [&](const int i) {
            Real r = radius.x(i);

            Real a = hypersurface(iA, i);
            Real K = hypersurface(iK, i);
            Real rho = matter(iRHO, i);
            Real S = matter(iS, i);

            Real dadr = ShootingMethod::get_arhs(a, K, r, rho);

            // Formula from mathematica. Unweighted Jacobi
	    /*
            dalpha(i) = ((2 * a * dadr * dr) * alpha(i - 1) +
                         (2 * a + dadr * dr) * alpha(i + 1)) /
                        (a * (4 + a * a * dr * dr * (3 * K * K + 8 * M_PI * (rho + S))));
	    */
            dalpha(i) =
                ((2 * a - dadr * dr) * alpha(i - 1) + (2 * a + dadr * dr) * alpha(i + 1) -
                 a * a * a * dr * dr * (3 * K * K + 8 * M_PI * (S + rho)) * alpha(i)) /
                (4 * a);
            if (dalpha(i) < error_tolerance) dalpha(i) = 1;
            // printf("%d: alphanew = %14e\n", i, dalpha(i));

            dalpha(i) -= alpha(i); // make it a delta
          });
          member.team_barrier();

          // Boundary conditions
          // r = 0
          dalpha(0) = alpha(1) - alpha(0);
          // r outer
	  /*
          Real aout = hypersurface(iA, npoints - 1);
          Real rmax = radius.max();
          dalpha(npoints - 1) =
             dr * ((1 - aout) / (rmax * aout)) + alpha(npoints - 2) - alpha(npoints -
             1);
	  */
          dalpha(npoints - 1) = 1 - alpha(npoints - 1);
          member.team_barrier();

          // Update
          par_for_inner(member, 0, npoints - 1,
                        [&](const int i) { alpha(i) += (2./3.)*dalpha(i); });
          member.team_barrier();
        }
      });

  return TaskStatus::complete;
}

Real LapseError(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return true;

  auto npoints = params.Get<int>("npoints");
  auto error_tolerance = params.Get<Real>("error_tolerance");

  auto alpha = params.Get<Alpha_t>("lapse");
  auto dalpha = params.Get<Alpha_t>("delta_lapse");

  Real max_err = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "GR1D: CheckConvergence",
      parthenon::DevExecSpace(), 0, npoints - 1,
      KOKKOS_LAMBDA(const int i, Real &eps) {
        Real reps = std::abs(dalpha(i) / (alpha(i) + error_tolerance));
        Real aeps = std::abs(dalpha(i));
        eps = std::max(eps, std::min(reps, aeps));
        PARTHENON_REQUIRE(!(std::isnan(reps) || std::isnan(aeps)), "NaNs detected!");
      },
      Kokkos::Max<Real>(max_err));
  max_err = std::abs(max_err);
  return max_err;
}

bool LapseConverged(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return true;

  auto error_tolerance = params.Get<Real>("error_tolerance");
  Real max_err = LapseError(pkg);

  return max_err < error_tolerance;
}

void DumpToTxt(const std::string &filename, StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<GR1D::Radius>("radius");

  auto matter = params.Get<Matter_t>("matter");
  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");
  auto alpha = params.Get<Alpha_t>("lapse");

  auto alpha_h = Kokkos::create_mirror_view(alpha);

  Kokkos::deep_copy(matter_h, matter);
  Kokkos::deep_copy(hypersurface_h, hypersurface);
  Kokkos::deep_copy(alpha_h, alpha);

  FILE *pf;
  pf = fopen(filename.c_str(), "w");
  fprintf(pf, "#r\ta\tK\talpha\trho\tj\tS\n");
  for (int i = 0; i < npoints; ++i) {
    Real r = radius.x(i);
    fprintf(pf, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e\n", r,
            hypersurface_h(Hypersurface::A, i), hypersurface_h(Hypersurface::K, i),
            alpha_h(i), matter_h(Matter::RHO, i), matter_h(Matter::J_R, i),
            matter_h(Matter::trcS, i));
  }
  fclose(pf);
}

} // namespace GR1D
