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
#include <sstream>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Phoebus
#include "geometry/geometry_utils.hpp"

#include "gr1d.hpp"
#include "gr1d_utils.hpp"

using namespace parthenon::package::prelude;

namespace GR1D {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto gr1d = std::make_shared<StateDescriptor>("GR1D");
  Params &params = gr1d->AllParams();

  bool enable_gr1d = pin->GetOrAddBoolean("GR1D", "enabled", false);
  params.Add("enable_gr1d", enable_gr1d);
  if (!enable_gr1d) return gr1d; // Short-circuit with nothing

  // Points and Refinement levels
  int npoints = pin->GetOrAddInteger("GR1D", "npoints", 100);
  {
    std::stringstream msg;
    msg << "npoints must be at least " << Multigrid::MIN_NPOINTS << std::endl;
    PARTHENON_REQUIRE_THROWS(npoints >= Multigrid::MIN_NPOINTS, msg);
  }

  int nlevels = log2(npoints);
  int npoints_reconstructed = 1 << nlevels;
  if (npoints != npoints_reconstructed + 1) {
    std::stringstream msg;
    msg << "GR1D: npoints = " << npoints << " is not 2^p - 1. Setting to "
        << npoints_reconstructed + 1<< std::endl;
    PARTHENON_WARN(msg);
    npoints = npoints_reconstructed + 1;
    nlevels = log2(npoints);
  }
  // Size of coarsest level should be Multigrid::MIN_NPOINTS If finest
  // level already has MIN_NPOINTS, there should be only one level
  nlevels = (nlevels - Multigrid::LEVEL_OFFSET) + 1;
  params.Add("npoints", npoints);
  params.Add("nlevels", nlevels);

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
  Matter_t matter("GR1D matter grid", nlevels, NMAT, npoints);
  Hypersurface_t hypersurface("GR1D hypersurface grid", nlevels, NHYPER, npoints);
  Alpha_t alpha("GR1D lapse grid", nlevels, npoints);
  Alpha_t alpha_r("GR1D alpha residual grid", nlevels, npoints);
  Alpha_t alpha_src("GR1D source term for alpha", nlevels, npoints);
  
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 0, nlevels - 1, 0, npoints - 1,
      KOKKOS_LAMBDA(const int l, const int i) {
        for (int v = 0; v < NMAT; ++v) {
          matter(l, v, i) = 0;
        }
        hypersurface(l, Hypersurface::A, i) = 1;
        hypersurface(l, Hypersurface::K, i) = 0;
        alpha(l, i) = (l == 0) ? 1 : 0;
	alpha_r(l, i) = 0;
        alpha_src(l, i) = 0;
      });

  auto matter_fine = matter.Get<2>();
  auto hypersurface_fine = hypersurface.Get<2>();

  auto matter_h = Kokkos::create_mirror_view(matter_fine);
  auto hypersurface_h = Kokkos::create_mirror_view(hypersurface_fine);

  params.Add("matter", matter);
  params.Add("matter_h", matter_h);
  params.Add("hypersurface", hypersurface);
  params.Add("hypersurface_h", hypersurface_h);
  params.Add("lapse", alpha);
  params.Add("lapse_residual", alpha_r);
  params.Add("lapse_source", alpha_src);

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
  auto matter = params.Get<Matter_t>("matter");

  auto matter_fine = matter.Get<2>();
  auto hypersurface_fine = hypersurface.Get<2>();

  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");

  Kokkos::deep_copy(matter_h, matter_fine);

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

    HypersurfaceRHS(r, state, src, rhs);
#pragma omp simd
    for (int v = 0; v < NHYPER; ++v) {
      k1[v] = state[v] + 0.5 * dr * rhs[v];
    }
#pragma omp simd
    for (int v = 0; v < NMAT_H; ++v) {
      src_k[v] = 0.5 * (matter_h(v, i) + matter_h(v, i + 1));
    }
    HypersurfaceRHS(r, k1, src_k, rhs_k);
#pragma omp simd
    for (int v = 0; v < NHYPER; ++v) {
      hypersurface_h(v, i + 1) = hypersurface_h(v, i) + dr * rhs_k[v];
    }
  }

  Kokkos::deep_copy(hypersurface_fine, hypersurface_h);

  return TaskStatus::complete;
}

TaskStatus RestrictHypersurface(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto nlevels = params.Get<int>("nlevels");

  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto matter = params.Get<Matter_t>("matter");
  auto alpha = params.Get<Alpha_t>("lapse");

  for (int level = 1; level < nlevels; ++level) {
    auto npoints_coarse = Multigrid::NPointsPLevel(level, npoints);
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag,
        "restrict to level " + std::to_string(level), parthenon::DevExecSpace(), 0,
        npoints_coarse - 1, KOKKOS_LAMBDA(const int icoarse) {
          Multigrid::RestrictVar(hypersurface, level, icoarse, NHYPER, npoints_coarse);
          Multigrid::RestrictVar(matter, level, icoarse, NMAT, npoints_coarse);
        });
  }
  return TaskStatus::complete;
}

TaskStatus ResetLevels(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto nlevels = params.Get<int>("nlevels");

  auto alpha = params.Get<Alpha_t>("lapse");
  auto alpha_r = params.Get<Alpha_t>("lapse_residual");
  auto alpha_src = params.Get<Alpha_t>("lapse_source");

  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 1, nlevels - 1, 0, npoints - 1,
      KOKKOS_LAMBDA(const int l, const int i) {
        alpha(l, i) = 0;
	alpha_r(l, i) = 0;
        alpha_src(l, i) = 0;
      });
  return TaskStatus::complete;
}

TaskStatus JacobiStepForLapse(StateDescriptor *pkg, const int l) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints_fine = params.Get<int>("npoints");
  auto npoints = Multigrid::NPointsPLevel(l, npoints_fine);

  auto error_tolerance = params.Get<Real>("error_tolerance");
  auto radius = params.Get<GR1D::Radius>("radius");

  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto matter = params.Get<Matter_t>("matter");
  auto alpha = params.Get<Alpha_t>("lapse");
  auto alpha_r = params.Get<Alpha_t>("lapse_residual");
  auto alpha_src = params.Get<Alpha_t>("lapse_source");

  const Real dr = radius.dx();

  const int iA = GR1D::Hypersurface::A;
  const int iK = GR1D::Hypersurface::K;
  const int iRHO = GR1D::Matter::RHO;
  const int iJ = GR1D::Matter::J_R;
  const int iS = GR1D::Matter::trcS;

  bool error_eqn = l > 0;
  Real jacobi_weight = (2./3.);//std::min(1.0,0.5*dr*dr);
  //jacobi_weight /= (1 << l);

  // Residual
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D compute residual",
      parthenon::DevExecSpace(), 1, npoints - 2, KOKKOS_LAMBDA(const int i) {
        Real r = radius.x(i);

        Real a = hypersurface(l, iA, i);
        Real K = hypersurface(l, iK, i);
        Real rho = matter(l, iRHO, i);
        Real S = matter(l, iS, i);

        Real dadr = ShootingMethod::GetARHS(a, K, r, rho);

	alpha_r(l, i) =
	  ((-2 * a + dadr * dr) * alpha(l, i - 1) -
	   (2 * a + dadr * dr) * alpha(l, i + 1) +
	   a * (4 + a * a * dr * dr * (3 * K * K + 8 * M_PI * (S + rho))) *
	   alpha(l, i)) /
	  (2 * a * dr * dr);
	if (error_eqn) {
	  alpha_r(l, i) += alpha_src(l,i);
	}

      });
  // update
  // don't update boundary if we're in error-correction mode.
  // the boundary = 0 in error_correction mode.
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D jacobi update",
      parthenon::DevExecSpace(), 0, npoints - 1,
      KOKKOS_LAMBDA(const int i) {

	Real r = radius.x(i);
        Real a = hypersurface(l, iA, i);
        Real K = hypersurface(l, iK, i);
        Real rho = matter(l, iRHO, i);
        Real S = matter(l, iS, i);

	if (i == 0) { // r inner
	  alpha(l, i) = error_eqn ? 0 : alpha(l, i + 1);
	} else if (i == npoints - 1) { // r outer
	  alpha(l, i) = error_eqn ? 0 : (dr + r * alpha(i - 1)) / (dr + r);
	} else {
	  Real Dinverse = -2*dr*dr/(4 + a*a*dr*dr*(3*K*K + 8*M_PI*(rho+S)));
	  Real update = Dinverse*alpha_r(l, i);
	  alpha(l,i) += jacobi_weight*update;
	}
	if (l == 0 && alpha(l,i) < error_tolerance) {
	  alpha(l,i) = error_tolerance;
	}
      });

  return TaskStatus::complete;
}

TaskStatus RestrictAlphaResidual(StateDescriptor *pkg, const int lcoarse) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints_finest = params.Get<int>("npoints");
  auto npoints = Multigrid::NPointsPLevel(lcoarse, npoints_finest);

  auto res = params.Get<Alpha_t>("lapse_residual");
  auto src = params.Get<Alpha_t>("lapse_source");

  const int lfine = lcoarse - 1;

  // skip boundaries. The residual and error vanish at boundaries.
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D restrict alpha",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int icoarse) {
        const int ifine = 2 * icoarse;
	if (icoarse == 0 || icoarse == npoints - 1) {
	  src(lcoarse,icoarse) = 0;
	} else {
	  src(lcoarse, icoarse) = 0.25 * (res(lfine, ifine - 1) + 2 * res(lfine, ifine) +
					  res(lfine, ifine + 1));
	}
      });

  return TaskStatus::complete;
}

TaskStatus ErrorCorrectAlpha(StateDescriptor *pkg, const int lfine) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints_finest = params.Get<int>("npoints");
  auto npoints = Multigrid::NPointsPLevel(lfine, npoints_finest);
  auto alpha = params.Get<Alpha_t>("lapse");

  const int lcoarse = lfine + 1;
  const int npoints_coarse = npoints / 2 + 1;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D error correction step",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int ifine) {
        const int icoarse = ifine / 2;
        if (ifine % 2 == 0) { // these line up at nodes
          alpha(lfine, ifine) += alpha(lcoarse, icoarse);
        } else { // these require interpolation
          alpha(lfine, ifine) +=
              0.5 * (alpha(lcoarse, icoarse) + alpha(lcoarse, icoarse + 1));
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
  auto dalpha = params.Get<Alpha_t>("lapse_residual");

  Real max_err = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "GR1D: CheckConvergence",
      parthenon::DevExecSpace(), 1, npoints - 2,
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

  auto matter_fine = matter.Get<2>();
  auto hypersurface_fine = hypersurface.Get<2>();

  auto alpha_fine = alpha.Get<1>();
  auto alpha_h = Kokkos::create_mirror_view(alpha_fine);

  Kokkos::deep_copy(matter_h, matter_fine);
  Kokkos::deep_copy(hypersurface_h, hypersurface_fine);
  Kokkos::deep_copy(alpha_h, alpha_fine);

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
