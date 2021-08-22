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
    msg << "npoints must be at least " << MIN_NPOINTS << std::endl;
    PARTHENON_REQUIRE_THROWS(npoints >= MIN_NPOINTS, msg);
  }
  params.Add("npoints", npoints);

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("GR1D", "rout", 100);
  params.Add("rin", rin);
  params.Add("rout", rout);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Matter_t matter("GR1D matter grid", NMAT, npoints);
  Hypersurface_t hypersurface("GR1D hypersurface grid", NHYPER, npoints);
  Alpha_t alpha("GR1D lapse grid", npoints);
  
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "GR1D initialize grids",
      parthenon::DevExecSpace(), 0, npoints - 1,
      KOKKOS_LAMBDA(const int i) {
        for (int v = 0; v < NMAT; ++v) {
          matter(v, i) = 0;
        }
        hypersurface(Hypersurface::A, i) = 1;
        hypersurface(Hypersurface::K, i) = 0;
        alpha(i) = 1;
      });

  auto matter_h = Kokkos::create_mirror_view(matter);
  auto hypersurface_h = Kokkos::create_mirror_view(hypersurface);
  auto alpha_h = Kokkos::create_mirror_view(alpha);

  Alpha_host_t alpha_m_l("GR1D alpha matrix, band below diagonal", npoints);
  Alpha_host_t alpha_m_d("GR1D alpha matrix, diagonal", npoints);
  Alpha_host_t alpha_m_u("GR1D alpha matrix, band above diagonal", npoints);
  Alpha_host_t alpha_m_b("GR1D alpha matrix eqn, rhs", npoints);

  params.Add("matter", matter);
  params.Add("matter_h", matter_h);
  params.Add("hypersurface", hypersurface);
  params.Add("hypersurface_h", hypersurface_h);
  params.Add("lapse", alpha);
  params.Add("lapse_h", alpha_h);
  // M . alpha = b
  params.Add("alpha_m_l", alpha_m_l); // hostonly arrays
  params.Add("alpha_m_d", alpha_m_d);
  params.Add("alpha_m_u", alpha_m_u);
  params.Add("alpha_m_b", alpha_m_b);

  // The radius object, returns radius vs index and index vs radius
  Radius radius(rin, rout, npoints);
  params.Add("radius", radius);

  return gr1d;
}

TaskStatus MatterToHost(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto matter = params.Get<Matter_t>("matter");
  auto matter_h = params.Get<Matter_host_t>("matter_h");
  Kokkos::deep_copy(matter_h, matter);

  return TaskStatus::complete;
}

TaskStatus IntegrateHypersurface(StateDescriptor *pkg) {
  using namespace ShootingMethod;

  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");

  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<GR1D::Radius>("radius");

  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");

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

  return TaskStatus::complete;
}

TaskStatus LinearSolveForAlpha(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<GR1D::Radius>("radius");

  // Everything done on host
  auto hypersurface = params.Get<Hypersurface_host_t>("hypersurface_h");
  auto matter = params.Get<Matter_host_t>("matter_h");
  auto alpha = params.Get<Alpha_host_t>("lapse_h");

  auto l = params.Get<Alpha_host_t>("alpha_m_l");
  auto d = params.Get<Alpha_host_t>("alpha_m_d");
  auto u = params.Get<Alpha_host_t>("alpha_m_u");
  auto b = params.Get<Alpha_host_t>("alpha_m_b");

  const Real rmax = radius.max();
  const Real dr = radius.dx();
  const Real idr = 1./dr;
  const Real dr2 = dr*dr;
  const Real idr2 = 1./dr2;

  const int iA = GR1D::Hypersurface::A;
  const int iK = GR1D::Hypersurface::K;
  const int iRHO = GR1D::Matter::RHO;
  const int iS = GR1D::Matter::trcS;

  auto GetCell = [&](const int i, Real &r, Real &a, Real &K, Real &rho, Real &S,
                     Real &dadr, Real &denom) {
    r = radius.x(i);
    a = hypersurface(iA, i);
    K = hypersurface(iK, i);
    rho = matter(iRHO, i);
    S = matter(iS, i);
    dadr = ShootingMethod::GetARHS(a, K, r, rho);
    denom = a*(4 + a * a * dr2 * (3 * K * K + 8 * M_PI * (S + rho)));
  };

  // define coefficients
  // by rows. Do first and last rows by hand
  for (int i = 1; i < npoints-1; ++i) {
    Real r, a, K, rho, S, dadr, denom;
    GetCell(i, r, a, K, rho, S, dadr, denom);
    d(i) = -r*(4 + a*a*dr*dr*(3*K*K + 8*M_PI*(S + rho)))/(2*dr);
    u(i) = 1 - (dadr*r/(2*a)) + r*idr;
    l(i) = -1 + (dadr*r)/(2*a) + r*idr;
    b(i) = 0;
  }
  { // row 0
    const int i = 0;
    Real r, a, K, rho, S, dadr, fac;
    GetCell(i, r, a, K, rho, S, dadr, fac);
    d(i) = -1;
    u(i) = 1;
    b(i) = 0;
  }
  { // row n - 1
    int i = npoints - 1;
    Real r, a, K, rho, S, dadr, fac;
    GetCell(i, r, a, K, rho, S, dadr, fac);
    d(i) = -(r + dr);
    l(i) = r;
    b(i) = -dr;
  }

  //Forward substitution
  for (int i = 1; i < npoints; ++i) {
    Real w = l(i)/d(i-1);
    d(i) = d(i) - w*u(i-1);
    b(i) = b(i) - w*b(i-1);
  }

  // Back substitution
  alpha(npoints - 1) = b(npoints-1)/d(npoints-1);
  for (int i = npoints - 2; i >= 0; i -= 1) {
    alpha(i) = (b(i) - u(i)*alpha(i+1))/d(i);
  }

  return TaskStatus::complete;
}

TaskStatus SpacetimeToDevice(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "GR1D", "Requires the GR1D package");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_gr1d");
  if (!enabled) return TaskStatus::complete;

  auto hypersurface = params.Get<Matter_t>("hypersurface");
  auto hypersurface_h = params.Get<Matter_host_t>("hypersurface_h");
  Kokkos::deep_copy(hypersurface, hypersurface_h);

  auto alpha = params.Get<Alpha_t>("lapse");
  auto alpha_h = params.Get<Alpha_host_t>("lapse_h");
  Kokkos::deep_copy(alpha, alpha_h);

  return TaskStatus::complete;
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
  auto alpha_h = params.Get<Alpha_host_t>("lapse_h");

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
