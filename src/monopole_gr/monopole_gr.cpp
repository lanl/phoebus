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
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

// Parthenon
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Singularity
#include <singularity-eos/eos/eos.hpp>

// Phoebus
#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/robust.hpp"

#include "monopole_gr_interface.hpp"
#include "monopole_gr_utils.hpp"

using namespace parthenon::package::prelude;
using parthenon::AllReduce;

namespace MonopoleGR {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto monopole_gr = std::make_shared<StateDescriptor>("monopole_gr");
  Params &params = monopole_gr->AllParams();

  bool enable_monopole_gr = pin->GetOrAddBoolean("monopole_gr", "enabled", false);
  params.Add("enable_monopole_gr", enable_monopole_gr);

  // Kill time derivatives. Only run on initialization
  bool force_static = pin->GetOrAddBoolean("monopole_gr", "force_static", false);
  params.Add("force_static", force_static);

  // Only run the first n subcycles
  int run_n_times = pin->GetOrAddInteger("monopole_gr", "run_n_times", -1);
  params.Add("run_n_times", run_n_times);
  int nth_call = 0;
  params.Add("nth_call", nth_call, true); // mutable

  // Time step control. Equivalent to CFL
  // Empirical. controls
  Real dtfac = pin->GetOrAddReal("monopole_gr", "dtfac", 0.9);
  params.Add("dtfac", dtfac);

  // Warn if the change in lapse is too different from dalpha/dt
  bool dtwarn = pin->GetOrAddBoolean("monopole_gr", "warn_on_dt", false);
  Real dtwarn_eps = pin->GetOrAddReal("monopole_gr", "dtwarn_eps", 1e-5);
  params.Add("dtwarn", dtwarn);
  params.Add("dtwarn_eps", dtwarn_eps);

  // Points
  int npoints = pin->GetOrAddInteger("monopole_gr", "npoints", 100);
  {
    std::stringstream msg;
    msg << "npoints must be at least " << MIN_NPOINTS << std::endl;
    PARTHENON_REQUIRE_THROWS(npoints >= MIN_NPOINTS, msg);
  }
  params.Add("npoints", npoints);

  // Short-circuit without allocating any memory
  if (!enable_monopole_gr) return monopole_gr;

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("monopole_gr", "rout", 100);

  Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  Real x3max = pin->GetReal("parthenon/mesh", "x3max");

  // Estimate r since we may not know if we're Spherical or Cartesian
  Real r_fluid_est = std::sqrt(x1max * x1max + x2max * x2max + x3max * x3max);
  if (r_fluid_est < rout) {
    std::stringstream msg;
    msg << "Outer radius of fluid may be less than outer radius of of spacetime.\n"
        << "x1max, x2max, x3max, rout_spacetime = " << x1max << ", " << x2max << ", "
        << x3max << ", " << rout << std::endl;
    PARTHENON_WARN(msg);
  }

  params.Add("rin", rin);
  params.Add("rout", rout);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  // TODO(JMM): Extra arrays for cell-centered matter?
  // Probably fine.
  // matter_cells is just shifte to r + dr/2, but contains same number of points.
  Matter_t matter("monopole_gr matter grid. face centered.", NMAT, npoints);
  Matter_t matter_cells("monopole_gr matter grid. cell centered.", NMAT, npoints);
  Volumes_t integration_volumes("monopole_gr matter integration volumes", npoints);
  Hypersurface_t hypersurface("monopole_gr hypersurface grid", NHYPER, npoints);

  // Keep two copies of alpha around... one from this cycle and one
  // from the last. They'll be swapped every cycle.
  Alpha_t alpha("monopole_gr lapse grid", npoints);
  Alpha_t alpha_last("monopole_gr lapse grid", npoints);

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "monopole_gr initialize grids",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        for (int v = 0; v < NMAT; ++v) {
          matter(v, i) = 0;
          matter_cells(v, i) = 0;
        }
        hypersurface(Hypersurface::A, i) = 1;
        hypersurface(Hypersurface::K, i) = 0;
        integration_volumes(i) = 0;
        alpha(i) = 1;
        alpha_last(i) = 1;
      });

  // Host mirrors
  auto matter_h = Kokkos::create_mirror_view(matter);
  auto matter_cells_h = Kokkos::create_mirror_view(matter_cells);
  auto integration_vols_h = Kokkos::create_mirror_view(integration_volumes);
  auto hypersurface_h = Kokkos::create_mirror_view(hypersurface);
  auto alpha_h = Kokkos::create_mirror_view(alpha);

  // Reduction objects.
  AllReduce<Matter_host_t> matter_reducer;
  matter_reducer.val = matter_h;
  AllReduce<Volumes_host_t> volumes_reducer;
  volumes_reducer.val = integration_vols_h;

  // Host-only scratch arrays for Thomas' Method
  Alpha_host_t alpha_m_l("monopole_gr alpha matrix, band below diagonal", npoints);
  Alpha_host_t alpha_m_d("monopole_gr alpha matrix, diagonal", npoints);
  Alpha_host_t alpha_m_u("monopole_gr alpha matrix, band above diagonal", npoints);
  Alpha_host_t alpha_m_b("monopole_gr alpha matrix eqn, rhs", npoints);

  // Device-only arrays for the geometry API
  Gradients_t gradients("monopole_gr gradients", NGRAD, npoints);
  Beta_t beta("monopole_gr shift", npoints);

  params.Add("matter", matter);
  params.Add("matter_h", matter_h);
  params.Add("matter_cells", matter_cells);
  params.Add("matter_cells_h", matter_cells_h);
  params.Add("integration_volumes", integration_volumes);
  params.Add("integration_volumes_h", integration_vols_h);
  params.Add("hypersurface", hypersurface);
  params.Add("hypersurface_h", hypersurface_h);
  params.Add("lapse", alpha);
  params.Add("lapse_h", alpha_h);
  params.Add("lapse_last", alpha_last);
  // mutable because the reducer is stateful
  params.Add("matter_reducer", matter_reducer, true);
  params.Add("volumes_reducer", volumes_reducer, true);
  // M . alpha = b
  params.Add("alpha_m_l", alpha_m_l); // hostonly arrays
  params.Add("alpha_m_d", alpha_m_d);
  params.Add("alpha_m_u", alpha_m_u);
  params.Add("alpha_m_b", alpha_m_b);
  // geometry arrays
  params.Add("gradients", gradients);
  params.Add("shift", beta);

  // The radius object, returns radius vs index and index vs radius
  Radius radius(rin, rout, npoints);
  params.Add("radius", radius);

  monopole_gr->EstimateTimestepBlock = EstimateTimestepBlock;

  return monopole_gr;
}

Real EstimateTimeStep(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  auto &params = pkg->AllParams();
  const auto force_static = params.Get<bool>("force_static");
  if (force_static) {
    return std::numeric_limits<Real>::max();
  }

  const auto dtfac = params.Get<Real>("dtfac");
  const auto npoints = params.Get<int>("npoints");
  auto alpha = params.Get<Alpha_t>("lapse");
  auto gradients = params.Get<Gradients_t>("gradients");

  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "monopole_gr time step",
      parthenon::DevExecSpace(), 0, npoints - 1,
      KOKKOS_LAMBDA(const int i, Real &lmin_dt) {
        Real alpha_tscale =
            std::abs(robust::ratio(alpha, gradients(Gradients::DALPHADT, i)));
        lmin_dt = std::min(lmin_dt, alpha_tscale);
      },
      Kokkos::Min<Real>(min_dt));
  return dtfac * min_dt;
}

// Could template this but whatever. I'd only save like 4 lines.
TaskStatus EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  StateDescriptor *pkg = pmb->packages.Get("monopole_gr").get();
  return EstimateTimeStep(pkg);
}
TaskStatus EstimateTimeStepMesh(MeshData<Real> *rc) {
  auto pmesh = rc->GetParentPointer();
  StateDescriptor *pkg = pmb->packages.Get("monopole_gr").get();
  return EstimateTimeStep(pkg);
}

TaskStatus MatterToHost(StateDescriptor *pkg, bool do_vols) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  const std::string matter_name = do_vols ? "matter_cells" : "matter";
  const std::string matter_h_name = do_vols ? "matter_cells_h" : "matter_h";
  auto matter = params.Get<Matter_t>(matter_name);
  auto matter_h = params.Get<Matter_host_t>(matter_h_name);
  Kokkos::deep_copy(matter_h, matter);

  if (do_vols) {
    auto vols = params.Get<Volumes_t>("integration_volumes");
    auto vols_h = params.Get<Volumes_host_t>("integration_volumes_h");
    Kokkos::deep_copy(vols_h, vols);
  }

  return TaskStatus::complete;
}

// TODO(JMM): This could leverage the cell-centered values
// we compute.
TaskStatus IntegrateHypersurface(StateDescriptor *pkg) {
  using namespace ShootingMethod;

  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");

  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<MonopoleGR::Radius>("radius");

  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");

  int iA = Hypersurface::A;
  int iK = Hypersurface::K;
  // int irho = Matter::RHO;
  // int iJ = Matter::J_R;
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
  for (int v = 0; v < NHYPER; ++v) {
    if (std::isnan(hypersurface_h(v, npoints - 1))) {
      PARTHENON_FAIL("Bad ODE integration.");
    }
  }

  return TaskStatus::complete;
}

TaskStatus LinearSolveForAlpha(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<MonopoleGR::Radius>("radius");

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
  const Real idr = 1. / dr;
  const Real dr2 = dr * dr;
  const Real idr2 = 1. / dr2;

  const int iA = MonopoleGR::Hypersurface::A;
  const int iK = MonopoleGR::Hypersurface::K;
  const int iRHO = MonopoleGR::Matter::RHO;
  const int iS = MonopoleGR::Matter::trcS;

  auto GetCell = [&](const int i, Real &r, Real &a, Real &K, Real &rho, Real &S,
                     Real &dadr) {
    r = radius.x(i);
    a = hypersurface(iA, i);
    K = hypersurface(iK, i);
    rho = matter(iRHO, i);
    S = matter(iS, i);
    dadr = ShootingMethod::GetARHS(a, K, r, rho);
  };

  // define coefficients
  // by rows. Do first and last rows by hand
  for (int i = 1; i < npoints - 1; ++i) {
    Real r, a, K, rho, S, dadr;
    GetCell(i, r, a, K, rho, S, dadr);
    d(i) = -r * (4 + a * a * dr * dr * (3 * K * K + 8 * M_PI * (S + rho))) / (2 * dr);
    u(i) = 1 - (dadr * r / (2 * a)) + r * idr;
    l(i) = -1 + (dadr * r) / (2 * a) + r * idr;
    b(i) = 0;
  }
  { // row 0
    const int i = 0;
    d(i) = -1;
    u(i) = 1;
    b(i) = 0;
  }
  { // row n - 1
    int i = npoints - 1;
    Real r = radius.x(i);
    d(i) = -(r + dr);
    l(i) = r;
    b(i) = -dr;
  }

  // Forward substitution
  for (int i = 1; i < npoints; ++i) {
    Real w = l(i) / d(i - 1);
    d(i) = d(i) - w * u(i - 1);
    b(i) = b(i) - w * b(i - 1);
  }

  // Back substitution
  alpha(npoints - 1) = b(npoints - 1) / d(npoints - 1);
  for (int i = npoints - 2; i >= 0; i -= 1) {
    alpha(i) = (b(i) - u(i) * alpha(i + 1)) / d(i);
  }

  return TaskStatus::complete;
}

TaskStatus SpacetimeToDevice(StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  const auto force_static = params.Get<bool>("force_static");

  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");
  Kokkos::deep_copy(hypersurface, hypersurface_h);

  auto alpha = params.Get<Alpha_t>("lapse");

  // Copy old data from alpha into alpha_last
  auto alpha_last = params.Get<Alpha_t>("lapse_last");
  std::swap(alpha, alpha_last);

  auto alpha_h = params.Get<Alpha_host_t>("lapse_h");
  Kokkos::deep_copy(alpha, alpha_h);

  auto matter = params.Get<Matter_t>("matter");
  auto matter_h = params.Get<Matter_host_t>("matter_h");
  Kokkos::deep_copy(matter, matter_h);

  // Fill device-side arrays
  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<MonopoleGR::Radius>("radius");
  const Real dr = radius.dx();
  const Real dr2 = dr * dr;
  auto beta = params.Get<Beta_t>("shift");
  auto gradients = params.Get<Gradients_t>("gradients");
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "monopole_gr gradients and shift",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        if (force_static) {
          hypersurface(Hypersurface::K, i) = 0;
        }

        auto mask = !(force_static);
        Real r = radius.x(i);
        Real a = hypersurface(Hypersurface::A, i);
        Real K = mask * hypersurface(Hypersurface::K, i);
        Real rho = matter(Matter::RHO, i);
        Real j = matter(Matter::J_R, i);
        Real S = matter(Matter::trcS, i);
        Real Srr = matter(Matter::Srr, i);
        Real dadr = ShootingMethod::GetARHS(a, K, r, rho);
        Real dKdr = mask * ShootingMethod::GetKRHS(a, K, r, j);

        Real a2 = a * a;
        Real a3 = a2 * a;
        Real K2 = K * K;
        Real dalphadr, d2alphadr2;
        if (i == 0) {
          dalphadr = 0;
          d2alphadr2 = (alpha(i) + alpha(i + 2) - 2 * alpha(i + 1)) / dr2;
        } else if (i == npoints - 1) {
          dalphadr = (alpha(i) - alpha(i - 1)) / dr;
          d2alphadr2 = (alpha(i - 2) + alpha(i) - 2 * alpha(i - 1)) / dr2;
        } else {
          dalphadr = (alpha(i + 1) - alpha(i - 1)) / (2. * dr);
          d2alphadr2 = (alpha(i - 1) + alpha(i + 1) - 2 * alpha(i)) / dr2;
        }

        beta(i) = mask * (-0.5 * alpha(i) * r * K);
        Real dbetadr =
            mask * (-0.5 * (r * K * dalphadr + alpha(i) * K + alpha(i) * r * dKdr));
        Real beta2 = beta(i) * beta(i);
        Real beta3 = beta(i) * beta2;

        gradients(Gradients::DADR, i) = dadr;
        gradients(Gradients::DKDR, i) = dKdr;
        gradients(Gradients::DALPHADR, i) = dalphadr;
        gradients(Gradients::DBETADR, i) = dbetadr;

        Real dadt = dadr * beta(i) + a * dbetadr - alpha(i) * a * K;
        if (i == 0) dadt = 0;
        Real dalphadt =
            beta(i) * ((a * beta(i) * dadt / alpha(i)) + a2 * K * beta(i) +
                       2 * dadr * (1 - beta2) - 2 * (a2 * beta(i) / alpha(i)) * dbetadr);
        Real dadror = (r > 1e-2) ? dadr / r : 1;
        Real dKdt = beta(i) * dKdr - (d2alphadr2 / a2) + (dadr / a3) * dalphadr +
                    alpha(i) * ((2 * dadror / a3) - 4 * K * K) +
                    4 * M_PI * alpha(i) * (S - rho - 2 * Srr);
        if (i == 0) dKdt = 0;
        Real dbetadt = -0.5 * r * (alpha(i) * dKdt + K * dalphadt);

        // printf("%d: %.15e %.15e %.15e %.15e\n", i, dadt, dalphadt, dKdt, dbetadt);
        gradients(Gradients::DADT, i) = mask * dadt;
        gradients(Gradients::DALPHADT, i) = mask * dalphadt;
        gradients(Gradients::DKDT, i) = mask * dKdt;
        gradients(Gradients::DBETADT, i) = mask * dbetadt;
      });

  return TaskStatus::complete;
}

TaskStatus CheckRateOfChange(StateDescriptor *pkg, Real dt) {
  auto dtwarn = params.Get<bool>("dtwarn");
  if (dtwarn && parthenon::Globals::my_rank == 0) {
    auto alpha = params.Get<Alpha_t>("lapse");
    auto alpha_last = params.Get<Alpha_t>("lapse_last");
    auto npoints = params.Get<int>("npoints");
    auto gradients = params.Get<Gradients_t>("gradients");
    auto dtwarn_eps = params.Get<Real>("dtwarn_eps");

    Real err;
    parthenon::par_reduce(
        parthenon::loop_pattern_flatrange_tag, "monopole_gr check evolution time scales",
        parthenon::DevExecSpace(), 0, npoints - 1,
        KOKKOS_LAMBDA(const int i, Real &lmax_err) {
          Real diff_alpha = (alpha(i) - alpha_last(i)) / dt;
          Real diff_dalpha = std::abs(diff_alpha - gradients(Gradients::DALPHADT, i));
          lmax_err = std::max(lmax_err, diff_dalpha);
        },
        Kokkos::max<Real>(err));
    if (err > dtwarn_eps) {
      stc::cerr
          << "\tWarning! (alpha^i+1 - alpha^i)/dt is very different from dalpha/dt.\n"
          << "\t\tTolerance set to " << dtwarn_eps << std::endl;
    }
  }
  return TaskStatus::complete;
}

TaskStatus DivideVols(StateDescriptor *pkg) {
  using robust::ratio;

  auto &params = pkg->AllParams();
  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  auto npoints = params.Get<int>("npoints");
  auto matter_cells_h = params.Get<Matter_host_t>("matter_cells_h");
  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto vols_h = params.Get<Volumes_host_t>("integration_volumes_h");

  // Divide by volumes
  for (int i = 0; i < npoints; ++i) {
    for (int v = 0; v < NMAT; ++v) {
      matter_cells_h(v, i) = ratio(matter_cells_h(v, i), std::abs(vols_h(i)));
    }
  }
  // Shift to the face-centered grid
  for (int i = 1; i < npoints; ++i) {
    for (int v = 0; v < NMAT; ++v) {
      matter_h(v, i) = 0.5 * (matter_cells_h(v, i) + matter_cells_h(v, i - 1));
    }
  }
  matter_cells_h(Matter::J_R, 0) = 0;
  matter_cells_h(Matter::Srr, 0) = 0;
  for (int v = 0; v < NMAT; ++v) {
    matter_h(v, 0) = matter_cells_h(v, 0);
  }

  return TaskStatus::complete;
}

void DumpToTxt(const std::string &filename, StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(pkg->label() == "monopole_gr",
                           "Requires the monopole_gr package");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return;

  auto npoints = params.Get<int>("npoints");
  auto radius = params.Get<MonopoleGR::Radius>("radius");

  auto matter = params.Get<Matter_t>("matter");
  auto matter_h = params.Get<Matter_host_t>("matter_h");
  auto hypersurface = params.Get<Hypersurface_t>("hypersurface");
  auto hypersurface_h = params.Get<Hypersurface_host_t>("hypersurface_h");
  auto alpha = params.Get<Alpha_t>("lapse");
  auto alpha_h = params.Get<Alpha_host_t>("lapse_h");

  auto gradients = params.Get<Gradients_t>("gradients");
  auto gradients_h = Kokkos::create_mirror_view(gradients);

  auto matter_cells_h = params.Get<Matter_host_t>("matter_cells_h");
  auto vols_h = params.Get<Volumes_host_t>("integration_volumes_h");

  Kokkos::deep_copy(matter_h, matter);
  Kokkos::deep_copy(hypersurface_h, hypersurface);
  Kokkos::deep_copy(alpha_h, alpha);
  Kokkos::deep_copy(gradients_h, gradients);

  FILE *pf;
  pf = fopen(filename.c_str(), "w");
  fprintf(pf, "#r\ta\tK\talpha\trho\tj\tS\n");
  for (int i = 0; i < npoints; ++i) {
    Real r = radius.x(i);
    fprintf(pf,
            "%.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e "
            "%.14e %.14e %.14e %.14e %.14e %.14e %.14e %.14e "
            "%.14e %.14e %.14e %.14e %.14e\n",
            r, hypersurface_h(Hypersurface::A, i), hypersurface_h(Hypersurface::K, i),
            alpha_h(i), matter_h(Matter::RHO, i), matter_h(Matter::J_R, i),
            matter_h(Matter::trcS, i), matter_h(Matter::Srr, i),
            gradients_h(Gradients::DADR, i), gradients_h(Gradients::DKDR, i),
            gradients_h(Gradients::DALPHADR, i), gradients_h(Gradients::DBETADR, i),
            gradients_h(Gradients::DADT, i), gradients_h(Gradients::DALPHADT, i),
            gradients_h(Gradients::DKDR, i), gradients_h(Gradients::DBETADT, i),
            matter_cells_h(Matter::RHO, i), matter_cells_h(Matter::J_R, i),
            matter_cells_h(Matter::trcS, i), matter_cells_h(Matter::Srr, i), vols_h(i));
  }
  fclose(pf);
}

} // namespace MonopoleGR
