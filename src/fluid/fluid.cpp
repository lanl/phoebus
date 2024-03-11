// © 2021-2022. Triad National Security, LLC. All rights reserved.
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

#include "fluid.hpp"

#include "analysis/history.hpp"
#include "con2prim.hpp"
#include "con2prim_robust.hpp"
#include "fixup/fixup.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"
#include "reconstruction.hpp"
#include "riemann.hpp"
#include "tmunu.hpp"

#include <globals.hpp>
#include <interface/sparse_pack.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using parthenon::MetadataFlag;

// statically defined vars from riemann.hpp
std::vector<std::string> riemann::FluxState::recon_vars, riemann::FluxState::flux_vars;

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace diag = diagnostic_variables;
  auto physics = std::make_shared<StateDescriptor>("fluid");
  Params &params = physics->AllParams();

  // Check that we are actually evolving the fluid
  const bool active = pin->GetBoolean("physics", "hydro");
  params.Add("active", active);
  if (active) { // Only set up these parameters if the fluid is evolved

    const bool zero_fluxes = pin->GetOrAddBoolean("fluid", "zero_fluxes", false);
    params.Add("zero_fluxes", zero_fluxes);

    const bool zero_sources = pin->GetOrAddBoolean("fluid", "zero_sources", false);
    params.Add("zero_sources", zero_sources);

    Real cfl = pin->GetOrAddReal("fluid", "cfl", 0.8);
    params.Add("cfl", cfl);

    std::string c2p_method = pin->GetOrAddString("fluid", "c2p_method", "robust");
    params.Add("c2p_method", c2p_method);
    if (c2p_method == "robust") {
      params.Add("c2p_func", ConservedToPrimitiveRobust<MeshBlockData<Real>>);
    } else if (c2p_method == "classic") {
      params.Add("c2p_func", ConservedToPrimitiveClassic<MeshBlockData<Real>>);
    } else {
      PARTHENON_THROW("Invalid c2p_method.");
    }

    Real c2p_tol = pin->GetOrAddReal("fluid", "c2p_tol", 1.e-8);
    params.Add("c2p_tol", c2p_tol);

    int c2p_max_iter = pin->GetOrAddInteger("fluid", "c2p_max_iter", 20);
    params.Add("c2p_max_iter", c2p_max_iter);

    // The factor by which the floors in c2p are reduced from the floors used in
    // ApplyFloors
    Real c2p_floor_scale_fac = pin->GetOrAddReal("fluid", "c2p_floor_scale_fac", 1.);
    PARTHENON_REQUIRE(c2p_floor_scale_fac <= 1. && c2p_floor_scale_fac > 0.,
                      "fluid/c2p_floor_scale_fac must be between 0 and 1!");
    params.Add("c2p_floor_scale_fac", c2p_floor_scale_fac);

    bool c2p_fail_on_floors = pin->GetOrAddBoolean("fluid", "c2p_fail_on_floors", false);
    params.Add("c2p_fail_on_floors", c2p_fail_on_floors);

    bool c2p_fail_on_ceilings =
        pin->GetOrAddBoolean("fluid", "c2p_fail_on_ceilings", false);
    params.Add("c2p_fail_on_ceilings", c2p_fail_on_ceilings);

    std::string recon = pin->GetOrAddString("fluid", "recon", "linear");
    PhoebusReconstruction::ReconType rt = PhoebusReconstruction::ReconType::linear;
    if (recon == "weno5" || recon == "weno5z") {
      PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                               "weno5 requires 4+ ghost cells");
      rt = PhoebusReconstruction::ReconType::weno5z;
    } else if (recon == "mp5") {
      PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                               "mp5 requires 4+ ghost cells");
      if (cfl > 0.4) {
        PARTHENON_WARN("mp5 often requires smaller cfl numbers for stability");
      }
      rt = PhoebusReconstruction::ReconType::mp5;
    } else if (recon == "linear") {
      rt = PhoebusReconstruction::ReconType::linear;
    } else if (recon == "constant") {
      rt = PhoebusReconstruction::ReconType::constant;
    } else {
      PARTHENON_THROW("Invalid Reconstruction option.  Choose from "
                      "[constant,linear,mp5,weno5,weno5z]");
    }
    params.Add("Recon", rt);

    std::string solver = pin->GetOrAddString("fluid", "riemann", "hll");
    riemann::solver rs = riemann::solver::HLL;
    if (solver == "llf") {
      rs = riemann::solver::LLF;
    } else if (solver == "hll") {
      rs = riemann::solver::HLL;
    } else {
      PARTHENON_THROW("Invalid Riemann Solver option. Choose from [llf, hll]");
    }
    params.Add("RiemannSolver", rs);
  }
  bool ye = pin->GetOrAddBoolean("fluid", "Ye", false);
  params.Add("Ye", ye);

  bool mhd = pin->GetOrAddBoolean("fluid", "mhd", false);
  params.Add("mhd", mhd);

  Real sigma_cutoff = pin->GetOrAddReal("fluid", "sigma_cutoff", 1.0);
  params.Add("sigma_cutoff", sigma_cutoff);

  std::vector<int> three_vec(1, 3);

  std::vector<MetadataFlag> prim_flags_vector = {Metadata::Cell, Metadata::Intensive,
                                                 Metadata::Vector, Metadata::Derived,
                                                 Metadata::OneCopy};
  std::vector<MetadataFlag> prim_flags_scalar = {Metadata::Cell, Metadata::Intensive,
                                                 Metadata::Derived, Metadata::OneCopy};
  std::vector<MetadataFlag> cons_flags_scalar = {Metadata::Cell, Metadata::Independent,
                                                 Metadata::Intensive, Metadata::Conserved,
                                                 Metadata::WithFluxes};
  std::vector<MetadataFlag> cons_flags_vector = {
      Metadata::Cell,      Metadata::Independent, Metadata::Intensive,
      Metadata::Conserved, Metadata::Vector,      Metadata::WithFluxes};

  const std::string bc_vars = pin->GetOrAddString("phoebus/mesh", "bc_vars", "conserved");
  params.Add("bc_vars", bc_vars);

  if (bc_vars == "conserved") {
    cons_flags_scalar.push_back(Metadata::FillGhost);
    cons_flags_vector.push_back(Metadata::FillGhost);
  } else if (bc_vars == "primitive") {
    prim_flags_scalar.push_back(Metadata::FillGhost);
    prim_flags_vector.push_back(Metadata::FillGhost);
    // TODO(BRR) Still set FillGhost on conserved variables to ensure buffers exist.
    // Fixing this requires modifying parthenon Metadata logic.
    cons_flags_scalar.push_back(Metadata::FillGhost);
    cons_flags_vector.push_back(Metadata::FillGhost);
  } else {
    PARTHENON_REQUIRE_THROWS(
        bc_vars == "conserved" || bc_vars == "primitive",
        "\"bc_vars\" must be either \"conserved\" or \"primitive\"!");
  }

  Metadata mprim_threev = Metadata(prim_flags_vector, three_vec);
  Metadata mprim_scalar = Metadata(prim_flags_scalar);
  Metadata mcons_scalar = Metadata(cons_flags_scalar);
  Metadata mcons_threev = Metadata(cons_flags_vector, three_vec);

  // TODO(BRR) Should these go in a "phoebus" package?
  const std::string ix1_bc = pin->GetString("phoebus", "ix1_bc");
  params.Add("ix1_bc", ix1_bc);
  const std::string ox1_bc = pin->GetString("phoebus", "ox1_bc");
  params.Add("ox1_bc", ox1_bc);
  const std::string ix2_bc = pin->GetString("phoebus", "ix2_bc");
  params.Add("ix2_bc", ix2_bc);
  const std::string ox2_bc = pin->GetString("phoebus", "ox2_bc");
  params.Add("ox2_bc", ox2_bc);

  int ndim = 1;
  if (pin->GetInteger("parthenon/mesh", "nx3") > 1)
    ndim = 3;
  else if (pin->GetInteger("parthenon/mesh", "nx2") > 1)
    ndim = 2;

  // add the primitive variables
  physics->template AddField<p::density>(mprim_scalar);
  physics->AddField(p::velocity::name(), mprim_threev);
  physics->AddField(p::energy::name(), mprim_scalar);
  if (mhd) {
    physics->AddField(p::bfield::name(), mprim_threev);
    if (ndim == 2) {
      physics->AddField(impl::emf::name(), mprim_scalar);
    } else if (ndim == 3) {
      physics->AddField(impl::emf::name(), mprim_threev);
    }
    physics->AddField(diag::divb::name(), mprim_scalar);
  }
  physics->AddField(p::pressure::name(), mprim_scalar);
  physics->AddField(p::temperature::name(), mprim_scalar);
  physics->AddField(p::entropy::name(), mprim_scalar);
  physics->AddField(p::cs::name(), mprim_scalar);
  physics->AddField(diag::ratio_divv_cs::name(), mprim_scalar);
  physics->AddField(diag::central_density::name(), mprim_scalar);
  physics->AddField(diag::localization_function::name(), mprim_scalar);
  physics->AddField(diag::entropy_z_0::name(), mprim_scalar);
  physics->AddField(p::gamma1::name(), mprim_scalar);
  if (ye) {
    physics->AddField(p::ye::name(), mprim_scalar);
  }
  if (pin->GetOrAddString("fluid", "c2p_method", "robust") == "robust") {
    physics->AddField(impl::c2p_mu::name(), mprim_scalar);
  }
  // Just want constant primitive fields around to serve as
  // background if we are not evolving the fluid, don't need
  // to do the rest.
  // TODO(BRR) logic gets very complicated accounting for this in situations where
  // radiation is active but fluid isn't -- for example we want fluid prims from c2p to
  // calculate opacities.
  // if (!active) return physics;

  // this fail flag should really be an enum or something
  // but parthenon doesn't yet support that kind of thing
  physics->AddField(impl::fail::name(), mprim_scalar);

  // add the conserved variables
  physics->AddField(c::density::name(), mcons_scalar);
  physics->AddField(c::momentum::name(), mcons_threev);
  physics->AddField(c::energy::name(), mcons_scalar);
  if (mhd) {
    physics->AddField(c::bfield::name(), mcons_threev);
  }
  if (ye) {
    physics->AddField(c::ye::name(), mcons_scalar);
  }

  AllReduce<std::vector<Real>> net_field_totals;
  AllReduce<std::vector<Real>> net_field_totals_2;
  physics->AddParam<>("net_field_totals", net_field_totals, true);
  physics->AddParam<>("net_field_totals_2", net_field_totals_2, true);

  // If fluid is not active, still don't add reconstruction variables
  if (!active) return physics;

#if SET_FLUX_SRC_DIAGS
  // DIAGNOSTIC STUFF FOR DEBUGGING
  std::vector<int> five_vec(1, 5);
  Metadata mdiv = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                            Metadata::Derived, Metadata::OneCopy},
                           five_vec);
  physics->AddField(diag::divf::name(), mdiv);
  Metadata mdiag = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                             Metadata::Derived, Metadata::OneCopy},
                            five_vec);
  physics->AddField(diag::src_terms::name(), mdiag);
#endif

  // set up the arrays for left and right states
  // add the base state for reconstruction/fluxes
  std::vector<std::string> rvars(
      {p::density::name(), p::velocity::name(), p::energy::name()});
  riemann::FluxState::ReconVars(rvars);
  if (mhd) riemann::FluxState::ReconVars(p::bfield::name());
  if (ye) riemann::FluxState::ReconVars(p::ye::name());

  std::vector<std::string> fvars(
      {c::density::name(), c::momentum::name(), c::energy::name()});
  riemann::FluxState::FluxVars(fvars);
  if (mhd) riemann::FluxState::FluxVars(c::bfield::name());
  if (ye) riemann::FluxState::FluxVars(c::ye::name());

  // add some extra fields for reconstruction
  rvars = std::vector<std::string>({p::pressure::name(), p::gamma1::name()});
  riemann::FluxState::ReconVars(rvars);

  auto recon_vars = riemann::FluxState::ReconVars();
  int nrecon = 0;
  for (const auto &v : recon_vars) {
    auto &m = physics->FieldMetadata(v);
    auto &shape = m.Shape();
    int size = 1;
    for (const auto &s : shape) {
      size *= s;
    }
    nrecon += size;
  }

  std::vector<int> recon_shape({nrecon, ndim});
  Metadata mrecon =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, recon_shape);
  physics->AddField(impl::ql::name(), mrecon);
  physics->AddField(impl::qr::name(), mrecon);

  std::vector<int> signal_shape(1, ndim);
  Metadata msignal =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, signal_shape);
  physics->AddField(impl::face_signal_speed::name(), msignal);
  physics->AddField(impl::cell_signal_speed::name(), msignal);

  std::vector<int> c2p_scratch_size(1, 5);
  Metadata c2p_meta =
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, c2p_scratch_size);
  physics->AddField(impl::c2p_scratch::name(), c2p_meta);

  // Reductions
  // By default compute integrated value of scalar conserved vars
  auto HstSum = parthenon::UserHistoryOperation::sum;
  using History::ReduceInGain;
  using History::ReduceOneVar;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};

  auto ReduceMass = [](MeshData<Real> *md) {
    return ReduceOneVar<Kokkos::Sum<Real>>(md, fluid_cons::density::name(), 0);
  };
  auto ReduceEn = [](MeshData<Real> *md) {
    return ReduceOneVar<Kokkos::Sum<Real>>(md, fluid_cons::energy::name(), 0);
  };
  auto CentralDensitySN = [](MeshData<Real> *md) {
    History::ReduceCentralDensitySN(md);
    return ReduceOneVar<Kokkos::Max<Real>>(md, diag::central_density::name(), 0);
  };
  auto Mgain = [](MeshData<Real> *md) {
    return ReduceInGain<Kokkos::Sum<Real, HostExecSpace>>(md, fluid_prim::density::name(),
                                                          0);
  };

  hst_vars.emplace_back(HistoryOutputVar(HstSum, CentralDensitySN, "central density SN"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, Mgain, "Mgain"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, ReduceMass, "total baryon number"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, ReduceEn, "total conserved energy tau"));

  for (int d = 0; d < 3; ++d) {
    auto ReduceMom = [d](MeshData<Real> *md) {
      return History::ReduceOneVar<Kokkos::Sum<Real>>(md, fluid_cons::momentum::name(),
                                                      d);
    };
    hst_vars.emplace_back(HistoryOutputVar(
        HstSum, ReduceMom, "total X" + std::to_string(d + 1) + " momentum"));
  }

  params.Add(parthenon::hist_param_key, hst_vars);

  // Fill Derived and Estimate Timestep
  physics->FillDerivedBlock = ConservedToPrimitive<MeshBlockData<Real>>;
  physics->EstimateTimestepBlock = EstimateTimestepBlock;

  return physics;
}

// template <typename T>
TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  return PrimitiveToConservedRegion(rc, ib, jb, kb);
}

// template <typename T>
TaskStatus PrimitiveToConservedRegion(MeshBlockData<Real> *rc, const IndexRange &ib,
                                      const IndexRange &jb, const IndexRange &kb) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  using parthenon::MakePackDescriptor;

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  const int ndim = pmesh->ndim;

  static auto desc =
      MakePackDescriptor<p::density, c::density, p::velocity, c::momentum, p::energy,
                         c::energy, p::bfield, c::bfield, p::ye, c::ye, p::pressure,
                         p::gamma1, impl::cell_signal_speed>(resolved_pkgs.get());
  auto v = desc.GetPack(rc);

  // We need these to check whether or not these variables are present
  // in the pack. They are -1 if not present.
  const int contains_b = v.ContainsHost(0, p::bfield());
  const int contains_ye = v.ContainsHost(0, p::ye());

  auto geom = Geometry::GetCoordinateSystem(rc);
  const int nblocks = v.GetNBlocks();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(), 0, nblocks - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &coords = v.GetCoordinates(b);
        Real gcov4[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
        Real gcon[3][3];
        geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
        Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
        Real shift[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

        Real S[3];
        const Real vel[] = {v(b, p::velocity(0), k, j, i), v(b, p::velocity(1), k, j, i),
                            v(b, p::velocity(2), k, j, i)};
        Real bcons[3];
        Real bp[3] = {0.0, 0.0, 0.0};
        if (contains_b) {
          for (int d = 0; d < 3; ++d) {
            bp[d] = v(b, p::bfield(d), k, j, i);
          }
        }
        Real ye_cons;
        Real ye_prim = 0.0;
        if (contains_ye) {
          ye_prim = v(b, p::ye(), k, j, i);
        }

        Real sig[3];
        prim2con::p2c(v(b, p::density(), k, j, i), vel, bp, v(b, p::energy(), k, j, i),
                      ye_prim, v(b, p::pressure(), k, j, i), v(b, p::gamma1(), k, j, i),
                      gcov4, gcon, shift, lapse, gdet, v(b, c::density(), k, j, i), S,
                      bcons, v(b, c::energy(), k, j, i), ye_cons, sig);

        for (int d = 0; d < 3; ++d) {
          v(b, c::momentum(d), k, j, i) = S[d];
        }

        if (contains_b) {
          for (int d = 0; d < 3; ++d) {
            v(b, c::bfield(d), k, j, i) = bcons[d];
          }
        }

        if (contains_ye) {
          v(b, c::ye(), k, j, i) = ye_cons;
        }

        for (int d = 0; d < ndim; d++) {
          v(b, impl::cell_signal_speed(d), k, j, i) = sig[d];
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitiveRegion(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb) {
  Mesh *pm = rc->GetMeshPointer();
  StateDescriptor *pkg = pm->packages.Get("fluid").get();
  auto c2p = pkg->Param<c2p_type<T>>("c2p_func");
  return c2p(rc, ib, jb, kb);
}

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
  return ConservedToPrimitiveRegion(rc, ib, jb, kb);
}

template <typename T>
TaskStatus ConservedToPrimitiveRobust(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb) {
  // TODO(JMM): This one will not work with meshblock packs
  auto *pmb = rc->GetParentPointer();

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  const Real c2p_floor_scale_fac = pkg->Param<Real>("c2p_floor_scale_fac");
  const bool c2p_fail_on_floors = pkg->Param<bool>("c2p_fail_on_floors");
  const bool c2p_fail_on_ceilings = pkg->Param<bool>("c2p_fail_on_ceilings");
  auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter,
                                                c2p_floor_scale_fac, c2p_fail_on_floors,
                                                c2p_fail_on_ceilings);

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto coords = pmb->coords;

  // TODO(JCD): move the setting of this into the solver so we can call this on MeshData
  auto fail = rc->Get(internal_variables::fail::name()).data;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve", DevExecSpace(), 0, invert.NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto status = invert(geom, eos, coords, k, j, i);
        fail(k, j, i) = (status == con2prim_robust::ConToPrimStatus::success
                             ? con2prim_robust::FailFlags::success
                             : con2prim_robust::FailFlags::fail);
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitiveClassic(T *rc, const IndexRange &ib, const IndexRange &jb,
                                       const IndexRange &kb) {
  using namespace con2prim;
  auto *pmesh = rc->GetMeshPointer();

  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  StateDescriptor *pkg = pmesh->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim::ConToPrimSetup(rc, c2p_tol, c2p_max_iter);

  StateDescriptor *eos_pkg = pmesh->packages.Get("eos").get();
  auto eos = eos_pkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto fail = rc->Get(internal_variables::fail::name()).data;

  // breaking con2prim into 3 kernels seems more performant.  WHY?
  // if we can combine them, we can get rid of the mesh sized scratch array
  // TODO(JCD): revisit this.  don't think it's required anymore.  in fact the
  //            original performance thing was related to the loop being a reduce
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Setup", DevExecSpace(), 0, invert.NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        invert.Setup(geom, k, j, i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve", DevExecSpace(), 0, invert.NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto status = invert(eos, k, j, i);
        fail(k, j, i) =
            (status == ConToPrimStatus::success ? FailFlags::success : FailFlags::fail);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Finalize", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        invert.Finalize(eos, geom, k, j, i);
      });

  return TaskStatus::complete;
}

TaskStatus CalculateFluidSourceTerms(MeshData<Real> *rc, MeshData<Real> *rc_src) {
  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  using phoebus::MakePackDescriptor;
  namespace c = fluid_cons;
  namespace diag = diagnostic_variables;

  Mesh *pmesh = rc->GetMeshPointer();
  auto &fluid = pmesh->packages.Get("fluid");
  if (!fluid->Param<bool>("active") || fluid->Param<bool>("zero_sources"))
    return TaskStatus::complete;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  static auto desc = MakePackDescriptor<c::momentum, c::energy
#if SET_FLUX_SRC_DIAGS
                                        ,
                                        diag::src_terms
#endif
                                        >(rc);
  auto src = desc.GetPack(rc_src);
  auto tmunu = BuildStressEnergyTensor(rc);
  auto geom = Geometry::GetCoordinateSystem(rc);
  const int nblocks = src.GetNBlocks();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TmunuSourceTerms", DevExecSpace(), 0, nblocks - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real Tmunu[ND][ND], dg[ND][ND][ND], gam[ND][ND][ND];
        tmunu(Tmunu, b, k, j, i);
        Real gdet = geom.DetG(CellLocation::Cent, b, k, j, i);

        geom.MetricDerivative(CellLocation::Cent, b, k, j, i, dg);
        Geometry::Utils::SetConnectionCoeffFromMetricDerivs(dg, gam);

        // momentum source terms
        SPACELOOP(l) {
          Real src_mom = 0.0;
          SPACETIMELOOP2(m, n) {
            src_mom += Tmunu[m][n] * (dg[n][l + 1][m] - gam[l + 1][n][m]);
          }
          src(b, c::momentum(l), k, j, i) = gdet * src_mom;
        }

        // energy source term
        {
          Real TGam = 0.0;
#if USE_VALENCIA
          // TODO(jcd): maybe use the lapse and shift here instead of gcon
          const Real alpha = geom.Lapse(CellLocation::Cent, b, k, j, i);
          const Real inv_alpha2 = robust::ratio(1, alpha * alpha);
          Real shift[NS];
          geom.ContravariantShift(CellLocation::Cent, b, k, j, i, shift);
          Real gcon0[4] = {-inv_alpha2, inv_alpha2 * shift[0], inv_alpha2 * shift[1],
                           inv_alpha2 * shift[2]};
          for (int m = 0; m < ND; m++) {
            for (int n = 0; n < ND; n++) {
              Real gam0 = 0;
              for (int r = 0; r < ND; r++) {
                gam0 += gcon0[r] * gam[r][m][n];
              }
              TGam += Tmunu[m][n] * gam0;
            }
          }
          Real Ta = 0.0;
          Real da[ND];
          // Real *da = &gam[1][0][0];
          geom.GradLnAlpha(CellLocation::Cent, b, k, j, i, da);
          for (int m = 0; m < ND; m++) {
            Ta += Tmunu[m][0] * da[m];
          }
          src(b, c::energy(), k, j, i) = gdet * alpha * (Ta - TGam);
#else
          SPACETIMELOOP2(mu, nu) { TGam += Tmunu[mu][nu] * gam[nu][0][mu]; }
          src(b, c::energy(), k, j, i) = gdet * TGam;
#endif // USE_VALENCIA
        }

#if SET_FLUX_SRC_DIAGS
        src(diag::src_terms(0), b, k, j, i) = 0.0;
        src(diag::src_terms(1), b, k, j, i) = src(c::momentum(0), k, j, i);
        src(diag::src_terms(2), b, k, j, i) = src(c::momentum(1), k, j, i);
        src(diag::src_terms(3), b, k, j, i) = src(c::momentum(2), k, j, i);
        src(diag::src_terms(4), b, k, j, i) = src(c::energy(), k, j, i);
#endif
      });

  return TaskStatus::complete;
}

TaskStatus CalculateFluxes(MeshBlockData<Real> *rc) {
  using namespace PhoebusReconstruction;
  Mesh *pmesh = rc->GetMeshPointer();
  auto &fluid = pmesh->packages.Get("fluid");
  if (!fluid->Param<bool>("active") || fluid->Param<bool>("zero_fluxes"))
    return TaskStatus::complete;

  auto flux = riemann::FluxState(rc);
  auto sig = rc->Get(internal_variables::face_signal_speed::name()).data;

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  const int ndim = pmesh->ndim;
  const int dk = (ndim == 3 ? 1 : 0);
  const int dj = (ndim > 1 ? 1 : 0);
  const int nrecon = flux.ql.GetDim(4) - 1;
  auto rt =
      pmesh->packages.Get("fluid")->Param<PhoebusReconstruction::ReconType>("Recon");
  auto st = pmesh->packages.Get("fluid")->Param<riemann::solver>("RiemannSolver");

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Reconstruct", DevExecSpace(), 0, 0, 0, nrecon,
      kb.s - dk, kb.e + dk, jb.s - dj, jb.e + dj,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int n, const int k, const int j) {
        // this approach (pullilng out pointers) depends on i being the fastest moving
        // index
        Real *pv = &flux.v(n, k, j, 0);

        Real *pvim2 = pv - 2;
        Real *pvim1 = pv - 1;
        Real *pvip1 = pv + 1;
        Real *pvip2 = pv + 2;
        Real *vi_l = &flux.ql(0, n, k, j, 1);
        Real *vi_r = &flux.qr(0, n, k, j, 0);

        Real *pvjm2 = &flux.v(n, k, j - 2 * dj, 0);
        Real *pvjm1 = &flux.v(n, k, j - dj, 0);
        Real *pvjp1 = &flux.v(n, k, j + dj, 0);
        Real *pvjp2 = &flux.v(n, k, j + 2 * dj, 0);
        Real *vj_l = &flux.ql(1 % ndim, n, k, j + dj, 0);
        Real *vj_r = &flux.qr(1 % ndim, n, k, j, 0);

        Real *pvkm2 = &flux.v(n, k - 2 * dk, j, 0);
        Real *pvkm1 = &flux.v(n, k - dk, j, 0);
        Real *pvkp1 = &flux.v(n, k + dk, j, 0);
        Real *pvkp2 = &flux.v(n, k + 2 * dk, j, 0);
        Real *vk_l = &flux.ql(2 % ndim, n, k + dk, j, 0);
        Real *vk_r = &flux.qr(2 % ndim, n, k, j, 0);

        switch (rt) {
        case ReconType::weno5z:
          ReconLoop<WENO5Z>(member, ib.s - 1, ib.e + 1, pvim2, pvim1, pv, pvip1, pvip2,
                            vi_l, vi_r);
          if (ndim > 1)
            ReconLoop<WENO5Z>(member, ib.s - 1, ib.e + 1, pvjm2, pvjm1, pv, pvjp1, pvjp2,
                              vj_l, vj_r);
          if (ndim > 2)
            ReconLoop<WENO5Z>(member, ib.s - 1, ib.e + 1, pvkm2, pvkm1, pv, pvkp1, pvkp2,
                              vk_l, vk_r);
          break;
        case ReconType::mp5:
          ReconLoop<MP5>(member, ib.s - 1, ib.e + 1, pvim2, pvim1, pv, pvip1, pvip2, vi_l,
                         vi_r);
          if (ndim > 1)
            ReconLoop<MP5>(member, ib.s - 1, ib.e + 1, pvjm2, pvjm1, pv, pvjp1, pvjp2,
                           vj_l, vj_r);
          if (ndim > 2)
            ReconLoop<MP5>(member, ib.s - 1, ib.e + 1, pvkm2, pvkm1, pv, pvkp1, pvkp2,
                           vk_l, vk_r);
          break;
        case ReconType::linear:
          ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvim1, pv, pvip1, vi_l,
                                     vi_r);
          if (ndim > 1)
            ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvjm1, pv, pvjp1, vj_l,
                                       vj_r);
          if (ndim > 2)
            ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvkm1, pv, pvkp1, vk_l,
                                       vk_r);
          break;
        case ReconType::constant:
          ReconLoop<PiecewiseConstant>(member, ib.s - 1, ib.e + 1, pv, vi_l, vi_r);
          if (ndim > 1)
            ReconLoop<PiecewiseConstant>(member, ib.s - 1, ib.e + 1, pv, vj_l, vj_r);
          if (ndim > 2)
            ReconLoop<PiecewiseConstant>(member, ib.s - 1, ib.e + 1, pv, vk_l, vk_r);
          break;
        default:
          PARTHENON_FAIL("Invalid recon option.");
        }
      });

#define FLUX(method)                                                                     \
  parthenon::par_for(                                                                    \
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(), X1DIR, ndim, kb.s - dk,   \
      kb.e + dk, jb.s - dj, jb.e + dj, ib.s - 1, ib.e + 1,                               \
      KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {                \
        sig(d - 1, k, j, i) = method(flux, d, k, j, i);                                  \
      });
  switch (st) {
  case riemann::solver::LLF:
    FLUX(riemann::llf);
    break;
  case riemann::solver::HLL:
    FLUX(riemann::hll);
    break;
  default:
    PARTHENON_THROW("Invalid riemann solver option.");
  }
#undef FLUX

  return TaskStatus::complete;
}

TaskStatus FluxCT(MeshBlockData<Real> *rc) {
  Mesh *pmesh = rc->GetMeshPointer();
  auto &fluid = pmesh->packages.Get("fluid");
  if (!fluid->Param<bool>("mhd") || !fluid->Param<bool>("active") ||
      fluid->Param<bool>("zero_fluxes"))
    return TaskStatus::complete;

  const int ndim = pmesh->ndim;
  if (ndim == 1) return TaskStatus::complete;
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto f1 = rc->Get(fluid_cons::bfield::name()).flux[X1DIR];
  auto f2 = rc->Get(fluid_cons::bfield::name()).flux[X2DIR];
  auto f3 = rc->Get(fluid_cons::bfield::name()).flux[X3DIR];
  auto emf = rc->Get(internal_variables::emf::name()).data;

  if (ndim == 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::EMF::2D", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e + 1, ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          emf(k, j, i) = 0.25 * (f1(1, k, j, i) + f1(1, k, j - 1, i) - f2(0, k, j, i) -
                                 f2(0, k, j, i - 1));
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::Flux::2D", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e + 1, ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          f1(0, k, j, i) = 0.0;
          f1(1, k, j, i) = 0.5 * (emf(k, j, i) + emf(k, j + 1, i));
          f2(0, k, j, i) = -0.5 * (emf(k, j, i) + emf(k, j, i + 1));
          f2(1, k, j, i) = 0.0;
        });
  } else if (ndim == 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::EMF::3D", DevExecSpace(), kb.s, kb.e + 1, jb.s,
        jb.e + 1, ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          emf(0, k, j, i) = 0.25 * (f2(2, k, j, i) + f2(2, k - 1, j, i) - f3(1, k, j, i) -
                                    f3(1, k, j - 1, i));
          emf(1, k, j, i) = -0.25 * (f1(2, k, j, i) + f1(2, k - 1, j, i) -
                                     f3(0, k, j, i) - f3(0, k, j, i - 1));
          emf(2, k, j, i) = 0.25 * (f1(1, k, j, i) + f1(1, k, j - 1, i) - f2(0, k, j, i) -
                                    f2(0, k, j, i - 1));
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::Flux::3D", DevExecSpace(), kb.s, kb.e + 1, jb.s,
        jb.e + 1, ib.s, ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          f1(0, k, j, i) = 0.0;
          f1(1, k, j, i) = 0.5 * (emf(2, k, j, i) + emf(2, k, j + 1, i));
          f1(2, k, j, i) = -0.5 * (emf(1, k, j, i) + emf(1, k + 1, j, i));
          f2(0, k, j, i) = -0.5 * (emf(2, k, j, i) + emf(2, k, j, i + 1));
          f2(1, k, j, i) = 0.0;
          f2(2, k, j, i) = 0.5 * (emf(0, k, j, i) + emf(0, k + 1, j, i));
          f3(0, k, j, i) = 0.5 * (emf(1, k, j, i) + emf(1, k, j, i + 1));
          f3(1, k, j, i) = -0.5 * (emf(0, k, j, i) + emf(0, k, j + 1, i));
          f3(2, k, j, i) = 0.0;
        });
  }

  return TaskStatus::complete;
}

TaskStatus CalculateDivB(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("active")) return TaskStatus::complete;
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd")) return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  // This is the problem for doing things with meshblock packs
  auto coords = pmb->coords;
  auto b = rc->Get(fluid_cons::bfield::name()).data;
  auto divb = rc->Get(diagnostic_variables::divb::name()).data;
  if (ndim == 2) {
    // todo(jcd): these are supposed to be node centered, and this misses the
    // high boundaries
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DivB::2D", DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          divb(k, j, i) = 0.5 *
                              (b(0, k, j, i) + b(0, k, j - 1, i) - b(0, k, j, i - 1) -
                               b(0, k, j - 1, i - 1)) /
                              coords.CellWidthFA(X1DIR, k, j, i) +
                          0.5 *
                              (b(1, k, j, i) + b(1, k, j, i - 1) - b(1, k, j - 1, i) -
                               b(1, k, j - 1, i - 1)) /
                              coords.CellWidthFA(X2DIR, k, j, i);
        });
  } else if (ndim == 3) {
    // todo(jcd): these are supposed to be node centered, and this misses the
    // high boundaries
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DivB::3D", DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          divb(k, j, i) =
              0.25 *
                  (b(0, k, j, i) + b(0, k, j - 1, i) + b(0, k - 1, j, i) +
                   b(0, k - 1, j - 1, i) - b(0, k, j, i - 1) - b(0, k, j - 1, i - 1) -
                   b(0, k - 1, j, i - 1) - b(0, k - 1, j - 1, i - 1)) /
                  coords.CellWidthFA(X1DIR, k, j, i) +
              0.25 *
                  (b(1, k, j, i) + b(1, k, j, i - 1) + b(1, k - 1, j, i) +
                   b(1, k - 1, j, i - 1) - b(1, k, j - 1, i) - b(1, k, j - 1, i - 1) -
                   b(1, k - 1, j - 1, i) - b(1, k - 1, j - 1, i - 1)) /
                  coords.CellWidthFA(X2DIR, k, j, i) +
              0.25 *
                  (b(2, k, j, i) + b(2, k, j, i - 1) + b(2, k, j - 1, i) +
                   b(2, k, j - 1, i - 1) - b(2, k - 1, j, i) - b(2, k - 1, j, i - 1) -
                   b(2, k - 1, j - 1, i) - b(2, k - 1, j - 1, i - 1)) /
                  coords.CellWidthFA(X3DIR, k, j, i);
        });
  }
  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  auto &pars = pmb->packages.Get("fluid")->AllParams();
  Real min_dt;
  auto csig = rc->Get(internal_variables::cell_signal_speed::name()).data;
  if (pars.hasKey("has_face_speeds")) {
    auto fsig = rc->Get(internal_variables::face_signal_speed::name()).data;
    pmb->par_reduce(
        "Hydro::EstimateTimestep::1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
          Real ldt = 0.0;
          for (int d = 0; d < ndim; d++) {
            const int di = (d == 0);
            const int dj = (d == 1);
            const int dk = (d == 2);
            const Real max_s =
                std::max(csig(d, k, j, i),
                         std::max(fsig(d, k, j, i), fsig(d, k + dk, j + dj, i + di)));
            ldt += max_s / coords.CellWidthFA(X1DIR + d, k, j, i);
          }
          lmin_dt = std::min(lmin_dt, 1.0 / (ldt + 1.e-50));
        },
        Kokkos::Min<Real>(min_dt));
  } else {
    pmb->par_reduce(
        "Hydro::EstimateTimestep::0", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
          Real ldt = 0.0;
          for (int d = 0; d < ndim; d++) {
            ldt += csig(d, k, j, i) / coords.CellWidthFA(X1DIR + d, k, j, i);
          }
          lmin_dt = std::min(lmin_dt, 1.0 / (ldt + 1.e-50));
        },
        Kokkos::Min<Real>(min_dt));
    pars.Add("has_face_speeds", true);
  }
  const auto &cfl = pars.Get<Real>("cfl");
  return cfl * min_dt;
}

} // namespace fluid
