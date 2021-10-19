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

#include "fluid.hpp"

#include "con2prim.hpp"
#include "con2prim_robust.hpp"
#include "con2prim_vanderholst.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"
#include "prim2con.hpp"
#include "reconstruction.hpp"
#include "riemann.hpp"
#include "tmunu.hpp"

#include <singularity-eos/eos/eos.hpp>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

// statically defined vars from riemann.hpp
std::vector<std::string> riemann::FluxState::recon_vars,
    riemann::FluxState::flux_vars;

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace diag = diagnostic_variables;
  auto physics = std::make_shared<StateDescriptor>("fluid");
  Params &params = physics->AllParams();

  const bool hydro = pin->GetBoolean("physics", "hydro");
  params.Add("hydro", hydro);

  Real cfl = pin->GetOrAddReal("fluid", "cfl", 0.8);
  params.Add("cfl", cfl);

  std::string c2p_method = pin->GetOrAddString("fluid", "c2p_method", "robust");
  params.Add("c2p_method", c2p_method);
  if (c2p_method == "robust") {
    params.Add("c2p_func", ConservedToPrimitiveRobust<MeshBlockData<Real>>);
  } else if (c2p_method == "classic") {
    params.Add("c2p_func", ConservedToPrimitiveClassic<MeshBlockData<Real>>);
  } else if (c2p_method == "vanderholst") {
    // TODO(BRR) only valid for ideal gas
    params.Add("c2p_func", ConservedToPrimitiveVanDerHolst<MeshBlockData<Real>>);
  } else {
    PARTHENON_THROW("Invalid c2p_method.");
  }

  Real c2p_tol = pin->GetOrAddReal("fluid", "c2p_tol", 1.e-8);
  params.Add("c2p_tol", c2p_tol);

  int c2p_max_iter = pin->GetOrAddInteger("fluid", "c2p_max_iter", 20);
  params.Add("c2p_max_iter", c2p_max_iter);

  std::string recon = pin->GetOrAddString("fluid", "recon", "linear");
  PhoebusReconstruction::ReconType rt =
      PhoebusReconstruction::ReconType::linear;
  if (recon == "weno5" || recon == "weno5z") {
    PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                             "weno5 requires 4+ ghost cells");
    rt = PhoebusReconstruction::ReconType::weno5z;
  } else if (recon == "weno5a") {
    PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                             "weno5 requires 4+ ghost cells");
    rt = PhoebusReconstruction::ReconType::weno5a;
  } else if (recon == "mp5") {
    PARTHENON_REQUIRE_THROWS(parthenon::Globals::nghost >= 4,
                             "mp5 requires 4+ ghost cells");
    if (cfl > 0.4) {
      PARTHENON_WARN("mp5 often requires smaller cfl numbers for stability");
    }
    rt = PhoebusReconstruction::ReconType::mp5;
  } else if (recon == "linear") {
    rt = PhoebusReconstruction::ReconType::linear;
  } else {
    PARTHENON_THROW(
        "Invalid Reconstruction option.  Choose from [linear,weno5]");
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

  bool ye = pin->GetOrAddBoolean("fluid", "Ye", false);
  params.Add("Ye", ye);

  bool mhd = pin->GetOrAddBoolean("fluid", "mhd", false);
  params.Add("mhd", mhd);

  Metadata m;
  std::vector<int> three_vec(1, 3);

  const std::string bc_vars = pin->GetOrAddString("phoebus/mesh", "bc_vars", "conserved");
  params.Add("bc_vars", bc_vars);

  Metadata  mprim_threev =
      Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                Metadata::Derived, Metadata::OneCopy},
               three_vec);
  Metadata mprim_scalar = Metadata({Metadata::Cell, Metadata::Intensive,
                                    Metadata::Derived, Metadata::OneCopy});
  Metadata mcons_scalar =
      Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
            Metadata::Conserved, Metadata::WithFluxes});
  Metadata mcons_threev =
      Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
            Metadata::Conserved, Metadata::Vector, Metadata::WithFluxes},
               three_vec);

  if (bc_vars == "conserved") {
    mcons_scalar.Set(Metadata::FillGhost);
    mcons_threev.Set(Metadata::FillGhost);
  } else if (bc_vars == "primitive") {
    mprim_scalar.Set(Metadata::FillGhost);
    mprim_threev.Set(Metadata::FillGhost);
    // TODO(BRR) Still set FillGhost on conserved variables to ensure buffers exist.
    // Fixing this requires modifying parthenon Metadata logic.
    mcons_scalar.Set(Metadata::FillGhost);
    mcons_threev.Set(Metadata::FillGhost);
  } else {
    PARTHENON_REQUIRE_THROWS(bc_vars == "conserved" || bc_vars == "primitive",
      "\"bc_vars\" must be either \"conserved\" or \"primitive\"!");
  }

  int ndim = 1;
  if (pin->GetInteger("parthenon/mesh", "nx3") > 1)
    ndim = 3;
  else if (pin->GetInteger("parthenon/mesh", "nx2") > 1)
    ndim = 2;

  // add the primitive variables
  physics->AddField(p::density, mprim_scalar);
  physics->AddField(p::velocity, mprim_threev);
  physics->AddField(p::energy, mprim_scalar);
  if (mhd) {
    physics->AddField(p::bfield, mprim_threev);
    if (ndim == 2) {
      physics->AddField(impl::emf, mprim_scalar);
    } else if (ndim == 3) {
      physics->AddField(impl::emf, mprim_threev);
    }
    physics->AddField(diag::divb, mprim_scalar);
  }
  physics->AddField(p::pressure, mprim_scalar);
  physics->AddField(p::temperature, mprim_scalar);
  physics->AddField(p::gamma1, mprim_scalar);
  if (ye) {
    physics->AddField(p::ye, mprim_scalar);
  }
  // this fail flag should really be an enum or something
  // but parthenon doesn't yet support that kind of thing
  physics->AddField(impl::fail, mprim_scalar);

  // add the conserved variables
  physics->AddField(c::density, mcons_scalar);
  physics->AddField(c::momentum, mcons_threev);
  physics->AddField(c::energy, mcons_scalar);
  if (mhd) {
    physics->AddField(c::bfield, mcons_threev);
  }
  if (ye) {
    physics->AddField(c::ye, mcons_scalar);
  }

  // DIAGNOSTIC STUFF FOR DEBUGGING
  std::vector<int> five_vec(1,5);
  Metadata mdiv = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                             Metadata::Derived, Metadata::OneCopy},
                             five_vec);
  physics->AddField("flux_divergence", mdiv);
  std::vector<int> seven_vec(1,7);
  Metadata mdiag = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                             Metadata::Derived, Metadata::OneCopy},
                             seven_vec);
  physics->AddField("src_terms", mdiag);


  // set up the arrays for left and right states
  // add the base state for reconstruction/fluxes
  std::vector<std::string> rvars({p::density, p::velocity, p::energy});
  riemann::FluxState::ReconVars(rvars);
  if (mhd)
    riemann::FluxState::ReconVars(p::bfield);
  if (ye)
    riemann::FluxState::ReconVars(p::ye);

  std::vector<std::string> fvars({c::density, c::momentum, c::energy});
  riemann::FluxState::FluxVars(fvars);
  if (mhd)
    riemann::FluxState::FluxVars(c::bfield);
  if (ye)
    riemann::FluxState::FluxVars(c::ye);

  // add some extra fields for reconstruction
  rvars = std::vector<std::string>({p::pressure, p::gamma1});
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
  Metadata mrecon = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
			     recon_shape);
  physics->AddField(impl::ql, mrecon);
  physics->AddField(impl::qr, mrecon);

  std::vector<int> signal_shape(1, ndim);
  Metadata msignal =
	  Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
		   signal_shape);
  physics->AddField(impl::face_signal_speed, msignal);
  physics->AddField(impl::cell_signal_speed, msignal);

  std::vector<int> c2p_scratch_size(1, 5);
  Metadata c2p_meta =
	  Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
		   c2p_scratch_size);
  physics->AddField(impl::c2p_scratch, c2p_meta);

  physics->FillDerivedBlock = ConservedToPrimitive<MeshBlockData<Real>>;
  physics->EstimateTimestepBlock = EstimateTimestepBlock;

  return physics;
}

//template <typename T>
TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc) {
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  return PrimitiveToConservedRegion(rc, ib, jb, kb);
}

// template <typename T>
TaskStatus PrimitiveToConservedRegion(MeshBlockData<Real> *rc, const IndexRange &ib, const IndexRange &jb, const IndexRange &kb) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  auto *pmb = rc->GetParentPointer().get();

  const std::vector<std::string> vars({p::density, c::density, p::velocity,
                                       c::momentum, p::energy, c::energy, p::bfield,
                                       c::bfield, p::ye, c::ye, p::pressure, p::gamma1,
                                       impl::cell_signal_speed});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;
  const int prs = imap[p::pressure].first;
  const int gm1 = imap[p::gamma1].first;
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  const int cb_lo = imap[c::bfield].first;
  const int cb_hi = imap[c::bfield].second;
  const int pye = imap[p::ye].second; // -1 if not present
  const int cye = imap[c::ye].second;
  const int sig_lo = imap[impl::cell_signal_speed].first;
  const int sig_hi = imap[impl::cell_signal_speed].second;

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real gcov4[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
        Real gcon[3][3];
        geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
        Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
        Real shift[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);

        Real S[3];
        const Real vel[] = {v(b, pvel_lo, k, j, i),
                            v(b, pvel_lo+1, k, j, i),
                            v(b, pvel_hi, k, j, i)};
        Real bcons[3];
        Real bp[3] = {0.0, 0.0, 0.0};
        if (pb_hi > 0) {
          bp[0] = v(b, pb_lo, k, j, i);
          bp[1] = v(b, pb_lo+1, k, j, i);
          bp[2] = v(b, pb_hi, k, j, i);
        }
        Real ye_cons;
        Real ye_prim = 0.0;
        if (pye > 0) {
          ye_prim = v(b, pye, k, j, i);
        }

        Real sig[3];
        prim2con::p2c(v(b, prho, k, j, i), vel, bp, v(b, peng, k, j, i), ye_prim,
            v(b, prs, k, j, i), v(b, gm1, k, j, i),
            gcov4, gcon, shift, lapse, gdet,
            v(b, crho, k, j, i), S, bcons, v(b, ceng, k, j, i), ye_cons, sig);

        v(b, cmom_lo, k, j, i) = S[0];
        v(b, cmom_lo+1, k, j, i) = S[1];
        v(b, cmom_hi, k, j, i) = S[2];

        if (pb_hi > 0) {
          v(b, cb_lo, k, j, i) = bcons[0];
          v(b, cb_lo+1, k, j, i) = bcons[1];
          v(b, cb_hi, k, j, i) = bcons[2];
        }

        if (pye > 0) {
          v(b, cye, k, j, i) = ye_cons;
        }

        for (int m = sig_lo; m <= sig_hi; m++) {
          v(b, m, k, j, i) = sig[m-sig_lo];
        }
      });

  return TaskStatus::complete;
}


template <typename T>
TaskStatus ConservedToPrimitiveRobust(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb) {
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter);

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto coords = pmb->coords;

  auto fail = rc->Get(internal_variables::fail).data;

  std::vector<std::string> diag_var({fluid_prim::energy});
  auto diag = rc->PackVariables(diag_var);

  // breaking con2prim into 3 kernels seems more performant.  WHY?
  // if we can combine them, we can get rid of the mesh sized scratch array
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto status = invert(geom, eos, coords, k, j, i);
        fail(k, j, i) = (status == con2prim_robust::ConToPrimStatus::success
                                 ? con2prim_robust::FailFlags::success
                                 : con2prim_robust::FailFlags::fail);
        if (i == 138 && j == 186) {
          printf("peng(%i %i) = %e\n", i, j, diag(0,k,j,i));
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  auto *pmb = rc->GetParentPointer().get();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  auto c2p = pkg->Param<c2p_type<T>>("c2p_func");
  return c2p(rc, ib, jb, kb);
}

template <typename T>
TaskStatus ConservedToPrimitiveVanDerHolst(T *rc, const IndexRange &ib, const IndexRange &jb,
                                           const IndexRange &kb) {
  printf("C2P...\n");
  using namespace con2prim_vanderholst;
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = ConToPrimSetup(rc, c2p_tol, c2p_max_iter);

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto coords = pmb->coords;

  auto fail = rc->Get(internal_variables::fail).data;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto status = invert.Solve(geom, eos, k, j, i);
        fail(k, j, i) = (status == ConToPrimStatus::success ? FailFlags::success
                                                            : FailFlags::fail);
      });

  /*std::vector<std::string> diag_var({fluid_cons::density});
  IndexRange iib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kkb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto diag = rc->PackVariables(diag_var);
  ParArrayND<Real> myflux("ughhh", iib.e - iib.s + 3);
  printf("divF:\n");
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    0, 0, 127+parthenon::Globals::nghost, 127+parthenon::Globals::nghost,
      iib.s - 1, iib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        myflux(i - (iib.s - 1)) = diag(0,k,j,i);
      });
  pmb->exec_space.fence();
  auto flux_h = myflux.GetHostMirrorAndCopy();
  for (int i = 0; i < iib.e - iib.s + 3; i++) {
    //printf("flux(%i) = %e\n", i, flux_h(i));
    printf("%e, ", flux_h(i));
    //if (i > 180 && i < 190) {
    //  printf("flux(%i) = %e\n", i, flux_h(i));
    //}
  }
  printf("%s:%i\n", __FILE__, __LINE__);*/
  pmb->exec_space.fence(); printf("C2P done!\n");

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitiveClassic(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *fix_pkg = pmb->packages.Get("fixup").get();
  auto bounds = fix_pkg->Param<fixup::Bounds>("bounds");

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim::ConToPrimSetup(rc, c2p_tol, c2p_max_iter);
  auto invert_robust = con2prim_robust::ConToPrimSetup(rc, bounds, c2p_tol, c2p_max_iter);

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  auto coords = pmb->coords;

  auto fail = rc->Get(internal_variables::fail).data;

  // breaking con2prim into 3 kernels seems more performant.  WHY?
  // if we can combine them, we can get rid of the mesh sized scratch array
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Setup", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        invert.Setup(geom, k, j, i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto status = invert(eos, k, j, i);
        /*if (status != ConToPrimStatus::success) {
          invert_robust(geom, eos, coords, k, j, i);
          status = ConToPrimStatus::success;
          if (j==256 && i == 440) {
            std::cout << "used robust solver on i=440, j=256" << std::endl;
          }
        }*/
        fail(k, j, i) = (status == ConToPrimStatus::success ? FailFlags::success
                                                            : FailFlags::fail);
      });
  // this is where we might stick fixup
  /*int fail_cnt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve", DevExecSpace(),
      0, invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    int &f) {
        f += (fail(k, j, i) == FailFlags::success ? 0 : 1);
      },
      Kokkos::Sum<int>(fail_cnt));
  PARTHENON_REQUIRE(fail_cnt == 0, "Con2Prim Failed!");*/
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Finalize", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        //invert.Finalize(eos, geom, k, j, i);
        if (!fail(k, j, i)) {
          auto status = invert.Finalize(eos, geom, k, j, i);
          fail(k, j, i) = (status == ConToPrimStatus::success ? FailFlags::success
                                                              : FailFlags::fail);
        }
      });

  return TaskStatus::complete;
}

// template <typename T>
TaskStatus CopyFluxDivergence(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> vars({fluid_cons::density, fluid_cons::momentum, fluid_cons::energy});
  PackIndexMap imap;
  auto divf = rc->PackVariables(vars, imap);
  const int crho = imap[fluid_cons::density].first;
  const int cmom_lo = imap[fluid_cons::momentum].first;
  const int cmom_hi = imap[fluid_cons::momentum].second;
  const int ceng = imap[fluid_cons::energy].first;
  std::vector<std::string> diag_vars({"flux_divergence"});
  auto diag = rc->PackVariables(diag_vars);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CopyDivF", DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (i == 121 && j == 145) {
          printf("divF: %e %e %e %e %e\n",
            divf(crho,k,j,i), divf(cmom_lo,k,j,i), divf(cmom_lo+1,k,j,i),
            divf(cmom_lo+2,k,j,i), divf(ceng,k,j,i));
        }
        diag(0,k,j,i) = divf(crho,k,j,i);
        diag(1,k,j,i) = divf(cmom_lo,k,j,i);
        diag(2,k,j,i) = divf(cmom_lo+1,k,j,i);
        diag(3,k,j,i) = divf(cmom_lo+2,k,j,i);
        diag(4,k,j,i) = divf(ceng,k,j,i);
      }
  );

  /*ParArrayND<Real> myflux("ughhh", ib.e - ib.s + 3);
  printf("divF:\n");
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    0, 0, 127+parthenon::Globals::nghost, 127+parthenon::Globals::nghost,
      ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        myflux(i - (ib.s - 1)) = divf(crho,k,j,i);
      });
  pmb->exec_space.fence();
  auto flux_h = myflux.GetHostMirrorAndCopy();
  for (int i = 0; i < ib.e - ib.s + 3; i++) {
    //printf("flux(%i) = %e\n", i, flux_h(i));
    printf("%e, ", flux_h(i));
    //if (i > 180 && i < 190) {
    //  printf("flux(%i) = %e\n", i, flux_h(i));
    //}
  }
  printf("\n");
  exit(-1);*/
  return TaskStatus::complete;
}


// template <typename T>
TaskStatus CalculateFluidSourceTerms(MeshBlockData<Real> *rc,
                                     MeshBlockData<Real> *rc_src) {
  printf("CALCULATE SOURCE...\n");
  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fluid")->Param<bool>("hydro"))
    return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> vars({fluid_cons::momentum, fluid_cons::energy});
  PackIndexMap imap;
  auto src = rc_src->PackVariables(vars, imap);
  const int cmom_lo = imap[fluid_cons::momentum].first;
  const int cmom_hi = imap[fluid_cons::momentum].second;
  const int ceng = imap[fluid_cons::energy].first;
  std::vector<std::string> diag_vars({"src_terms"});
  auto diag = rc->PackVariables(diag_vars);

  auto tmunu = BuildStressEnergyTensor(rc);
  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TmunuSourceTerms", DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real Tmunu[ND][ND], gam[ND][ND][ND];
        tmunu(Tmunu, k, j, i);
        geom.ConnectionCoefficient(CellLocation::Cent, k, j, i, gam);
	      Real gdet = geom.DetG(CellLocation::Cent, k, j, i);
        diag(0,k,j,i) = 0.0;
        // momentum source terms
        for (int l = 0; l < NS; l++) {
          Real src_mom = 0.0;
          for (int m = 0; m < ND; m++) {
            for (int n = 0; n < ND; n++) {
              // gam is ALL INDICES DOWN
              src_mom -= Tmunu[m][n] * gam[l+1][n][m];
            }
          }
	        src(cmom_lo + l, k, j, i) = gdet*src_mom;
        }

        { // energy source term
          // TODO(jcd): maybe use the lapse and shift here instead of gcon
          Real gcon[4][4];
          geom.SpacetimeMetricInverse(CellLocation::Cent, k, j, i, gcon);
          Real TGam = 0.0;
          for (int m = 0; m < ND; m++) {
            for (int n = 0; n < ND; n++) {
              Real gam0 = 0;
              for (int r = 0; r < ND; r++) {
                gam0 += gcon[0][r] * gam[r][m][n];
              }
              TGam += Tmunu[m][n] * gam0;
            }
          }
          Real Ta = 0.0;
          Real da[ND];
          //Real *da = &gam[1][0][0];
          geom.GradLnAlpha(CellLocation::Cent, k, j, i, da);
          for (int m = 0; m < ND; m++) {
            Ta += Tmunu[m][0] * da[m];
          }
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          src(ceng, k, j, i) = gdet * alpha * (Ta - TGam);
          diag(4,k,j,i) = src(ceng,k,j,i);
          diag(5,k,j,i) = gdet*alpha*Ta;
          diag(6,k,j,i) = -gdet*alpha*TGam;
          //std::cerr << Ta << " " << TGam << std::endl;
        }

        // re-use gam for metric derivative
        geom.MetricDerivative(CellLocation::Cent, k, j, i, gam);
        for (int l = 0; l < NS; l++) {
          Real src_mom = 0.0;
          for (int m = 0; m < ND; m++) {
            for (int n = 0; n < ND; n++) {
              src_mom += Tmunu[m][n] * gam[n][l+1][m];
            }
          }
          src(cmom_lo + l, k, j, i) += gdet*src_mom;
          diag(l+1, k, j, i) = src(cmom_lo+l,k,j,i);
        }

        /*if (i == 121 && j == 145) {
          printf("src: %e %e %e %e %e\n", 0., src(cmom_lo,k,j,i),
            src(cmom_lo+1,k,j,i), src(cmom_lo+2,k,j,i), src(ceng,k,j,i));
        }*/

        // TODO(BRR) break code
        /*src(ceng, k, j, i) = 0.;
        SPACELOOP(ii) {
          src(cmom_lo+ii, k, j, i) = 0.;
        }*/
      });

  /*{
  pmb->exec_space.fence();
  std::vector<std::string> diagpack_vars({fluid_prim::density, fluid_prim::velocity, fluid_prim::energy,
  fluid_cons::density, fluid_cons::momentum, fluid_cons::energy});
  PackIndexMap dimap;
  auto diagpack = rc->PackVariables(diagpack_vars, dimap);
  const int prho = dimap[fluid_prim::density].first;
  const int pvel_lo = dimap[fluid_prim::velocity].first;
  const int peng = dimap[fluid_prim::energy].first;
  const int crho = dimap[fluid_cons::density].first;
  const int cmom_lo = dimap[fluid_cons::momentum].first;
  const int ceng = dimap[fluid_cons::energy].first;
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    0, 0,
      KOKKOS_LAMBDA(const int n) {
        int k = 0;
        int j = 186;
        int i = 138;
        printf("src prim: %e %e %e %e %e\n",
        diagpack(prho,k,j,i), diagpack(pvel_lo,k,j,i),
        diagpack(pvel_lo+1,k,j,i), diagpack(pvel_lo+2,k,j,i),
        diagpack(peng, k,j,i));
        printf("src cons: %e %e %e %e %e\n",
        diagpack(crho,k,j,i), diagpack(cmom_lo,k,j,i),
        diagpack(cmom_lo+1,k,j,i), diagpack(cmom_lo+2,k,j,i),
        diagpack(ceng,k,j,i));
      });
  pmb->exec_space.fence();
  }*/

  return TaskStatus::complete;
}

// template <typename T>
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fluid")->Param<bool>("hydro"))
    return TaskStatus::complete;

  auto flux = riemann::FluxState(rc);
  auto sig = rc->Get(internal_variables::face_signal_speed).data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int dk = (pmb->pmy_mesh->ndim == 3 ? 1 : 0);
  const int dj = (pmb->pmy_mesh->ndim > 1 ? 1 : 0);
  const int nrecon = flux.ql.GetDim(4) - 1;
  auto rt = pmb->packages.Get("fluid")->Param<PhoebusReconstruction::ReconType>(
      "Recon");
  auto st = pmb->packages.Get("fluid")->Param<riemann::solver>("RiemannSolver");

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    X1DIR, pmb->pmy_mesh->ndim, 0, nrecon, kb.s - dk, kb.e + dk, jb.s - dj, jb.s + dj,
      ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int d, const int n, const int k, const int j, const int i) {
        PARTHENON_REQUIRE(!isnan(flux.v(n,k,j,i)), "bad prim for recon");
        //flux.ql(d-1,n,k,j,i)
      });
  pmb->exec_space.fence();

#define RECON(method)                                                          \
  parthenon::par_for(                                                          \
      DEFAULT_LOOP_PATTERN, "Reconstruct", DevExecSpace(), X1DIR,              \
      pmb->pmy_mesh->ndim, 0, nrecon, kb.s - dk, kb.e + dk, jb.s - dj,         \
      jb.e + dj, ib.s - 1, ib.e + 1,                                           \
      KOKKOS_LAMBDA(const int d, const int n, const int k, const int j,        \
                    const int i) {                                             \
        method(d, n, k, j, i, flux.v, flux.ql, flux.qr);                       \
      });
  switch (rt) {
  case PhoebusReconstruction::ReconType::weno5z:
    RECON(PhoebusReconstruction::WENO5Z);
    break;
  case PhoebusReconstruction::ReconType::weno5a:
    RECON(PhoebusReconstruction::WENO5A);
    break;
  case PhoebusReconstruction::ReconType::mp5:
    RECON(PhoebusReconstruction::MP5);
    break;
  case PhoebusReconstruction::ReconType::linear:
    RECON(PhoebusReconstruction::PiecewiseLinear);
    break;
  default:
    PARTHENON_THROW("Invalid recon option.");
  }
#undef RECON

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    X1DIR, pmb->pmy_mesh->ndim, 0, nrecon, kb.s - dk, kb.e + dk, jb.s - dj, jb.s + dj,
      ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int d, const int n, const int k, const int j, const int i) {
        PARTHENON_REQUIRE(!isnan(flux.ql(d-1,n,k,j,i)), "bad ql after recon");
        PARTHENON_REQUIRE(!isnan(flux.qr(d-1,n,k,j,i)), "bad qr after recon");
        //flux.ql(d-1,n,k,j,i)
      });
  pmb->exec_space.fence();

#define FLUX(method)                                                           \
  parthenon::par_for(                                                          \
      DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(), X1DIR,          \
      pmb->pmy_mesh->ndim, kb.s - dk, kb.e + dk, jb.s - dj, jb.e + dj,         \
      ib.s - 1, ib.e + 1,                                                      \
      KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {      \
        sig(d - 1, k, j, i) = method(flux, d, k, j, i);                        \
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

  /*printf("a\n");
  ParArrayND<Real> myflux("ughhh", ib.e - ib.s + 3);
  printf("a\n");
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "test", DevExecSpace(),
    X1DIR, X1DIR, 0, nrecon, 0, 0, 127+parthenon::Globals::nghost, 127+parthenon::Globals::nghost,
      ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int d, const int n, const int k, const int j, const int i) {
        myflux(i - (ib.s - 1)) = flux.v.flux(X1DIR, 0, k, j, i);
      });
  printf("a\n");
  pmb->exec_space.fence();
  printf("a\n");
  auto flux_h = myflux.GetHostMirrorAndCopy();
  printf("a\n");
  for (int i = 0; i < ib.e - ib.s + 3; i++) {
    //printf("flux(%i) = %e\n", i, flux_h(i));
    //printf("%e, ", flux_h(i));
    if (i > 180 && i < 190) {
      printf("flux(%i) = %e\n", i, flux_h(i));
    }
  }
  printf("\n");
  auto rv = flux.ReconVars();
  for (auto &v : rv) {
    printf("  %s\n", v.c_str());
  }*/
  //exit(-1);


  return TaskStatus::complete;
}

TaskStatus FluxCT(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd"))
    return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;
  if (ndim == 1)
    return TaskStatus::complete;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto f1 = rc->Get(fluid_cons::bfield).flux[X1DIR];
  auto f2 = rc->Get(fluid_cons::bfield).flux[X2DIR];
  auto f3 = rc->Get(fluid_cons::bfield).flux[X3DIR];
  auto emf = rc->Get(internal_variables::emf).data;

  if (ndim == 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::EMF::2D", DevExecSpace(), kb.s, kb.e,
        jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          emf(k, j, i) = 0.25 * (f1(1, k, j, i) + f1(1, k, j - 1, i) -
                                 f2(0, k, j, i) - f2(0, k, j, i - 1));
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::Flux::2D", DevExecSpace(), kb.s, kb.e,
        jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          f1(0, k, j, i) = 0.0;
          f1(1, k, j, i) = 0.5 * (emf(k, j, i) + emf(k, j + 1, i));
          f2(0, k, j, i) = -0.5 * (emf(k, j, i) + emf(k, j, i + 1));
          f2(1, k, j, i) = 0.0;
        });
  } else if (ndim == 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::EMF::3D", DevExecSpace(), kb.s, kb.e + 1,
        jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          emf(0, k, j, i) = 0.25 * (f2(2, k, j, i) + f2(2, k - 1, j, i) -
                                    f3(1, k, j, i) - f3(1, k, j - 1, i));
          emf(1, k, j, i) = -0.25 * (f1(2, k, j, i) + f1(2, k - 1, j, i) -
                                     f3(0, k, j, i) - f3(0, k, j, i - 1));
          emf(2, k, j, i) = 0.25 * (f1(1, k, j, i) + f1(1, k, j - 1, i) -
                                    f2(0, k, j, i) - f2(0, k, j, i - 1));
        });
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxCT::Flux::3D", DevExecSpace(), kb.s,
        kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
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
  auto pmb = rc->GetBlockPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd"))
    return TaskStatus::complete;

  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  auto b = rc->Get(fluid_cons::bfield).data;
  auto divb = rc->Get(diagnostic_variables::divb).data;
  if (ndim == 2) {
    // todo(jcd): these are supposed to be node centered, and this misses the
    // high boundaries
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DivB::2D", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          divb(k, j, i) = 0.5 *
                              (b(0, k, j, i) + b(0, k, j - 1, i) -
                               b(0, k, j, i - 1) - b(0, k, j - 1, i - 1)) /
                              coords.Dx(X1DIR, k, j, i) +
                          0.5 *
                              (b(1, k, j, i) + b(1, k, j, i - 1) -
                               b(1, k, j - 1, i) - b(1, k, j - 1, i - 1)) /
                              coords.Dx(X2DIR, k, j, i);
        });
  } else if (ndim == 3) {
    // todo(jcd): these are supposed to be node centered, and this misses the
    // high boundaries
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DivB::3D", DevExecSpace(), kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          divb(k, j, i) =
              0.25 *
                  (b(0, k, j, i) + b(0, k, j - 1, i) + b(0, k - 1, j, i) +
                   b(0, k - 1, j - 1, i) - b(0, k, j, i - 1) -
                   b(0, k, j - 1, i - 1) - b(0, k - 1, j, i - 1) -
                   b(0, k - 1, j - 1, i - 1)) /
                  coords.Dx(X1DIR, k, j, i) +
              0.25 *
                  (b(1, k, j, i) + b(1, k, j, i - 1) + b(1, k - 1, j, i) +
                   b(1, k - 1, j, i - 1) - b(1, k, j - 1, i) -
                   b(1, k, j - 1, i - 1) - b(1, k - 1, j - 1, i) -
                   b(1, k - 1, j - 1, i - 1)) /
                  coords.Dx(X2DIR, k, j, i) +
              0.25 *
                  (b(2, k, j, i) + b(2, k, j, i - 1) + b(2, k, j - 1, i) +
                   b(2, k, j - 1, i - 1) - b(2, k - 1, j, i) -
                   b(2, k - 1, j, i - 1) - b(2, k - 1, j - 1, i) -
                   b(2, k - 1, j - 1, i - 1)) /
                  coords.Dx(X3DIR, k, j, i);
        });
  }
  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  auto &pars = pmb->packages.Get("fluid")->AllParams();
  Real min_dt;
  auto csig = rc->Get(internal_variables::cell_signal_speed).data;
  if (pars.hasKey("has_face_speeds")) {
    auto fsig = rc->Get(internal_variables::face_signal_speed).data;
    pmb->par_reduce(
        "Hydro::EstimateTimestep::1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
          Real ldt = 0.0;
          for (int d = 0; d < ndim; d++) {
            const int di = (d == 0);
            const int dj = (d == 1);
            const int dk = (d == 2);
            const Real max_s = std::max(
                csig(d, k, j, i),
                std::max(fsig(d, k, j, i), fsig(d, k + dk, j + dj, i + di)));
            ldt += max_s / coords.Dx(X1DIR + d, k, j, i);
          }
          lmin_dt = std::min(lmin_dt, 1.0 / ldt);
        },
        Kokkos::Min<Real>(min_dt));
  } else {
    pmb->par_reduce(
        "Hydro::EstimateTimestep::0", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
          Real ldt = 0.0;
          for (int d = 0; d < ndim; d++) {
            ldt += csig(d, k, j, i) / coords.Dx(X1DIR + d, k, j, i);
          }
          lmin_dt = std::min(lmin_dt, 1.0 / ldt);
        },
        Kokkos::Min<Real>(min_dt));
    pars.Add("has_face_speeds", true);
  }
  const auto &cfl = pars.Get<Real>("cfl");
  return cfl * min_dt;
}


} // namespace fluid
