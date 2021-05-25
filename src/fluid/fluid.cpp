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
#include "reconstruction.hpp"
#include "tmunu.hpp"

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

  Real cfl = pin->GetOrAddReal("fluid", "cfl", 0.8);
  params.Add("cfl", cfl);

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

  Metadata mprim_threev =
      Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector,
                Metadata::Derived, Metadata::OneCopy},
               three_vec);
  Metadata mprim_scalar = Metadata({Metadata::Cell, Metadata::Intensive,
                                    Metadata::Derived, Metadata::OneCopy});
  Metadata mcons_scalar =
      Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::FillGhost});
  Metadata mcons_threev =
      Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
                Metadata::Conserved, Metadata::Vector, Metadata::FillGhost},
               three_vec);

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
  Metadata mrecon = Metadata({Metadata::Cell, Metadata::OneCopy}, recon_shape);
  physics->AddField(impl::ql, mrecon);
  physics->AddField(impl::qr, mrecon);

  std::vector<int> signal_shape(1, ndim);
  Metadata msignal =
      Metadata({Metadata::Cell, Metadata::OneCopy}, signal_shape);
  physics->AddField(impl::face_signal_speed, msignal);
  physics->AddField(impl::cell_signal_speed, msignal);

  std::vector<int> c2p_scratch_size(1, 5);
  Metadata c2p_meta =
      Metadata({Metadata::Cell, Metadata::OneCopy}, c2p_scratch_size);
  physics->AddField(impl::c2p_scratch, c2p_meta);

  physics->FillDerivedBlock = ConservedToPrimitive<MeshBlockData<Real>>;
  physics->EstimateTimestepBlock = EstimateTimestepBlock;

  return physics;
}

// template <typename T>
TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({p::density, c::density, p::velocity,
                                 c::momentum, p::energy, c::energy, p::bfield,
                                 c::bfield, p::ye, c::ye, p::pressure});

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
  const int pb_lo = imap[p::bfield].first;
  const int pb_hi = imap[p::bfield].second;
  const int cb_lo = imap[c::bfield].first;
  const int cb_hi = imap[c::bfield].second;
  int pye = imap[p::ye].second; // -1 if not present
  int cye = imap[c::ye].second;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real gcov[3][3];
        geom.Metric(CellLocation::Cent, k, j, i, gcov);
        Real gcov4[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov4);
        Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
        Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
        Real shift[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, shift);
        Real vsq = 0.0;
        Real Bdotv = 0.0;
        Real BdotB = 0.0;
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 3; n++) {
            vsq += gcov[m][n] * v(b, pvel_lo + m, k, j, i) *
                   v(b, pvel_lo + n, k, j, i);
          }
          for (int n = pb_lo; n <= pb_hi; n++) {
            Bdotv += gcov[m][n - pb_lo] * v(b, pvel_lo + m, k, j, i) *
                     v(b, n, k, j, i);
            BdotB += gcov[m][n - pb_lo] * v(b, pb_lo + m, k, j, i) *
                     v(b, n, k, j, i);
          }
        }

        // Lorentz factor
        const Real W = 1.0 / sqrt(1.0 - vsq);

        // get the magnetic field 4-vector
        Real bcon[] = {W * Bdotv / lapse, 0.0, 0.0, 0.0};
        for (int m = pb_lo; m <= pb_hi; m++) {
          bcon[m - pb_lo + 1] =
              v(b, m, k, j, i) / W + lapse * bcon[0] *
                                         (v(b, m - pb_lo + pvel_lo, k, j, i) -
                                          shift[m - pb_lo] / lapse);
        }
        const Real bsq = (BdotB + lapse * lapse * bcon[0] * bcon[0]) / (W * W);
        Real bcov[3] = {0.0, 0.0, 0.0};
        if (pb_hi > 0) {
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 4; n++) {
              bcov[m] += gcov4[m + 1][n] * bcon[n];
            }
          }
        }

        // conserved density D = \sqrt{\gamma} \rho W
        v(b, crho, k, j, i) = gdet * v(b, prho, k, j, i) * W;

        // enthalpy
        Real rhohWsq = (v(b, prho, k, j, i) + v(b, peng, k, j, i) +
                        v(b, prs, k, j, i) + bsq) *
                       W * W;

        // momentum
        for (int m = 0; m < 3; m++) {
          Real vcov = 0.0;
          for (int n = 0; n < 3; n++) {
            vcov += gcov[m][n] * v(b, pvel_lo + n, k, j, i);
          }
          v(b, cmom_lo + m, k, j, i) =
              gdet * rhohWsq * vcov - lapse * bcon[0] * bcov[m];
        }

        // energy
        v(b, ceng, k, j, i) =
            gdet * (rhohWsq - (v(b, prs, k, j, i) + 0.5 * bsq) -
                    lapse * lapse * bcon[0] * bcon[0]) -
            v(b, crho, k, j, i);

        for (int m = cb_lo; m <= cb_hi; m++) {
          v(b, m, k, j, i) = gdet * v(b, m - cb_lo + pb_lo, k, j, i);
        }
        // conserved lepton density
        if (pye > 0) {
          v(b, cye, k, j, i) = v(b, crho, k, j, i) * v(b, pye, k, j, i);
        }
      });

  return TaskStatus::complete;
}

template <typename T> TaskStatus ConservedToPrimitive(T *rc) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim::ConToPrimSetup(rc, c2p_tol, c2p_max_iter);

  StateDescriptor *eos_pkg = pmb->packages.Get("eos").get();
  auto eos = eos_pkg->Param<singularity::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);

  auto fail = rc->Get(internal_variables::fail).data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

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
        fail(k, j, i) = (status == ConToPrimStatus::success ? FailFlags::success
                                                            : FailFlags::fail);
      });
  // this is where we might stick fixup
  int fail_cnt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve", DevExecSpace(),
      0, invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    int &f) {
        f += (fail(k, j, i) == FailFlags::success ? 0 : 1);
      },
      Kokkos::Sum<int>(fail_cnt));
  PARTHENON_REQUIRE(fail_cnt == 0, "Con2Prim Failed!");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Finalize", DevExecSpace(), 0,
      invert.NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        invert.Finalize(eos, geom, k, j, i);
      });

  return TaskStatus::complete;
}

// template <typename T>
TaskStatus CalculateFluidSourceTerms(MeshBlockData<Real> *rc,
                                     MeshBlockData<Real> *rc_src) {
  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> vars({fluid_cons::momentum, fluid_cons::energy});
  PackIndexMap imap;
  auto src = rc_src->PackVariables(vars, imap);
  const int cmom_lo = imap[fluid_cons::momentum].first;
  const int cmom_hi = imap[fluid_cons::momentum].second;
  const int ceng = imap[fluid_cons::energy].first;

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
        }

/*        if ((i == 64 && j == 64) || (i == 16 && j == 16)) {
//        {
          printf("Sources: %e %e %e %e %e\n",
            0., src(ceng, k, j, i), src(cmom_lo, k, j, i),
            src(cmom_lo+1, k, j, i), src(cmom_lo+2, k, j, i));

          //Real Wik_UU[3][3] = {0};
          Real dg_DDD[4][4][4];
          geom.MetricDerivative(CellLocation::Cent, k, j, i, dg_DDD);
          Real NewSMom[3] = {0};

          // Now try to calculate via Porth:
          Real ncon[4] = {1, 0, 0, 0};
          Real ncov[4] = {-1, 0, 0, 0};
          Real delta[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
          Real proj[4][4];
          Real Wij[4][4];
          SPACETIMELOOP2(mu, nu) {
            Wij[mu][nu] = 0.;
            proj[mu][nu] = delta[mu][nu] + ncon[mu]*ncov[nu];
          }
          SPACETIMELOOP(mu) SPACETIMELOOP(nu) SPACETIMELOOP(kap) SPACETIMELOOP(lam) {
            Wij[mu][nu] += proj[mu][kap]*proj[nu][lam]*Tmunu[kap][lam];
          }
//          SPACETIMELOOP2(mu, nu) {
//            printf("Wij[%i][%i] = %e\n", mu, nu, Wij[mu][nu]);
//          }

          for (int jj = 0; jj < 3; jj++) {
            for (int ii = 0; ii < 3; ii++) {
              for (int kk = 0; kk < 3; kk++) {
                //NewSMom[jj] += 0.5*Wij[ii+1][kk+1]*dg_DDD[jj+1][ii+1][kk+1];
                NewSMom[jj] += 0.5*Wij[ii+1][kk+1]*dg_DDD[kk+1][ii+1][jj+1];
              }
            }
            printf("NewSMom[%i] = %e\n", jj, NewSMom[jj]);
            src(cmom_lo + jj, k, j, i) = NewSMom[jj];
          }
        }
          printf("Tmunu:\n");
          for (int n = 0; n < 4; n++) {
            printf("%e %e %e %e\n", Tmunu[n][0], Tmunu[n][1], Tmunu[n][2], Tmunu[n][3]);
          }
          printf("\n");
          printf("Sources:\n");
          printf("%e %e %e %e %e\n",
            0.,
            src(ceng,k,j,i),
            src(cmom_lo,k,j,i),
            src(cmom_lo+1,k,j,i),
            src(cmom_lo+2,k,j,i));

          // Now try to calculate via Porth:
          Real ncon[4] = {1, 0, 0, 0};
          Real ncov[4] = {-1, 0, 0, 0};
          Real delta[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
          Real proj[4][4];
          Real Wij[4][4];
          SPACETIMELOOP2(mu, nu) {
            Wij[mu][nu] = 0.;
            proj[mu][nu] = delta[mu][nu] + ncon[mu]*ncov[nu];
          }
          SPACETIMELOOP(mu) SPACETIMELOOP(nu) SPACETIMELOOP(kap) SPACETIMELOOP(lam) {
            Wij[mu][nu] += proj[mu][kap]*proj[nu][lam]*Tmunu[kap][lam];
          }
          Real ceng_src = 0.;
          for (int ii = 1; ii < 4; ii++) {
            for (int jj = 1; jj < 4; jj++) {
              for (int kk = 1; kk < 4; kk++) {
                ceng_src += 0.5*Wij[ii][kk]*gam[jj][ii][kk];
              }
            }
          }
          printf("ceng_src: %e\n", ceng_src);
          exit(-1);
        }*/
      });
  return TaskStatus::complete;
}

// template <typename T>
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();

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
