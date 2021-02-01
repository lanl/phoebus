#include "fluid.hpp"
#include "reconstruction.hpp"

#include <kokkos_abstraction.hpp>


namespace fluid {

std::vector<std::string> FluxState::recon_vars;
std::vector<std::string> FluxState::flux_vars;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  auto physics = std::make_shared<StateDescriptor>("fluid");
  Params &params = physics->AllParams();

  Real cfl = pin->GetOrAddReal("fluid", "cfl", 0.8);
  params.Add("cfl", cfl);

  Real c2p_tol = pin->GetOrAddReal("fluid", "c2p_tol", 1.e-8);
  params.Add("c2p_tol", c2p_tol);

  int c2p_max_iter = pin->GetOrAddInteger("fluid", "c2p_max_iter", 20);
  params.Add("c2p_max_iter", c2p_max_iter);

  std::string recon = pin->GetOrAddString("fluid", "recon", "linear");
  PhoebusReconstruction::ReconType rt = PhoebusReconstruction::ReconType::linear;
  if (recon == "weno5") {
    rt = PhoebusReconstruction::ReconType::weno5;
  } else if (recon == "linear") {
    rt = PhoebusReconstruction::ReconType::linear;
  } else {
    PARTHENON_THROW("Invalid Reconstruction option.  Choose from [linear,weno5]");
  }
  params.Add("Recon", rt);

  std::string solver = pin->GetOrAddString("fluid", "riemann", "hll");
  RiemannSolver rs = RiemannSolver::HLL;
  if (solver == "llf") {
    rs = RiemannSolver::LLF;
  } else if (solver == "hll") {
    rs = RiemannSolver::HLL;
  } else {
    PARTHENON_THROW("Invalid Riemann Solver option. Choose from [llf, hll]");
  }
  params.Add("RiemannSolver", rs);

  bool ye = pin->GetOrAddBoolean("fluid", "Ye", false);
  params.Add("Ye", ye);

  Metadata m;
  std::vector<int> three_vec(1,3);

  Metadata mprim_threev = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector, Metadata::Derived, Metadata::OneCopy}, three_vec);
  Metadata mprim_scalar = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  Metadata mcons_scalar = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive, Metadata::Conserved, Metadata::FillGhost});
  Metadata mcons_threev = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive, Metadata::Conserved, Metadata::Vector, Metadata::FillGhost}, three_vec);

  // add the primitive variables
  physics->AddField(p::density, mprim_scalar);
  physics->AddField(p::velocity, mprim_threev);
  physics->AddField(p::energy, mprim_scalar);
  //physics->AddField("p.bfield", mprim_threev);
  physics->AddField(p::pressure, mprim_scalar);
  physics->AddField(p::temperature, mprim_scalar);
  physics->AddField(p::gamma1, mprim_scalar);
  physics->AddField(p::cs, mprim_scalar);
  if (ye) {
    physics->AddField(p::ye, mprim_scalar);
  }

  // add the conserved variables
  physics->AddField(c::density, mcons_scalar);
  physics->AddField(c::momentum, mcons_threev);
  physics->AddField(c::energy, mcons_scalar);
  //physics->AddField("c.bfield", mcons_threev);
  if (ye) {
    physics->AddField(c::ye, mcons_scalar);
  }

  // add the base state for reconstruction/fluxes
  std::vector<std::string> rvars({p::density,
                                  p::velocity,
                                  p::energy});
  FluxState::ReconVars(rvars);
  if (ye) FluxState::ReconVars(p::ye);

  std::vector<std::string> fvars({c::density,
                                  c::momentum,
                                  c::energy});
  FluxState::FluxVars(fvars);
  if (ye) FluxState::FluxVars(c::ye);

  // add some extra fields for reconstruction
  rvars = std::vector<std::string>({p::pressure,
                                    p::gamma1});
  FluxState::ReconVars(rvars);

  // set up the arrays for left and right states
  int ndim = 1;
  if (pin->GetInteger("parthenon/mesh", "nx3") > 1) ndim = 3;
  else if (pin->GetInteger("parthenon/mesh", "nx2") > 1) ndim = 2;

  auto recon_vars = FluxState::ReconVars();
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

  std::vector<int> recon_shape({nrecon,ndim});
  Metadata mrecon = Metadata({Metadata::Cell, Metadata::OneCopy}, recon_shape);
  physics->AddField("ql", mrecon);
  physics->AddField("qr", mrecon);

  physics->FillDerivedBlock = ConservedToPrimitive<MeshBlockData<Real>>;
  physics->EstimateTimestepBlock = EstimateTimestepBlock;

  return physics;
}

//template <typename T>
TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc) {
  namespace p = primitive_variables;
  namespace c = conserved_variables;
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({p::density, c::density,
                                 p::velocity, c::momentum,
                                 p::energy, c::energy,
                                 p::ye, c::ye,
                                 p::pressure});

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
  int pye = imap[p::ye].second; // -1 if not present
  int cye = imap[c::ye].second;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      //todo(jcd): make these real
      Real gcov[3][3];
      geom.Metric(CellLocation::Cent, k, j, i, gcov);
      Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);
      Real vsq = 0.0;
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          vsq += gcov[m][n] * v(b,pvel_lo+m, k, j, i) * v(b, pvel_lo+n, k, j, i);
        }
      }
      Real W = 1.0/sqrt(1.0 - vsq);
      // conserved density D = \sqrt{\gamma} \rho W
      v(b, crho, k, j, i) = gdet * v(b, prho, k, j, i) * W;

      // enthalpy
      Real rhohWsq = (v(b, prho, k, j, i) + v(b, peng, k, j, i) + v(b, prs, k, j, i))*W*W;
      for (int m = 0; m < 3; m++) {
        Real vcov = 0.0;
        for (int n = 0; n < 3; n++) {
          vcov += gcov[m][n]*v(b, pvel_lo+n, k, j, i);
        }
        v(b, cmom_lo+m, k, j, i) = gdet*rhohWsq*vcov;
      }

      v(b, ceng, k, j, i) = gdet*(rhohWsq - v(b, prs, k, j, i)) - v(b, crho, k, j, i);
      if (pye > 0) {
        v(b, cye, k, j, i) = v(b, crho, k, j, i) * v(b, pye, k, j, i);
      }
    });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *pkg = pmb->packages.Get("fluid").get();
  const Real c2p_tol = pkg->Param<Real>("c2p_tol");
  const int c2p_max_iter = pkg->Param<int>("c2p_max_iter");
  auto invert = con2prim::ConToPrimSetup(rc, c2p_tol, c2p_max_iter);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  //TODO(JCD): need to tag failed cells and get rid of this reduction
  int fail_cnt;
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "ConToPrim", DevExecSpace(),
    0, invert.NumBlocks()-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &fail) {
      auto status = invert(k, j, i);
      if (status == ConToPrimStatus::failure) fail++;
    }, Kokkos::Sum<int>(fail_cnt));

  PARTHENON_REQUIRE(fail_cnt==0, "Con2Prim Failed!");

  return TaskStatus::complete;
}

//template <typename T>
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();

  auto flux = FluxState(rc);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int dk = (pmb->pmy_mesh->ndim == 3 ? 1 : 0);
  const int dj = (pmb->pmy_mesh->ndim >  1 ? 1 : 0);
  const int nrecon = flux.ql.GetDim(4)-1;
  auto rt = pmb->packages.Get("fluid")->Param<PhoebusReconstruction::ReconType>("Recon");
  switch (rt) {
    case PhoebusReconstruction::ReconType::weno5:
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "Reconstruct", DevExecSpace(),
      X1DIR, pmb->pmy_mesh->ndim,
      kb.s-dk, kb.e+dk, jb.s-dj, jb.e+dj, ib.s-1, ib.e+1,
      KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {
        //PhoebusReconstruction::PiecewiseLinear(d, 0, nrecon, k, j, i, flux.v, flux.ql, flux.qr);
        PhoebusReconstruction::WENO5(d, 0, nrecon, k, j, i, flux.v, flux.ql, flux.qr);
        });
      break;
    case PhoebusReconstruction::ReconType::linear:
      parthenon::par_for(DEFAULT_LOOP_PATTERN, "Reconstruct", DevExecSpace(),
      X1DIR, pmb->pmy_mesh->ndim,
      kb.s-dk, kb.e+dk, jb.s-dj, jb.e+dj, ib.s-1, ib.e+1,
      KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {
        PhoebusReconstruction::PiecewiseLinear(d, 0, nrecon, k, j, i, flux.v, flux.ql, flux.qr);
        });
      break;
    default:
      PARTHENON_THROW("Invalid recon option.");
  }

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
    X1DIR, pmb->pmy_mesh->ndim,
    kb.s, kb.e+dk, jb.s, jb.e+dj, ib.s, ib.e+1,
    KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {
      flux.Solve(d, k, j, i);
    });

  return TaskStatus::complete;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &cs = rc->Get("cs").data;
  auto &v   = rc->Get("p.velocity").data;

  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  auto &phys = pmb->packages.Get("fluid");
  Real min_dt;
  pmb->par_reduce("Hydro::EstimateTimestep::0",
    kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
      const auto &c = cs(k,j,i);
      lmin_dt = std::min(lmin_dt,1.0/( (std::abs(v(0,k,j,i))+c)/coords.Dx(X1DIR,k,j,i)
                         + (ndim>1)*(std::abs(v(1,k,j,i))+c)/coords.Dx(X2DIR,k,j,i)
                         + (ndim>2)*(std::abs(v(2,k,j,i))+c)/coords.Dx(X3DIR,k,j,i)));
    }, Kokkos::Min<Real>(min_dt));
  const auto& cfl = phys->Param<Real>("cfl");
  return cfl*min_dt;
}

} // namespace fluid
