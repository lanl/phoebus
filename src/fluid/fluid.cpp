#include "fluid.hpp"
#include "reconstruction.hpp"

#include <kokkos_abstraction.hpp>


namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("fluid");
  Params &params = physics->AllParams();

  Real cfl = pin->GetOrAddReal("fluid", "cfl", 0.8);
  params.Add("cfl", cfl);

  Real c2p_tol = pin->GetOrAddReal("fluid", "c2p_tol", 1.e-8);
  params.Add("c2p_tol", c2p_tol);

  int c2p_max_iter = pin->GetOrAddInteger("fluid", "c2p_max_iter", 20);
  params.Add("c2p_max_iter", c2p_max_iter);

  Metadata m;
  std::vector<int> three_vec(1,3);

  Metadata mprim_threev = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Vector, Metadata::Derived, Metadata::OneCopy}, three_vec);
  Metadata mprim_scalar = Metadata({Metadata::Cell, Metadata::Intensive, Metadata::Derived, Metadata::OneCopy});
  Metadata mcons_scalar = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive, Metadata::Conserved, Metadata::FillGhost});
  Metadata mcons_threev = Metadata({Metadata::Cell, Metadata::Independent, Metadata::Intensive, Metadata::Conserved, Metadata::Vector, Metadata::FillGhost}, three_vec);

  // add the primitive variables
  physics->AddField("p.density", mprim_scalar);
  physics->AddField("p.velocity", mprim_threev);
  physics->AddField("p.energy", mprim_scalar);
  //physics->AddField("p.bfield", mprim_threev);
  physics->AddField("pressure", mprim_scalar);
  physics->AddField("temperature", mprim_scalar);
  physics->AddField("gamma1", mprim_scalar);
  physics->AddField("cs", mprim_scalar);

  // add the conserved variables
  physics->AddField("c.density", mcons_scalar);
  physics->AddField("c.momentum", mcons_threev);
  physics->AddField("c.energy", mcons_scalar);
  //physics->AddField("c.bfield", mcons_threev);

  //Metadata mrecon = Metadata({Metadata::Face, Metadata::OneCopy}, std::vector<int>(1, 7));
  //physics->AddField("ql", mrecon);
  //physics->AddField("qr", mrecon);

  physics->FillDerivedBlock = ConservedToPrimitive<MeshBlockData<Real>>;
  physics->EstimateTimestepBlock = EstimateTimestepBlock;

  return physics;
}

//template <typename T>
TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure"});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);

  const int prho = imap["p.density"].first;
  const int crho = imap["c.density"].first;
  const int pvel_lo = imap["p.velocity"].first;
  const int pvel_hi = imap["p.velocity"].second;
  const int cmom_lo = imap["c.momentum"].first;
  const int cmom_hi = imap["c.momentum"].second;
  const int peng = imap["p.energy"].first;
  const int ceng = imap["c.energy"].first;
  const int prs = imap["pressure"].first;

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
      Real rhoh = v(b, prho, k, j, i) + v(b, peng, k, j, i) + v(b, prs, k, j, i);
      for (int m = 0; m < 3; m++) {
        Real vcov = 0.0;
        for (int n = 0; n < 3; n++) {
          vcov += gcov[m][n]*v(b, pvel_lo+n, k, j, i);
        }
        v(b, cmom_lo+m, k, j, i) = gdet*rhoh*W*W*vcov;
      }

      v(b, ceng, k, j, i) = gdet*(rhoh*W*W - v(b, prs, k, j, i)) - v(b, crho, k, j, i);
    });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer().get();

  auto invert = con2prim::ConToPrimSetup(rc);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

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

  std::vector<std::string> vars({"p.density", "p.velocity", "p.energy", "pressure", "gamma1"});
  std::vector<std::string> flxs({"c.density", "c.momentum", "c.energy"});

  PackIndexMap imap;
  auto v = rc->PackVariablesAndFluxes(vars, flxs, imap);
  const int recon_size = imap["gamma1"].first;

  auto geom = Geometry::GetCoordinateSystem(rc);

  auto ql = ParArrayND<Real>("ql", pmb->pmy_mesh->ndim, recon_size+1,
                        v.GetDim(3), v.GetDim(2), v.GetDim(1));
  auto qr = ParArrayND<Real>("qr", pmb->pmy_mesh->ndim, recon_size+1,
                        v.GetDim(3), v.GetDim(2), v.GetDim(1));

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int dk = (pmb->pmy_mesh->ndim == 3 ? 1 : 0);
  const int dj = (pmb->pmy_mesh->ndim >  1 ? 1 : 0);
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "Reconstruct", DevExecSpace(),
    X1DIR, pmb->pmy_mesh->ndim,
    kb.s-dk, kb.e+dk, jb.s-dj, jb.e+dj, ib.s-1, ib.e+1,
    KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {
      PhoebusReconstruction::PiecewiseLinear(d, 0, recon_size, k, j, i, v, ql, qr);
    });

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
    X1DIR, pmb->pmy_mesh->ndim,
    kb.s, kb.e+dk, jb.s, jb.e+dj, ib.s, ib.e+1,
    KOKKOS_LAMBDA(const int d, const int k, const int j, const int i) {
      llf(d, k, j, i, geom, ql, qr, v);
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
