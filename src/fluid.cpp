#include "fluid.hpp"
#include "con2prim.hpp"
#include "reconstruction.hpp"

#include <kokkos_abstraction.hpp>


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("Fluid");

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
  physics->AddField("gamma1", mprim_scalar);

  // add the conserved variables
  physics->AddField("c.density", mcons_scalar);
  physics->AddField("c.momentum", mcons_threev);
  physics->AddField("c.energy", mcons_scalar);
  //physics->AddField("c.bfield", mcons_threev);

  Metadata mrecon = Metadata({Metadata::Face, Metadata::OneCopy}, std::vector<int>(1, 7));
  physics->AddField("ql", mrecon);
  physics->AddField("qr", mrecon);

  return physics;
}

//template <typename T>
TaskStatus PrimitiveToConserved(MeshData<Real> *rc) {
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure"});

  PackIndexMap imap
  auto &v = rc->PackVariables(vars, imap);

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

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      //todo(jcd): make these real
      Real gcov[3][3];
      geometry::gamma_cov(geometry::center, v.coords(b), k, j, i, gcov);
      Real gdet = sqrt(geometry::gamma_det(geometry::center, v.coords(b), k, j, i));
      Real vsq = 0.0;
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          vsq += gcov[m][n] * v(b,pvel_lo+m, k, j, i) * v(b, pvel_lo+n, k, j, i);
        }
      }
      Real W = 1.0/sqrt(1.0 - vsq);
      // conserved density D = \sqrt{\gamma} \rho W
      v(b, crho, k, j, i) = gdet * v(b, prho, k, j, i) * W;

      Real rhoh = v(b, prho, k, j, i) + v(b, peng, k, j, i) + v(b, prs, k, j, i);
      for (int m = 0; m < 3; m++) {
        vcov = 0.0;
        for (int n = 0; n < 3; n++) {
          vcov += gcov[m][n]*v(b, pvel_lo+n, k, j, i);
        }
        v(b, cmom_lo+m, k, j, i) = gdet*rhoh*W*W*vcov;
      }

      v(b, ceng, k, j, i) = gdet*(rhoh*W*W - v(b, prs, k, j, i)) - v(b, crho, k, j, i);
    });

  return TaskStatus::complete;
}

//template <typename T>
TaskStatus ConservedToPrimitive(MeshData<Real> *rc) {
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure", "temperature"});

  auto &eos = pmb->packages["EOS"]->Param<singularity::EOS>("d.d.EOS");

  PackIndexMap imap
  auto &v = rc->PackVariables(vars, imap);
  auto &geom = 
  ConToPrim invert(v, imap, eos, geom);

  const int ifail;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  int fail_cnt;
  parthenon::par_reduce(DEFAULT_LOOP_PATTERN, "ConToPrim", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &fail) {
      auto status = invert(b, k, j, i);
      if (status == ConToPrimStatus::failure) fail++;
    }, Kokkos::Sum<int>(fail_cnt));
  return TaskStatus::complete;
}

//template <typename T>
TaskStatus CalculateFluxes(MeshData<Real> *rc) {
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "p.velocity", "p.energy", "pressure"});
  std::vector<std::string> flxs({"c.density", "c.momentum", "c.energy"});

  PackIndexMap imap;
  auto &v = rc->PackVariablesAndFluxes(vars, flxs, imap);

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {

    });

  return TaskStatus::complete;
}
