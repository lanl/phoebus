#include "fluid.hpp"
#include "con2prim.hpp"
#include "reconstruction.hpp"

#include <kokkos_abstraction.hpp>

namespace fluid {

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

template <typename T>
TaskStatus PrimitiveToConserved(T *rc) {
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

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      //todo(jcd): make these real
      Real gcov[3][3];
      geom.Metric(CellLocation::Cent, b, k, j, i, gcov);
      Real gdet = geom.DetGamma(CellLocation::Cent, b, k, j, i);
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

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure", "temperature"});

  auto &eos = pmb->packages["EOS"]->Param<singularity::EOS>("d.d.EOS");

  PackIndexMap imap
  auto &v = rc->PackVariables(vars, imap);
  auto geom = Geometry::GetCoordinateSystem(rc);
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

template <typename T>
TaskStatus CalculateFluxes(T *rc) {
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "p.velocity", "p.energy", "pressure", "gamma1"});
  std::vector<std::string> flxs({"c.density", "c.momentum", "c.energy"});

  PackIndexMap imap;
  auto &v = rc->PackVariablesAndFluxes(vars, flxs, imap);
  const int recon_size = imap["gamma1"].first;

  auto geom = Geometry::GetCoordinateSystem(rc);

  ql = ParArrayND<Real>("ql", v.GetDim(5), pmb->pmy_mesh->ndim, recon_size+1,
                        v.GetDim(3), v.GetDim(2), v.GetDim(1));
  qr = ParArrayND<Real>("qr", v.GetDim(5), pmb->pmy_mesh->ndim, recon_size+1,
                        v.GetDim(3), v.GetDim(2), v.GetDim(1));

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const int dk = (pmb->pmy_mesh->ndim == 3 ? 1 : 0);
  const int dj = (pmb->pmy_mesh->ndim >  1 ? 1 : 0)
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "Reconstruct", DevExecSpace(),
    0, v.GetDim(5)-1, X1DIR, pmb->pmy_mesh->ndim,
    kb.s-dk, kb.e+dk, jb.s-dj, jb.e+dj, ib.s-1, ib.e+1,
    KOKKOS_LAMBDA(const int b, const int d, const int k, const int j, const int i) {
      PhoebusReconstruction::PiecewiseLinear(b, d, 0, recon_size, k, j, i, v, ql, qr);
    }

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "CalculateFluxes", DevExecSpace(),
    0, v.GetDim(5)-1, X1DIR, pmb->pmy_mesh->ndim,
    kb.s, kb.e+dk, jb.s, jb.e+dj, ib.s, ib.e+1,
    KOKKOS_LAMBDA(const int b, const int d, const int k, const int j, const int i) {
      llf(b, d, k, j, i, geom, ql, qr, v);
    });

  return TaskStatus::complete;
}

#define DELTA(i,j) (i==j ? 1 : 0)

KOKKOS_FUNCTION
void prim_to_flux(const int b, const int d, const int k, const int j, const int i,
                  const Geometry::CoordinateSystem &geom, const ParArrayND<Real> &q,
                  Real &cs, Real *U, Real *F) {
  const int dir = d-1;
  const Real &rho = q(b,dir,0,k,j,i);
  const Real vcon[] = {q(b,dir,1,k,j,i), q(b,dir,2,k,j,i), q(b,dir,3,k,j,i)};
  const Real &v = vcon[dir];
  const Real &u = q(b,dir,4,k,j,i);
  const Real &P = q(b,dir,5,k,j,i);
  const Real &gm1 = q(b,dir,6,k,j,i);
  cs = sqrt(gm1*P/rho);

  CellLocation loc = DirectionToFaceID(d);

  Real vcov[3];
  for (int m = 0; m < 3; m++) {
    vcov[m] = 0.0;
    for (int n = 0; n < 3; n++) {
      vcov[m] += geom.Metric(m,n,loc,b,k,j,i)*vcon[n];
    }
  }

  // get Lorentz factor
  Real vsq = 0.0;
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      vsq += vcon[m]*vcov[n];
    }
  }
  Real W = 1.0/sqrt(1.0 - vsq);

  // conserved density
  U[0] = rhol*W;

  // conserved momentum
  Real H = rho + u + P;
  for (int m = 1; m <= 3; m++) {
    U[m] = H*W*W*vcov[m];
  }

  // conserved energy
  U[4] = H*W*W - P - U[0];

  // Get fluxes
  const Real alpha = geom.Lapse(loc, b, k, j, i);
  for (int m = 0; m < 3; m++) {
    //TODO(JCD): is the component of beta d or dir?
    const Real vtil = vcon[m] - geom.ContravariantShift(d, loc, k, j, i)/alpha;

    // mass flux
    F[0] = U[0]*vtil;

    // momentum flux
    for (int n = 1; n <= 3; n++) {
      F[n] = U[n]*vtil + P*DELTA(d,n);
    }

    // energy flux
    F[4] = U[4]*vtil + P*v;
  }

  return;
}

#undef DELTA

template <typename T>
KOKKOS_INLINE_FUNCTION
void llf(const int b, const int d, const int k, const int j, const int i,
         const Geometry::CoordinateSystem &geom, const ParArrayND<Real> &ql,
         const ParArrayND<Real> &qr, T &v) {

  Real Ul[5], Ur[5];
  Real Fl[5], Fr[5];
  Real cl, cr;

  prim_to_flux(b, d, k, j, i, geom, ql, cl, Ul, Fl);
  prim_to_flux(b, d, k, j, i, geom, qr, cr, Ur, Fr);

  const Real cmax = (cl > cr ? cl : cr);

  const Real gdet = geom.DetGamma(loc, b, k, j, i);
  for (int m = 0; m < 5; m++) {
    v.flux(b,d,0,k,j,i) = 0.5*(Fl[m] + Fr[m] - cmax*(Ur[m] - Ul[m])) * gdet;
  }
}


TaskStatus PrimitiveToConserved<MeshData<Real>>;
TaskStatus PrimitiveToConserved<MeshBlockData<Real>>;
TaskStatus ConservedToPrimitive<MeshData<Real>>;
TaskStatus ConservedToPrimitive<MeshBlockData<Real>>;
TaskStatus CalculateFluxes<MeshData<Real>>;
TaskStatus CalculateFluxes<MeshBlockData<Real>>;

} // namespace fluid
