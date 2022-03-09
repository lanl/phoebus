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


#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "radiation/radiation.hpp"
#include "reconstruction.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"

namespace radiation {

template <class T>
  class ReconstructionIndexer  {
 public:
  KOKKOS_INLINE_FUNCTION
  ReconstructionIndexer(const T& v, const int chunk_size, const int offset, const int block = 0) :
      v_(v), chunk_size_(chunk_size), offset_(offset), block_(block) {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real& operator()(const int idir, const int ivar, const int k, const int j, const int i) const {
    const int idx = idir*chunk_size_ + ivar + offset_;
    return v_(block_, idx, k, j, i);
  }

 private:
  const T& v_;
  const int ntot_ = 1;
  const int chunk_size_;
  const int offset_;
  const int block_;
};

template <class T, class CLOSURE, bool STORE_GUESS, bool EDDINGTON_KNOWN>
TaskStatus MomentCon2PrimImpl(T* rc) {
  printf("%s:%i\n", __FILE__, __LINE__);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  std::vector<std::string> variables{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity, ir::xi, ir::phi};
  if (EDDINGTON_KNOWN) {
    variables.push_back(ir::tilPi);
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (EDDINGTON_KNOWN) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }
  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);

  auto iXi = imap.GetFlatIdx(ir::xi);
  auto iPhi = imap.GetFlatIdx(ir::phi);

  auto geom = Geometry::GetCoordinateSystem(rc);
  const Real pi = acos(-1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Con2Prim", DevExecSpace(),
      0, v.GetDim(5)-1, // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        Vec con_v{{v(b, pv(0), k, j, i),
                   v(b, pv(1), k, j, i),
                   v(b, pv(2), k, j, i)}};
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        const Real isdetgam = 1.0/geom.DetGamma(CellLocation::Cent, b, k, j, i);

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);
        CLOSURE c(con_v, &g);

        Real J;
        Vec covH;
        Tens2 conTilPi;
        Real E = v(b, cE(ispec), k, j, i) * isdetgam;
        Vec covF ={{v(b, cF(ispec, 0), k, j, i) * isdetgam,
                    v(b, cF(ispec, 1), k, j, i) * isdetgam,
                    v(b, cF(ispec, 2), k, j, i) * isdetgam}};

        Real xi = 0.0;
        Real phi = pi;
        if (STORE_GUESS) {
          xi = v(b, iXi(ispec), k, j, i);
          phi = 1.0001*v(b, iPhi(ispec), k, j, i);
        }
        if (STORE_GUESS) {
          v(b, iXi(ispec), k, j, i) = xi;
          v(b, iPhi(ispec), k, j, i) = phi;
        }
        if (EDDINGTON_KNOWN) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
          if (i == 128) {
            SPACELOOP2(ii, jj) {
             printf("conTilPi(%i, %i) = %e\n", ii, jj, conTilPi(ii, jj));
            }
          }
        } else {
          c.GetCovTilPiFromCon(E, covF, xi, phi, &conTilPi);
        }
        c.Con2Prim(E, covF, conTilPi, &J, &covH);
        if (std::isnan(J) || std::isnan(covH(0))) PARTHENON_FAIL("Radiation Con2Prim NaN.");

        v(b, pJ(ispec), k, j, i) = J;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          v(b, pH(ispec, idir), k, j, i) = covH(idir)/J; // Used the scaled value of the rest frame flux for reconstruction
        }
      });

  return TaskStatus::complete;
}

template<class T>
TaskStatus MomentCon2Prim(T* rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");

  using settings = ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentCon2PrimImpl<T, ClosureM1<Vec, Tens2, settings>, true, false>(rc);
  }
  else if (method == "moment_eddington") {
    return MomentCon2PrimImpl<T, ClosureEdd<Vec, Tens2, settings>, false, false>(rc);
  } else if (method == "mocmc") {
    return MomentCon2PrimImpl<T, ClosureMOCMC<Vec, Tens2, settings>, false, true>(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown");
  }
  return TaskStatus::fail;
}
//template TaskStatus MomentCon2Prim<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MomentCon2Prim<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE, bool EDDINGTON_KNOWN>
TaskStatus MomentPrim2ConImpl(T* rc, IndexDomain domain) {
  printf("%s:%i\n", __FILE__, __LINE__);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(domain);
  IndexRange jb = pm->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pm->cellbounds.GetBoundsK(domain);

  std::vector<std::string> variables{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity, ir::tilPi};
  if (EDDINGTON_KNOWN) {
    variables.push_back(ir::tilPi);
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (EDDINGTON_KNOWN) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Prim2Con", DevExecSpace(),
      0, v.GetDim(5)-1, // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        // Set up the background
        Vec con_v{{v(b, pv(0), k, j, i),
                   v(b, pv(1), k, j, i),
                   v(b, pv(2), k, j, i)}};
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, 0, b, j, i);
        CLOSURE c(con_v, &g);

        Real E;
        Vec covF;
        Tens2 conTilPi;
        Real J = v(b, pJ(ispec), k, j, i);
        Vec covH ={{v(b, pH(ispec, 0), k, j, i)*J,
                    v(b, pH(ispec, 1), k, j, i)*J,
                    v(b, pH(ispec, 2), k, j, i)*J}};

        if (EDDINGTON_KNOWN) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          c.GetCovTilPiFromPrim(J, covH, &conTilPi);
        }
        c.Prim2Con(J, covH, conTilPi, &E, &covF);

        v(b, cE(ispec), k, j, i) = sdetgam * E;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) {
          v(b, cF(ispec, idir), k, j, i) = sdetgam * covF(idir);
        }
      });

  return TaskStatus::complete;
}


template<class T>
TaskStatus MomentPrim2Con(T* rc, IndexDomain domain) {
  printf("%s:%i\n", __FILE__, __LINE__);
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings = ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentPrim2ConImpl<T, ClosureM1<Vec, Tens2, settings>, false>(rc, domain);
  }
  else if (method == "moment_eddington") {
    return MomentPrim2ConImpl<T, ClosureEdd<Vec, Tens2, settings>, false>(rc, domain);
  }
  else if (method == "mocmc") {
    return MomentPrim2ConImpl<T, ClosureMOCMC<Vec, Tens2, settings>, true>(rc, domain);
  }
  else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}

template TaskStatus MomentPrim2Con<MeshBlockData<Real>>(MeshBlockData<Real> *, IndexDomain);

template <class T>
TaskStatus ReconstructEdgeStates(T* rc) {
  printf("%s:%i\n", __FILE__, __LINE__);

  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  bool eddington_known = false;
  if (method == "mocmc") {
    eddington_known = true;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = ( pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = ( pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = ( pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  VariablePack<Real> ql_base = rc->PackVariables(std::vector<std::string>{ir::ql}, imap_ql);
  VariablePack<Real> qr_base = rc->PackVariables(std::vector<std::string>{ir::qr}, imap_qr);
  std::vector<std::string> variables = {pr::J, pr::H, ir::dJ};
  if (eddington_known) {
    variables.push_back(ir::tilPi);
  }
  VariablePack<Real> v = rc->PackVariables(variables, imap);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (eddington_known) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  ParArrayND<Real> ql_v = rc->Get(ir::ql_v).data;
  ParArrayND<Real> qr_v = rc->Get(ir::qr_v).data;
  VariablePack<Real> v_vel = rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
  auto qIdx = imap_ql.GetFlatIdx(ir::ql);

  const int nspec = qIdx.DimSize(1);
  const int nrecon = 4*nspec;

  const int offset = imap_ql[ir::ql].first;

  const int nblock = ql_base.GetDim(5);
  const int ndim = pmb->pmy_mesh->ndim;
  auto& coords = pmb->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(),
      X1DIR, ndim, // Loop over directions for reconstruction
      0, nblock-1, // Loop over blocks
      kb.s - dk, kb.e + dk, // z-loop
      jb.s - dj, jb.e + dj, // y-loop
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int iface, const int b, const int k, const int j, const int i) {
        ReconstructionIndexer<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionIndexer<VariablePack<Real>> qr(qr_base, nrecon, offset, b);
        // Reconstruct radiation
        for (int ivar = 0; ivar<nrecon; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(iface, ivar, k, j, i, v, ql, qr);
        }
        // Reconstruct velocity for radiation
        for (int ivar = 0; ivar<3; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(iface, ivar, k, j, i, v_vel, ql_v, qr_v);
        }

        // Calculate spatial derivatives of J at zone faces for diffusion limit
        //    x-->
        //    +---+---+
        //    | a | b |
        //    +---+---+
        //    | c Q d |
        //  ^ +---+---+
        //  | | e | f |
        //  y +---+---+
        //
        //  dJ/dx (@ Q) = (d - c)/dx
        //  dJ/dy (@ Q) = (a + b - e - f)/(4*dy)

        const int off_k = (iface == 3 ? 1 : 0);
        const int off_j = (iface == 2 ? 1 : 0);
        const int off_i = (iface == 1 ? 1 : 0);
        const int st_k[3] = {0, 0, 1};
        const int st_j[3] = {0, 1, 0};
        const int st_i[3] = {1, 0, 0};
        for (int ispec=0; ispec<nspec; ++ispec) {
          for (int idir = X1DIR; idir <= ndim; ++idir) {
            // Calculate the derivatives in the plane of the face (and put junk in the derivative perpendicular to the face)
            const Real dy = coords.Dx(idir, k, j, i);
            v(b, idx_dJ(ispec, idir-1, iface-1), k, j, i) = (v(b, idx_J(ispec), k+st_k[idir-1], j+st_j[idir-1], i+st_i[idir-1])
                                                              -v(b, idx_J(ispec), k-st_k[idir-1], j-st_j[idir-1], i-st_i[idir-1])
                                                              +v(b, idx_J(ispec), k+st_k[idir-1]-off_k, j+st_j[idir-1]-off_j, i+st_i[idir-1]-off_i)
                                                              -v(b, idx_J(ispec), k-st_k[idir-1]-off_k, j-st_j[idir-1]-off_j, i-st_i[idir-1]-off_i))/(4*dy);

          }
          // Overwrite the derivative perpendicular to the face
          const Real dx = coords.Dx(iface, k, j, i);
          v(b, idx_dJ(ispec, iface-1, iface-1), k, j, i) =  (v(b, idx_J(ispec), k, j, i) - v(b, idx_J(ispec), k-off_k, j-off_j, i-off_i))/dx;
        }
      });
  return TaskStatus::complete;
}
template TaskStatus ReconstructEdgeStates<MeshBlockData<Real>>(MeshBlockData<Real> *);

// This really only works for MeshBlockData right now since fluxes don't have a block index
template <class T, class CLOSURE, bool EDDINGTON_KNOWN>
TaskStatus CalculateFluxesImpl(T* rc) {
  printf("%s:%i\n", __FILE__, __LINE__);

  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = ( pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = ( pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = ( pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  std::vector<std::string> vars {ir::ql, ir::qr, ir::ql_v, ir::qr_v, ir::dJ, ir::kappaH};
  std::vector<std::string> flxs {cr::E, cr::F};

  auto v = rc->PackVariablesAndFluxes(vars, flxs, imap);

  auto idx_qlv = imap.GetFlatIdx(ir::ql_v);
  auto idx_qrv = imap.GetFlatIdx(ir::qr_v);
  auto idx_ql = imap.GetFlatIdx(ir::ql);
  auto idx_qr = imap.GetFlatIdx(ir::qr);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);

  auto idx_Ff = imap.GetFlatIdx(cr::F);
  auto idx_Ef = imap.GetFlatIdx(cr::E);

  const int nspec = idx_ql.DimSize(1);

  // const int nblock = 1; //v.GetDim(5);

  auto geom = Geometry::GetCoordinateSystem(rc);

  const Real kappaH_min = 1.e-20;

  auto& coords = pmb->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Fluxes", DevExecSpace(),
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions
      //0, nblock-1, // Loop over reconstructed variables
      kb.s - dk, kb.e + dk, // z-loop
      jb.s - dj, jb.e + dj, // y-loop
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir_in, const int k, const int j, const int i) {

        const int idir = idir_in - 1;

        const int koff = (idir_in == 3 ? 1 : 0);
        const int joff = (idir_in == 2 ? 1 : 0);
        const int ioff = (idir_in == 1 ? 1 : 0);

        CellLocation face;
          switch (idir) {
            case(0):
              face = CellLocation::Face1;
              break;
            case(1):
              face = CellLocation::Face2;
              break;
            case(2):
              face = CellLocation::Face3;
              break;
          }

        Vec con_beta;
        Tens2 cov_gamma;
        geom.Metric(face, k, j, i, cov_gamma.data);
        geom.ContravariantShift(face, k, j, i, con_beta.data);
        const Real sdetgam = geom.DetGamma(face, k, j, i);
        typename CLOSURE::LocalGeometryType g(geom, face, 0, k, j, i);

        const Real dx = coords.Dx(idir_in, k, j, i)*sqrt(cov_gamma(idir, idir));

        for (int ispec = 0; ispec<nspec; ++ispec) {

          const Real& Jl = v(idx_ql(ispec, 0, idir), k, j, i);
          const Real& Jr = v(idx_qr(ispec, 0, idir), k, j, i);
          const Vec Hl = {Jl*v(idx_ql(ispec, 1, idir), k, j, i),
                          Jl*v(idx_ql(ispec, 2, idir), k, j, i),
                          Jl*v(idx_ql(ispec, 3, idir), k, j, i)};
          const Vec Hr = {Jr*v(idx_qr(ispec, 1, idir), k, j, i),
                          Jr*v(idx_qr(ispec, 2, idir), k, j, i),
                          Jr*v(idx_qr(ispec, 3, idir), k, j, i)};


          Vec con_vl{{v(idx_qlv(0, idir), k, j, i),
                      v(idx_qlv(1, idir), k, j, i),
                      v(idx_qlv(2, idir), k, j, i)}};
          Vec con_vr{{v(idx_qrv(0, idir), k, j, i),
                      v(idx_qrv(1, idir), k, j, i),
                      v(idx_qrv(2, idir), k, j, i)}};

          Vec cov_dJ{{v(idx_dJ(ispec, 0, idir), k, j, i),
                      v(idx_dJ(ispec, 1, idir), k, j, i),
                      v(idx_dJ(ispec, 2, idir), k, j, i)}};

          // Calculate the geometric mean of the opacity on either side of the interface, this
          // is necessary for handling the asymptotic limit near sharp surfaces
          Real kappaH = sqrt((v(idx_kappaH(ispec), k, j, i)*v(idx_kappaH(ispec), k - koff, j - joff, i - ioff)));

          const Real a = tanh(ratio(1.0, std::pow(std::abs(kappaH*dx), 1)));

          // Calculate the observer frame quantities on either side of the interface
          /// TODO: (LFR) Add other contributions to the asymptotic flux
          Vec HasymL = -cov_dJ/(3*kappaH + 3*kappaH_min);
          Vec HasymR = HasymL;
          CLOSURE cl(con_vl, &g);
          CLOSURE cr(con_vr, &g);
          Real El, Er;
          Tens2 con_tilPil, con_tilPir;
          Vec covFl, conFl, conFl_asym;
          Vec covFr, conFr, conFr_asym;
          Tens2 Pl, Pl_asym; // P^i_j on the left side of the interface
          Tens2 Pr, Pr_asym; // P^i_j on the right side of the interface

          // Fluxes in the asymptotic limit
          if (EDDINGTON_KNOWN) {
            // Use reconstructed values of tilPi
          } else {
            cl.GetCovTilPiFromPrim(Jl, HasymL, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, HasymR, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, HasymL, con_tilPil, &conFl_asym, &Pl_asym);
          cr.getFluxesFromPrim(Jr, HasymR, con_tilPir, &conFr_asym, &Pr_asym);

          // Regular fluxes
          if (EDDINGTON_KNOWN) {
            // Use reconstructed values of tilPi
          } else {
            cl.GetCovTilPiFromPrim(Jl, Hl, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, Hr, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, Hl, con_tilPil, &conFl, &Pl);
          cr.getFluxesFromPrim(Jr, Hr, con_tilPir, &conFr, &Pr);
          cl.Prim2Con(Jl, Hl, con_tilPil, &El, &covFl);
          cr.Prim2Con(Jr, Hr, con_tilPir, &Er, &covFr);

          // Mix the fluxes by the Peclet number
          // TODO: (LFR) Make better choices
          const Real speed = a*1.0 + (1-a)*std::max(sqrt(cl.v2), sqrt(cr.v2));
          conFl = a*conFl + (1-a)*conFl_asym;
          conFr = a*conFr + (1-a)*conFr_asym;
          Pl = a*Pl + (1-a)*Pl_asym;
          Pr = a*Pr + (1-a)*Pr_asym;

          // Correct the fluxes with the shift terms
          conFl(idir) -= con_beta(idir)*El;
          conFr(idir) -= con_beta(idir)*Er;

          SPACELOOP(ii) {
            Pl(idir, ii) -= con_beta(idir)*covFl(ii);
            Pr(idir, ii) -= con_beta(idir)*covFr(ii);
          }

          // Calculate the numerical flux using LLF
          v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.5*sdetgam*(conFl(idir) + conFr(idir) + speed*(El - Er));

          SPACELOOP(ii) v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) = 0.5*sdetgam*(Pl(idir, ii) + Pr(idir, ii)
                                                                                + speed*(covFl(ii) - covFr(ii)));
          if (sdetgam < std::numeric_limits<Real>::min()*10) {
            v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.0;
            SPACELOOP(ii) v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) = 0.0;
          }
        }
      });

  return TaskStatus::complete;
}

template<class T>
TaskStatus CalculateFluxes(T* rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings = ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateFluxesImpl<T, ClosureM1<Vec, Tens2, settings>, false >(rc);
  } else if (method == "moment_eddington") {
    return CalculateFluxesImpl<T, ClosureEdd<Vec, Tens2, settings>, false >(rc);
  } else if (method == "mocmc") {
    return CalculateFluxesImpl<T, ClosureMOCMC<Vec, Tens2, settings>, true >(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateFluxes<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE, bool EDDINGTON_KNOWN>
TaskStatus CalculateGeometricSourceImpl(T *rc, T *rc_src) {
  printf("%s:%i\n", __FILE__, __LINE__);

  constexpr int ND = Geometry::NDFULL;
  //constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace p = fluid_prim;
  PackIndexMap imap;
  std::vector<std::string> vars{cr::E, cr::F, pr::J, pr::H, p::velocity};
  if (EDDINGTON_KNOWN) {
    vars.push_back(ir::tilPi);
  }
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(p::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (EDDINGTON_KNOWN) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  PackIndexMap imap_src;
  std::vector<std::string> vars_src{cr::E, cr::F};
  auto v_src = rc_src->PackVariables(vars_src, imap_src);
  auto idx_E_src = imap_src.GetFlatIdx(cr::E);
  auto idx_F_src = imap_src.GetFlatIdx(cr::F);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_E.DimSize(1);
  printf("%s:%i\n", __FILE__, __LINE__);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::GeometricSource", DevExecSpace(),
      0, nblock-1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        // Set up the background state
        Vec con_v{{v(iblock, pv(0), k, j, i),
                   v(iblock, pv(1), k, j, i),
                   v(iblock, pv(2), k, j, i)}};
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, iblock, k, j, i);
        CLOSURE c(con_v, &g);

        Real alp = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
        Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
        Vec con_beta;
        geom.ContravariantShift(CellLocation::Cent, k, j, i, con_beta.data);
        Real beta2 = 0.0;
        SPACELOOP2(ii,jj) beta2 += con_beta(ii)*con_beta(jj)*cov_gamma(ii,jj);

        Real dlnalp[ND];
        Real Gamma[ND][ND][ND];
        geom.GradLnAlpha(CellLocation::Cent, iblock, k, j, i, dlnalp);
        geom.ConnectionCoefficient(CellLocation::Cent, iblock, k, j, i, Gamma);

        // Get the gradient of the shift from the Christoffel symbols of the first kind
        // Get the extrinsic curvature from the Christoffel symbols of the first kind
        // All indices are covariant
        Tens2 dbeta, K;
        const Real iFac = 1.0/(alp + beta2/alp);
        SPACELOOP2(ii, jj) {
          dbeta(ii,jj) = Gamma[ii+1][jj+1][0] + Gamma[ii+1][0][jj+1];
          K(ii,jj) = Gamma[ii+1][0][jj+1];
          SPACELOOP(kk) K(ii,jj) -= Gamma[ii+1][kk+1][jj+1]*con_beta(kk);
          K(ii,jj) *= iFac;
        }



        for (int ispec = 0; ispec<nspec; ++ispec) {
          Real E = v(iblock, idx_E(ispec), k, j, i)/sdetgam;
          Real J = v(iblock, idx_J(ispec), k, j, i);
          Vec covF{{v(iblock, idx_F(ispec, 0), k, j, i)/sdetgam,
                     v(iblock, idx_F(ispec, 1), k, j, i)/sdetgam,
                     v(iblock, idx_F(ispec, 2), k, j, i)/sdetgam}};
          Vec covH{{J*v(iblock, idx_H(ispec, 0), k, j, i),
                    J*v(iblock, idx_H(ispec, 1), k, j, i),
                    J*v(iblock, idx_H(ispec, 2), k, j, i)}};
          Vec conF;
          g.raise3Vector(covF, &conF);
          Tens2 conP, con_tilPi;

          if (EDDINGTON_KNOWN) {

          } else {
            c.GetCovTilPiFromPrim(J, covH, &con_tilPi);
          }
          c.getConPFromPrim(J, covH, con_tilPi, &conP);

          Real srcE = 0.0;
          SPACELOOP2(ii, jj) srcE += K(ii,jj)*conP(ii, jj);
          SPACELOOP(ii) srcE -= dlnalp[ii+1]*conF(ii);
          srcE *= alp;

          Vec srcF{0,0,0};
          SPACELOOP(ii) {
            SPACELOOP(jj) srcF(ii) += covF(jj)*dbeta(ii,jj);
            srcF(ii) -= alp*E*dlnalp[ii+1];
            SPACELOOP2(jj, kk) srcF(ii) += alp*conP(jj,kk)*Gamma[jj+1][kk+1][ii+1];
          }
          v_src(iblock, idx_E_src(ispec), k, j, i) = sdetgam*srcE;
          SPACELOOP(ii) v_src(iblock, idx_F_src(ispec, ii), k, j, i) = sdetgam*srcF(ii);
        }
  });
  printf("%s:%i\n", __FILE__, __LINE__);
  return TaskStatus::complete;
}
template<class T>
TaskStatus CalculateGeometricSource(T* rc, T* rc_src) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings = ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateGeometricSourceImpl<T, ClosureM1<Vec, Tens2, settings>, false>(rc, rc_src);
  }
  else if (method == "moment_eddington") {
    return CalculateGeometricSourceImpl<T, ClosureEdd<Vec, Tens2, settings>, false>(rc, rc_src);
  }
  else if (method == "mocmc") {
    return CalculateGeometricSourceImpl<T, ClosureMOCMC<Vec, Tens2, settings>, true>(rc, rc_src);
  }
  else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateGeometricSource<MeshBlockData<Real>>(MeshBlockData<Real> *, MeshBlockData<Real> *);

template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) {
  printf("%s:%i\n", __FILE__, __LINE__);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E, cr::F, p::density, p::temperature, p::ye, p::velocity,
                                pr::J, pr::H, ir::kappaJ, ir::kappaH, ir::JBB};
  if (update_fluid) {
    vars.push_back(c::energy);
    vars.push_back(c::momentum);
    vars.push_back(c::ye);
  }

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;

  int ceng(-1), cmom_lo(-1), cye(-1);
  if (update_fluid) {
    ceng = imap[c::energy].first;
    cmom_lo = imap[c::momentum].first;
    cye = imap[c::ye].first;
  }

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_E.DimSize(1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(),
      0, nblock-1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        for (int ispec = 0; ispec<nspec; ++ispec) {

          // Set up the background state
          Vec con_v{{v(iblock, pv(0), k, j, i),
                     v(iblock, pv(1), k, j, i),
                     v(iblock, pv(2), k, j, i)}};
          Tens2 cov_gamma;
          geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);
          Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
          Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
          LocalThreeGeometry g(geom, CellLocation::Cent, iblock, k, j, i);

          /// TODO: (LFR) Move beyond Eddington for this update
          ClosureEdd<Vec, Tens2> c(con_v, &g);

          Real Estar = v(iblock, idx_E(ispec), k, j, i)/sdetgam;
          Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i)/sdetgam,
                        v(iblock, idx_F(ispec, 1), k, j, i)/sdetgam,
                        v(iblock, idx_F(ispec, 2), k, j, i)/sdetgam};

          Real dE;
          Vec cov_dF;


          // Treat the Eddington tensor explicitly for now
          Real& J = v(iblock, idx_J(ispec), k, j, i);
          Vec cov_H{{J*v(iblock, idx_H(ispec, 0), k, j, i),
                     J*v(iblock, idx_H(ispec, 1), k, j, i),
                     J*v(iblock, idx_H(ispec, 2), k, j, i),
                    }};
          Tens2 con_tilPi;

          c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);

          Real B = v(iblock, idx_JBB(ispec), k, j, i);
          Real tauJ = alpha*dt*v(iblock, idx_kappaJ(ispec), k, j, i);
          Real tauH = alpha*dt*v(iblock, idx_kappaH(ispec), k, j, i);
          Real kappaH =  v(iblock, idx_kappaH(ispec), k, j, i);
          c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, B,
                               tauJ, tauH, &dE, &cov_dF);

          // Add source corrections to conserved iration variables
          v(iblock, idx_E(ispec), k, j, i) += sdetgam*dE;
          for (int idir=0; idir<3; ++idir) {
            v(iblock, idx_F(ispec, idir), k, j, i) += sdetgam*cov_dF(idir);
          }

          // Add source corrections to conserved fluid variables
          if (update_fluid) {
            v(iblock, cye, k, j, i) -= sdetgam*0.0;
            v(iblock, ceng, k, j, i) -= sdetgam*dE;
            v(iblock, cmom_lo + 0, k, j, i) -= sdetgam*cov_dF(0);
            v(iblock, cmom_lo + 1, k, j, i) -= sdetgam*cov_dF(1);
            v(iblock, cmom_lo + 2, k, j, i) -= sdetgam*cov_dF(2);
          }

        }
      });

  return TaskStatus::complete;
}
template TaskStatus MomentFluidSource<MeshData<Real>>(MeshData<Real> *, Real, bool);
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real, bool);

template <class T>
TaskStatus MomentCalculateOpacities(T *rc) {
  printf("%s:%i\n", __FILE__, __LINE__);

  auto *pmb = rc->GetParentPointer().get();

  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  const bool rad_mocmc_active = (rad->Param<std::string>("method") == "mocmc");
  if (rad_mocmc_active) {
    // For MOCMC, opacities are calculated by averaging over samples in interaction call
    return TaskStatus::complete;
  }

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{p::density, p::temperature, p::ye, p::velocity, ir::kappaJ, ir::kappaH, ir::JBB};

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first;
  int pT = imap[p::temperature].first;
  int pYe = imap[p::ye].first;

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Mainly for testing purposes, probably should be able to do this with the opacity code itself
  const auto B_fake = rad->Param<Real>("B_fake");
  const auto use_B_fake = rad->Param<bool>("use_B_fake");
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  // Get the device opacity object
  using namespace singularity::neutrinos;
  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_kappaJ.DimSize(1);

  /// TODO: (LFR) Fix this junk
  RadiationType dev_species[3] = {species[0], species[1], species[2]};

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(),
      0, nblock-1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        for (int ispec = 0; ispec<nspec; ++ispec) {
          /// TODO: (LFR) Need to make a grid variable holding the energy integrated opacity so that we can
          ///             create a task to fill the opacity based on MoCMC or some other rule.
          /// TOMAYBENOTDO: (BRR) Can't we just use kappaH and kappaJ?
          const Real enu = 10.0; // Assume we are gray for now or can take the peak opacity at enu = 10 MeV
          const Real rho =  v(iblock, prho, k, j, i);
          const Real Temp =  v(iblock, pT, k, j, i);
          const Real Ye = v(iblock, pYe, k, j, i);

          Real kappa = d_opacity.AbsorptionCoefficient(rho, Temp, Ye, dev_species[ispec], enu);
          //const Real emis = d_opacity.Emissivity(rho, Temp, Ye, dev_species[ispec]);
          //Real B = emis/kappa;
          Real B = d_opacity.ThermalDistributionOfT(Temp, dev_species[ispec]);
          if (use_B_fake) B = B_fake;

          v(iblock, idx_JBB(ispec), k, j, i) = B;
          v(iblock, idx_kappaJ(ispec), k, j, i) = kappa*(1.0 - scattering_fraction);
          v(iblock, idx_kappaH(ispec), k, j, i) = kappa;

        }
      });

  return TaskStatus::complete;
}
template TaskStatus MomentCalculateOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *);
} //namespace irationMoments
