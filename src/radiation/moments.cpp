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

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "phoebus_utils/programming_utils.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/local_three_geometry.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

namespace radiation {

template <class T>
class ReconstructionIndexer {
 public:
  KOKKOS_INLINE_FUNCTION
  ReconstructionIndexer(const T &v, const int chunk_size, const int offset,
                        const int block = 0)
      : v_(v), chunk_size_(chunk_size), offset_(offset), block_(block) {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int idir, const int ivar, const int k, const int j,
                   const int i) const {
    const int idx = idir * chunk_size_ + ivar + offset_;
    return v_(block_, idx, k, j, i);
  }

 private:
  const T &v_;
  const int ntot_ = 1;
  const int chunk_size_;
  const int offset_;
  const int block_;
};

template <class T, class CLOSURE, bool STORE_GUESS>
TaskStatus MomentCon2PrimImpl(T *rc) {
  printf("%s:%i MomentCon2PrimImpl\n", __FILE__, __LINE__);
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  std::vector<std::string> variables{cr::E,  cr::F,  pr::J, pr::H, fluid_prim::velocity,
                                     ir::xi, ir::phi};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    variables.push_back(ir::tilPi);
  }
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto cE = imap.GetFlatIdx(cr::E);
  auto pJ = imap.GetFlatIdx(pr::J);
  auto cF = imap.GetFlatIdx(cr::F);
  auto pH = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  // TODO(BRR) Should be able to just get an invalid iTilPi back from imap when MOCMC off
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }
  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);

  auto iXi = imap.GetFlatIdx(ir::xi);
  auto iPhi = imap.GetFlatIdx(ir::phi);

  auto geom = Geometry::GetCoordinateSystem(rc);
  const Real pi = acos(-1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Con2Prim", DevExecSpace(), 0,
      v.GetDim(5) - 1,  // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e,       // z-loop
      jb.s, jb.e,       // y-loop
      ib.s, ib.e,       // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        Vec con_v{{v(b, pv(0), k, j, i), v(b, pv(1), k, j, i), v(b, pv(2), k, j, i)}};
        Tens2 cov_gamma;
        geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        const Real isdetgam = 1.0 / geom.DetGamma(CellLocation::Cent, b, k, j, i);

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j, i);
        CLOSURE c(con_v, &g);

        Real J;
        Vec covH;
        Tens2 conTilPi;
        Real E = v(b, cE(ispec), k, j, i) * isdetgam;
        Vec covF = {{v(b, cF(ispec, 0), k, j, i) * isdetgam,
                     v(b, cF(ispec, 1), k, j, i) * isdetgam,
                     v(b, cF(ispec, 2), k, j, i) * isdetgam}};
         if (i == 4 && j == 4) {
         printf("[%i %i %i][%i] E: %e covF: %e %e %e\n", k, j, i, ispec,
           E, covF(0), covF(1), covF(2));
         }

        if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          Real xi = 0.0;
          Real phi = pi;
          if (STORE_GUESS) {
            xi = v(b, iXi(ispec), k, j, i);
            phi = 1.0001 * v(b, iPhi(ispec), k, j, i);
          }
          c.GetCovTilPiFromCon(E, covF, xi, phi, &conTilPi);
          if (STORE_GUESS) {
            v(b, iXi(ispec), k, j, i) = xi;
            v(b, iPhi(ispec), k, j, i) = phi;
          }
        }
        c.Con2Prim(E, covF, conTilPi, &J, &covH);
         if (i == 4) {
         printf("[%i %i %i][%i] J: %e covH: %e %e %e\n", k, j, i, ispec,
           J, covH(0), covH(1), covH(2));
         }
        if (std::isnan(J) || std::isnan(covH(0)) || std::isnan(covH(1)) ||
            std::isnan(covH(2))) {
          PARTHENON_FAIL("Radiation Con2Prim NaN.");
        }

        v(b, pJ(ispec), k, j, i) = J;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          v(b, pH(ispec, idir), k, j, i) = robust::ratio(covH(idir), J);
            //  covH(idir) /
            //  J; // Used the scaled value of the rest frame flux for reconstruction
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentCon2Prim(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");

  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentCon2PrimImpl<T, ClosureM1<Vec, Tens2, settings>, true>(rc);
  } else if (method == "moment_eddington") {
    return MomentCon2PrimImpl<T, ClosureEdd<Vec, Tens2, settings>, false>(rc);
  } else if (method == "mocmc") {
    return MomentCon2PrimImpl<T, ClosureMOCMC<Vec, Tens2, settings>, false>(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown");
  }
  return TaskStatus::fail;
}
// template TaskStatus MomentCon2Prim<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MomentCon2Prim<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE>
TaskStatus MomentPrim2ConImpl(T *rc, IndexDomain domain) {
  printf("%s:%i MomentPrim2ConImpl\n", __FILE__, __LINE__);
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(domain);
  IndexRange jb = pm->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pm->cellbounds.GetBoundsK(domain);

  std::vector<std::string> variables{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
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
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Prim2Con", DevExecSpace(), 0,
      v.GetDim(5) - 1,  // Loop over meshblocks
      specB.s, specB.e, // Loop over species
      kb.s, kb.e,       // z-loop
      jb.s, jb.e,       // y-loop
      ib.s, ib.e,       // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        // Set up the background
        Vec con_v{{v(b, pv(0), k, j, i), v(b, pv(1), k, j, i), v(b, pv(2), k, j, i)}};
        // TODO(BRR) Remove?
        //Tens2 cov_gamma;
        //geom.Metric(CellLocation::Cent, b, k, j, i, cov_gamma.data);
        const Real sdetgam = geom.DetGamma(CellLocation::Cent, b, k, j, i);

        typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, 0, b, j, i);
        CLOSURE c(con_v, &g);

        Real E;
        Vec covF;
        Tens2 conTilPi;
        Real J = v(b, pJ(ispec), k, j, i);
        Vec covH = {{v(b, pH(ispec, 0), k, j, i) * J, v(b, pH(ispec, 1), k, j, i) * J,
                     v(b, pH(ispec, 2), k, j, i) * J}};

        if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
          SPACELOOP2(ii, jj) { conTilPi(ii, jj) = v(b, iTilPi(ispec, ii, jj), k, j, i); }
        } else {
          c.GetCovTilPiFromPrim(J, covH, &conTilPi);
        }

        c.Prim2Con(J, covH, conTilPi, &E, &covF);

        v(b, cE(ispec), k, j, i) = sdetgam * E;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) {
          v(b, cF(ispec, idir), k, j, i) = sdetgam * covF(idir);
        }
         if (i == 4) {
         printf("[%i %i %i][%i] E: %e covF: %e %e %e\n", k, j, i, ispec,
           v(b, cE(ispec), k, j, i), 
           v(b, cF(ispec, 0), k, j, i),
           v(b, cF(ispec, 1), k, j, i),
           v(b, cF(ispec, 2), k, j, i));
         }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentPrim2Con(T *rc, IndexDomain domain) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return MomentPrim2ConImpl<T, ClosureM1<Vec, Tens2, settings>>(rc, domain);
  } else if (method == "moment_eddington") {
    return MomentPrim2ConImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc, domain);
  } else if (method == "mocmc") {
    return MomentPrim2ConImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc, domain);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}

template TaskStatus MomentPrim2Con<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                        IndexDomain);

template <class T>
TaskStatus ReconstructEdgeStates(T *rc) {
  printf("%s:%i ReconstructEdgeStates\n", __FILE__, __LINE__);
  using namespace PhoebusReconstruction;

  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  bool eddington_known = false;
  if (method == "mocmc") {
    eddington_known = true;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = (pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = (pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = (pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  VariablePack<Real> ql_base =
      rc->PackVariables(std::vector<std::string>{ir::ql}, imap_ql);
  VariablePack<Real> qr_base =
      rc->PackVariables(std::vector<std::string>{ir::qr}, imap_qr);
  std::vector<std::string> variables = {pr::J, pr::H};
  if (eddington_known) {
    variables.push_back(ir::tilPi);
  }
  variables.push_back(ir::dJ);
  VariablePack<Real> v = rc->PackVariables(variables, imap);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (eddington_known) {
    iTilPi = imap.GetFlatIdx(ir::tilPi);
  }

  ParArrayND<Real> ql_v = rc->Get(ir::ql_v).data;
  ParArrayND<Real> qr_v = rc->Get(ir::qr_v).data;
  VariablePack<Real> v_vel =
      rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
  auto qIdx = imap_ql.GetFlatIdx(ir::ql);

  const int nspec = qIdx.DimSize(1);
  int nrecon = 4 * nspec;
  if (eddington_known) {
    nrecon = (4 + 9) * nspec; // TODO(BRR) 6 instead of 9 for conTilPi by symmetry
  }

  const int offset = imap_ql[ir::ql].first;

  const int nblock = ql_base.GetDim(5);
  const int ndim = pmb->pmy_mesh->ndim;
  auto &coords = pmb->coords;

  // TODO(JCD): par_for_outer doesn't have a 4d loop pattern which is needed for blocks
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(), 0, 0, 0,
      nrecon + 2, kb.s - dk, kb.e + dk, jb.s - dj, jb.e + dj,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int n, const int k, const int j) {
        const int b = 0; // this will be replaced by the block arg to the lambda
        ReconstructionIndexer<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionIndexer<VariablePack<Real>> qr(qr_base, nrecon, offset, b);

        const VariablePack<Real> &var = (n < nrecon ? v : v_vel);
        const int var_id = n % nrecon;
        Real *pv = &var(b, var_id, k, j, 0);
        Real *pvim1 = pv - 1;
        Real *pvip1 = pv + 1;
        Real *pvjm1 = &var(b, var_id, k, j - dj, 0);
        Real *pvjp1 = &var(b, var_id, k, j + dj, 0);
        Real *pvkm1 = &var(b, var_id, k - dk, j, 0);
        Real *pvkp1 = &var(b, var_id, k + dk, j, 0);
        if (std::isnan(pv[4])) {
          printf("[%i %i %i %i %i] nan!\n", b, var_id, k, j, 4);
          exit(-1);
        }
        Real *vi_l, *vi_r, *vj_l, *vj_r, *vk_l, *vk_r;
        if (n < nrecon) {
          vi_l = &ql(0, n, k, j, 1);
          vi_r = &qr(0, n, k, j, 0);
          vj_l = &ql(1 % ndim, n, k, j + dj, 0);
          vj_r = &qr(1 % ndim, n, k, j, 0);
          vk_l = &ql(2 % ndim, n, k + dk, j, 0);
          vk_r = &qr(2 % ndim, n, k, j, 0);
        } else {
          vi_l = &ql_v(0, var_id, k, j, 1);
          vi_r = &qr_v(0, var_id, k, j, 0);
          vj_l = &ql_v(1 % ndim, var_id, k, j + dj, 0);
          vj_r = &qr_v(1 % ndim, var_id, k, j, 0);
          vk_l = &ql_v(2 % ndim, var_id, k + dk, j, 0);
          vk_r = &qr_v(2 % ndim, var_id, k, j, 0);
        }

        // TODO(JCD): do we want to enable other recon methods like weno5?
        // x-direction
        ReconLoop<PiecewiseLinear>(member, ib.s - 1, ib.e + 1, pvim1, pv, pvip1, vi_l,
                                   vi_r);
        // y-direction
        if (ndim > 1)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvjm1, pv, pvjp1, vj_l, vj_r);
        // z-direction
        if (ndim > 2)
          ReconLoop<PiecewiseLinear>(member, ib.s, ib.e, pvkm1, pv, pvkp1, vk_l, vk_r);

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
        if (n < nspec) {
          const Real idx = 1.0 / coords.Dx(X1DIR, k, j, 0);
          const Real idx4 = 0.25 * idx;
          const Real idy = 1.0 / coords.Dx(X2DIR, k, j, 0);
          const Real idy4 = 0.25 * idy;
          const Real idz = 1.0 / coords.Dx(X3DIR, k, j, 0);
          const Real idz4 = 0.25 * idz;
          Real *J = &v(b, idx_J(n), k, j, 0);
          Real *Jjm1 = &v(b, idx_J(n), k, j - dj, 0);
          Real *Jjp1 = &v(b, idx_J(n), k, j + dj, 0);
          Real *Jkm1 = &v(b, idx_J(n), k - dk, j, 0);
          Real *Jkp1 = &v(b, idx_J(n), k + dk, j, 0);
          Real *Jkp1jm1 = &v(b, idx_J(n), k + dk, j - dj, 0);
          Real *Jkm1jm1 = &v(b, idx_J(n), k - dk, j - dj, 0);
          Real *Jkm1jp1 = &v(b, idx_J(n), k - dk, j + dj, 0);
          // x-direction faces
          Real *dJdx = &v(b, idx_dJ(n, 0, 0), k, j, 0);
          Real *dJdy = &v(b, idx_dJ(n, 1, 0), k, j, 0);
          Real *dJdz = &v(b, idx_dJ(n, 2, 0), k, j, 0);
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e + 1, [&](const int i) {
                dJdx[i] = (J[i] - J[i - 1]) * idx;
                dJdy[i] = (Jjp1[i] + Jjp1[i - 1] - Jjm1[i] - Jjm1[i - 1]) * idy4;
                dJdz[i] = (Jkp1[i] + Jkp1[i - 1] - Jkm1[i] - Jkm1[i - 1]) * idz4;
              });
          if (ndim > 1) {
            // y-direction faces
            dJdx = &v(b, idx_dJ(n, 0, 1), k, j, 0);
            dJdy = &v(b, idx_dJ(n, 1, 1), k, j, 0);
            dJdz = &v(b, idx_dJ(n, 2, 1), k, j, 0);
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e, [&](const int i) {
                  dJdx[i] = (J[i + 1] + Jjm1[i + 1] - J[i - 1] - Jjm1[i - 1]) * idx4;
                  dJdy[i] = (J[i] - Jjm1[i]) * idy;
                  dJdz[i] = (Jkp1[i] + Jkp1jm1[i] - Jkm1[i] - Jkm1jm1[i]) * idz4;
                });
          }
          if (ndim > 2) {
            // z-direction faces
            dJdx = &v(b, idx_dJ(n, 0, 2), k, j, 0);
            dJdy = &v(b, idx_dJ(n, 1, 2), k, j, 0);
            dJdz = &v(b, idx_dJ(n, 2, 2), k, j, 0);
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e, [&](const int i) {
                  dJdx[i] = (J[i + 1] + Jkm1[i + 1] - J[i - 1] - Jkm1[i - 1]) * idx4;
                  dJdy[i] = (Jjp1[i] + Jkm1jp1[i] - Jjm1[i] - Jkm1jm1[i]) * idy4;
                  dJdz[i] = (J[i] - Jkm1[i]) * idz;
                });
          }
        }
      });

  return TaskStatus::complete;
}
template TaskStatus ReconstructEdgeStates<MeshBlockData<Real>>(MeshBlockData<Real> *);

// This really only works for MeshBlockData right now since fluxes don't have a block
// index
template <class T, class CLOSURE>
TaskStatus CalculateFluxesImpl(T *rc) {
  printf("%s:%i CalculateFluxesImpl\n", __FILE__, __LINE__);
  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = (pmb->pmy_mesh->ndim > 0 ? 1 : 0);

  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = (pmb->pmy_mesh->ndim > 1 ? 1 : 0);

  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = (pmb->pmy_mesh->ndim > 2 ? 1 : 0);

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;

  PackIndexMap imap_ql, imap_qr, imap;
  std::vector<std::string> vars{ir::ql, ir::qr, ir::ql_v, ir::qr_v, ir::dJ, ir::kappaH};
  std::vector<std::string> flxs{cr::E, cr::F};

  auto v = rc->PackVariablesAndFluxes(vars, flxs, imap);

  auto idx_qlv = imap.GetFlatIdx(ir::ql_v);
  auto idx_qrv = imap.GetFlatIdx(ir::qr_v);
  auto idx_ql = imap.GetFlatIdx(ir::ql);
  auto idx_qr = imap.GetFlatIdx(ir::qr);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);

  auto idx_Ff = imap.GetFlatIdx(cr::F);
  auto idx_Ef = imap.GetFlatIdx(cr::E);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  // const int nblock = 1; //v.GetDim(5);

  auto geom = Geometry::GetCoordinateSystem(rc);

  const Real kappaH_min = 1.e-20;

  auto &coords = pmb->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::Fluxes", DevExecSpace(), X1DIR,
      pmb->pmy_mesh->ndim, // Loop over directions
      // 0, nblock-1, // Loop over reconstructed variables
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
        case (0):
          face = CellLocation::Face1;
          break;
        case (1):
          face = CellLocation::Face2;
          break;
        case (2):
          face = CellLocation::Face3;
          break;
        }

        Vec con_beta;
        Tens2 cov_gamma;
        geom.Metric(face, k, j, i, cov_gamma.data);
        geom.ContravariantShift(face, k, j, i, con_beta.data);
        const Real sdetgam = geom.DetGamma(face, k, j, i);
        typename CLOSURE::LocalGeometryType g(geom, face, 0, k, j, i);

        const Real dx = coords.Dx(idir_in, k, j, i) * sqrt(cov_gamma(idir, idir));

        for (int ispec = 0; ispec < num_species; ++ispec) {

          const Real &Jl = v(idx_ql(ispec, 0, idir), k, j, i);
          const Real &Jr = v(idx_qr(ispec, 0, idir), k, j, i);
          const Vec Hl = {Jl * v(idx_ql(ispec, 1, idir), k, j, i),
                          Jl * v(idx_ql(ispec, 2, idir), k, j, i),
                          Jl * v(idx_ql(ispec, 3, idir), k, j, i)};
          const Vec Hr = {Jr * v(idx_qr(ispec, 1, idir), k, j, i),
                          Jr * v(idx_qr(ispec, 2, idir), k, j, i),
                          Jr * v(idx_qr(ispec, 3, idir), k, j, i)};

          Vec con_vl{{v(idx_qlv(0, idir), k, j, i), v(idx_qlv(1, idir), k, j, i),
                      v(idx_qlv(2, idir), k, j, i)}};
          Vec con_vr{{v(idx_qrv(0, idir), k, j, i), v(idx_qrv(1, idir), k, j, i),
                      v(idx_qrv(2, idir), k, j, i)}};
          if (i == 4) {
            printf("Jl: %e Jr: %e\n", Jl, Jr);
            printf("Hl: %e %e %e Hr: %e %e %e\n", Hl(0), Hl(1), Hl(2),
              Hr(0), Hr(1), Hr(2));
            }

          Vec cov_dJ{{v(idx_dJ(ispec, 0, idir), k, j, i),
                      v(idx_dJ(ispec, 1, idir), k, j, i),
                      v(idx_dJ(ispec, 2, idir), k, j, i)}};

          // Calculate the geometric mean of the opacity on either side of the interface,
          // this is necessary for handling the asymptotic limit near sharp surfaces
          Real kappaH = sqrt((v(idx_kappaH(ispec), k, j, i) *
                              v(idx_kappaH(ispec), k - koff, j - joff, i - ioff)));

          const Real a = tanh(ratio(1.0, std::pow(std::abs(kappaH * dx), 1)));

          // Calculate the observer frame quantities on either side of the interface
          /// TODO: (LFR) Add other contributions to the asymptotic flux
          Vec HasymL = -cov_dJ / (3 * kappaH + 3 * kappaH_min);
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
          if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            // Use reconstructed values of tilPi
            SPACELOOP2(ii, jj) {
              con_tilPil(ii, jj) = v(idx_ql(ispec, 4 + ii + 3 * jj, idir), k, j, i);
              con_tilPir(ii, jj) = v(idx_qr(ispec, 4 + ii + 3 * jj, idir), k, j, i);
            }
          } else {
            cl.GetCovTilPiFromPrim(Jl, HasymL, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, HasymR, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, HasymL, con_tilPil, &conFl_asym, &Pl_asym);
          cr.getFluxesFromPrim(Jr, HasymR, con_tilPir, &conFr_asym, &Pr_asym);

          // Regular fluxes
          if (!programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            // Recalculate Eddington if using J, H
            cl.GetCovTilPiFromPrim(Jl, Hl, &con_tilPil);
            cr.GetCovTilPiFromPrim(Jr, Hr, &con_tilPir);
          }
          cl.getFluxesFromPrim(Jl, Hl, con_tilPil, &conFl, &Pl);
          cr.getFluxesFromPrim(Jr, Hr, con_tilPir, &conFr, &Pr);
          cl.Prim2Con(Jl, Hl, con_tilPil, &El, &covFl);
          cr.Prim2Con(Jr, Hr, con_tilPir, &Er, &covFr);

          // Mix the fluxes by the Peclet number
          // TODO: (LFR) Make better choices
          const Real speed = a * 1.0 + (1 - a) * std::max(sqrt(cl.v2), sqrt(cr.v2));
          conFl = a * conFl + (1 - a) * conFl_asym;
          conFr = a * conFr + (1 - a) * conFr_asym;
          Pl = a * Pl + (1 - a) * Pl_asym;
          Pr = a * Pr + (1 - a) * Pr_asym;

          // Correct the fluxes with the shift terms
          conFl(idir) -= con_beta(idir) * El;
          conFr(idir) -= con_beta(idir) * Er;

          SPACELOOP(ii) {
            Pl(idir, ii) -= con_beta(idir) * covFl(ii);
            Pr(idir, ii) -= con_beta(idir) * covFr(ii);
          }

          // Calculate the numerical flux using LLF
          v.flux(idir_in, idx_Ef(ispec), k, j, i) =
              0.5 * sdetgam * (conFl(idir) + conFr(idir) + speed * (El - Er));

          SPACELOOP(ii) {
            v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) =
                0.5 * sdetgam *
                (Pl(idir, ii) + Pr(idir, ii) + speed * (covFl(ii) - covFr(ii)));
          }
          if (i == 4) {
            printf("[%i %i %i][%i][dir: %i] F = %e %e %e %e\n",
            k,j,i,ispec,idir_in,v.flux(idir_in, idx_Ef(ispec), k, j, i),
            v.flux(idir_in, idx_Ff(ispec, 0), k, j, i),
            v.flux(idir_in, idx_Ff(ispec, 1), k, j, i),
            v.flux(idir_in, idx_Ff(ispec, 2), k, j, i));
          }
          if (sdetgam < std::numeric_limits<Real>::min() * 10) {
            v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.0;
            SPACELOOP(ii) v.flux(idir_in, idx_Ff(ispec, ii), k, j, i) = 0.0;
          }
        }
      });

  return TaskStatus::complete;
}

template <class T>
TaskStatus CalculateFluxes(T *rc) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateFluxesImpl<T, ClosureM1<Vec, Tens2, settings>>(rc);
  } else if (method == "moment_eddington") {
    return CalculateFluxesImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc);
  } else if (method == "mocmc") {
    return CalculateFluxesImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateFluxes<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T, class CLOSURE>
TaskStatus CalculateGeometricSourceImpl(T *rc, T *rc_src) {
  constexpr int ND = Geometry::NDFULL;
  // constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace p = fluid_prim;
  PackIndexMap imap;
  std::vector<std::string> vars{cr::E, cr::F, pr::J, pr::H, p::velocity};
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
    vars.push_back(ir::tilPi);
  }
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E);
  auto idx_F = imap.GetFlatIdx(cr::F);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_H = imap.GetFlatIdx(pr::H);
  auto pv = imap.GetFlatIdx(p::velocity);
  vpack_types::FlatIdx iTilPi({-1}, -1);
  if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
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

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::GeometricSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        // Set up the background state
        Vec con_v{{v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
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
        SPACELOOP2(ii, jj) beta2 += con_beta(ii) * con_beta(jj) * cov_gamma(ii, jj);

        Real dlnalp[ND];
        Real Gamma[ND][ND][ND];
        geom.GradLnAlpha(CellLocation::Cent, iblock, k, j, i, dlnalp);
        geom.ConnectionCoefficient(CellLocation::Cent, iblock, k, j, i, Gamma);

        // Get the gradient of the shift from the Christoffel symbols of the first kind
        // Get the extrinsic curvature from the Christoffel symbols of the first kind
        // All indices are covariant
        Tens2 dbeta, K;
        const Real iFac = 1.0 / (alp + beta2 / alp);
        SPACELOOP2(ii, jj) {
          dbeta(ii, jj) = Gamma[ii + 1][jj + 1][0] + Gamma[ii + 1][0][jj + 1];
          K(ii, jj) = Gamma[ii + 1][0][jj + 1];
          SPACELOOP(kk) K(ii, jj) -= Gamma[ii + 1][kk + 1][jj + 1] * con_beta(kk);
          K(ii, jj) *= iFac;
        }

        for (int ispec = 0; ispec < nspec; ++ispec) {
          Real E = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
          Real J = v(iblock, idx_J(ispec), k, j, i);
          Vec covF{{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                    v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                    v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam}};
          Vec covH{{J * v(iblock, idx_H(ispec, 0), k, j, i),
                    J * v(iblock, idx_H(ispec, 1), k, j, i),
                    J * v(iblock, idx_H(ispec, 2), k, j, i)}};
          Vec conF;
          g.raise3Vector(covF, &conF);
          Tens2 conP, con_tilPi;

          if (programming::is_specialization_of<CLOSURE, ClosureMOCMC>::value) {
            SPACELOOP2(ii, jj) {
              con_tilPi(ii, jj) = v(iblock, iTilPi(ispec, ii, jj), k, j, i);
            }
          } else {
            c.GetCovTilPiFromPrim(J, covH, &con_tilPi);
          }
          c.getConPFromPrim(J, covH, con_tilPi, &conP);

          Real srcE = 0.0;
          SPACELOOP2(ii, jj) srcE += K(ii, jj) * conP(ii, jj);
          SPACELOOP(ii) srcE -= dlnalp[ii + 1] * conF(ii);
          srcE *= alp;

          Vec srcF{0, 0, 0};
          SPACELOOP(ii) {
            SPACELOOP(jj) srcF(ii) += covF(jj) * dbeta(ii, jj);
            srcF(ii) -= alp * E * dlnalp[ii + 1];
            SPACELOOP2(jj, kk)
            srcF(ii) += alp * conP(jj, kk) * Gamma[jj + 1][kk + 1][ii + 1];
          }
          v_src(iblock, idx_E_src(ispec), k, j, i) = sdetgam * srcE;
          SPACELOOP(ii) v_src(iblock, idx_F_src(ispec, ii), k, j, i) = sdetgam * srcF(ii);
        }
      });
  return TaskStatus::complete;
}
template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src) {
  auto *pm = rc->GetParentPointer().get();
  StateDescriptor *rad = pm->packages.Get("radiation").get();
  auto method = rad->Param<std::string>("method");
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return CalculateGeometricSourceImpl<T, ClosureM1<Vec, Tens2, settings>>(rc, rc_src);
  } else if (method == "moment_eddington") {
    return CalculateGeometricSourceImpl<T, ClosureEdd<Vec, Tens2, settings>>(rc, rc_src);
  } else if (method == "mocmc") {
    return CalculateGeometricSourceImpl<T, ClosureMOCMC<Vec, Tens2, settings>>(rc,
                                                                               rc_src);
  } else {
    PARTHENON_FAIL("Radiation method unknown!");
  }
  return TaskStatus::fail;
}
template TaskStatus CalculateGeometricSource<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                                  MeshBlockData<Real> *);

// TODO(BRR) Unsure how else to get radiation parameters from MeshData
template <>
TaskStatus MomentFluidSource(MeshData<Real> *rc, Real dt, bool update_fluid) {
  for (int n = 0; n < rc->NumBlocks(); n++) {
    MomentFluidSource(rc->GetBlockData(n).get(), dt, update_fluid);
  }
  return TaskStatus::complete;
}

template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) {
  auto *pmb = rc->GetParentPointer().get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  namespace cr = radmoment_cons;
  namespace pr = radmoment_prim;
  namespace ir = radmoment_internal;
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E,      cr::F,       p::density, p::temperature,
                                p::ye,      p::velocity, pr::J,      pr::H,
                                ir::kappaJ, ir::kappaH,  ir::JBB};
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
  // TODO(BRR) This updates all neutrino species (including contributions to fluid)
  // regardless of whether they are active
  // int nspec = idx_E.DimSize(1);

  auto species = rad->Param<std::vector<RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");
  RadiationType species_d[3] = {};
  for (int s = 0; s < num_species; s++) {
    species_d[s] = species[s];
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        for (int ispec = 0; ispec < num_species; ++ispec) {

          // Set up the background state
          Vec con_v{{v(iblock, pv(0), k, j, i), v(iblock, pv(1), k, j, i),
                     v(iblock, pv(2), k, j, i)}};
          Tens2 cov_gamma;
          geom.Metric(CellLocation::Cent, iblock, k, j, i, cov_gamma.data);
          Real alpha = geom.Lapse(CellLocation::Cent, iblock, k, j, i);
          Real sdetgam = geom.DetGamma(CellLocation::Cent, iblock, k, j, i);
          LocalThreeGeometry g(geom, CellLocation::Cent, iblock, k, j, i);

          /// TODO: (LFR) Move beyond Eddington for this update
          ClosureEdd<Vec, Tens2> c(con_v, &g);

          Real Estar = v(iblock, idx_E(ispec), k, j, i) / sdetgam;
          Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i) / sdetgam,
                        v(iblock, idx_F(ispec, 1), k, j, i) / sdetgam,
                        v(iblock, idx_F(ispec, 2), k, j, i) / sdetgam};

          Real dE;
          Vec cov_dF;

          // Treat the Eddington tensor explicitly for now
          Real &J = v(iblock, idx_J(ispec), k, j, i);
          Vec cov_H{{
              J * v(iblock, idx_H(ispec, 0), k, j, i),
              J * v(iblock, idx_H(ispec, 1), k, j, i),
              J * v(iblock, idx_H(ispec, 2), k, j, i),
          }};
          Tens2 con_tilPi;

          c.GetCovTilPiFromPrim(J, cov_H, &con_tilPi);

          Real B = v(iblock, idx_JBB(ispec), k, j, i);
          Real tauJ = alpha * dt * v(iblock, idx_kappaJ(ispec), k, j, i);
          Real tauH = alpha * dt * v(iblock, idx_kappaH(ispec), k, j, i);
          Real kappaH = v(iblock, idx_kappaH(ispec), k, j, i);
          c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, B, tauJ, tauH, &dE, &cov_dF);

          // Add source corrections to conserved iration variables
          v(iblock, idx_E(ispec), k, j, i) += sdetgam * dE;
          for (int idir = 0; idir < 3; ++idir) {
            v(iblock, idx_F(ispec, idir), k, j, i) += sdetgam * cov_dF(idir);
          }

          // Add source corrections to conserved fluid variables
          if (update_fluid) {
            if (cye > 0) {
              v(iblock, cye, k, j, i) -= sdetgam * 0.0;
            }
            v(iblock, ceng, k, j, i) -= sdetgam * dE;
            v(iblock, cmom_lo + 0, k, j, i) -= sdetgam * cov_dF(0);
            v(iblock, cmom_lo + 1, k, j, i) -= sdetgam * cov_dF(1);
            v(iblock, cmom_lo + 2, k, j, i) -= sdetgam * cov_dF(2);
          }
        }
      });

  return TaskStatus::complete;
}
// template TaskStatus MomentFluidSource<MeshData<Real>>(MeshData<Real> *, Real, bool);
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real,
                                                           bool);

template <class T>
TaskStatus MomentCalculateOpacities(T *rc) {
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
  std::vector<std::string> vars{p::density, p::temperature, p::ye,  p::velocity,
                                ir::kappaJ, ir::kappaH,     ir::JBB};

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

  // Mainly for testing purposes, probably should be able to do this with the opacity code
  // itself
  const auto scattering_fraction = rad->Param<Real>("scattering_fraction");

  // Get the device opacity object
  using namespace singularity::neutrinos;
  const auto &d_opacity = opac->Param<Opacity>("d.opacity");
  const auto &d_mean_opacity = opac->Param<MeanOpacity>("d.mean_opacity");

  // Get the background geometry
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5);
  int nspec = idx_kappaJ.DimSize(1);

  /// TODO: (LFR) Fix this junk
  RadiationType dev_species[3] = {species[0], species[1], species[2]};

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RadMoments::FluidSource", DevExecSpace(), 0,
      nblock - 1, // Loop over blocks
      kb.s, kb.e, // z-loop
      jb.s, jb.e, // y-loop
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int iblock, const int k, const int j, const int i) {
        for (int ispec = 0; ispec < nspec; ++ispec) {
          const Real rho = v(iblock, prho, k, j, i);
          const Real Temp = v(iblock, pT, k, j, i);
          const Real Ye = pYe > 0 ? v(iblock, pYe, k, j, i) : 0.5;

          Real kappa = d_mean_opacity.RosselandMeanAbsorptionCoefficient(
              rho, Temp, Ye, dev_species[ispec]);
          Real JBB = d_opacity.EnergyDensityFromTemperature(Temp, dev_species[ispec]);

          v(iblock, idx_JBB(ispec), k, j, i) = JBB;
          v(iblock, idx_kappaJ(ispec), k, j, i) = kappa * (1.0 - scattering_fraction);
          v(iblock, idx_kappaH(ispec), k, j, i) = kappa;
        }
      });

  return TaskStatus::complete;
}
template TaskStatus MomentCalculateOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *);
} // namespace radiation
