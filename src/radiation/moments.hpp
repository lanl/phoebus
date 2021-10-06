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

#ifndef RADIATION_MOMENTS_HPP_
#define RADIATION_MOMENTS_HPP_

#include "radiation.hpp" 
#include "reconstruction.hpp"
#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

namespace radiation {

template <class T> 
  class ReconstructionWrapper  {
 public:
  KOKKOS_INLINE_FUNCTION
  ReconstructionWrapper(const T& v, const int chunk_size, const int offset, const int block = 0) : 
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

template <class T>
TaskStatus MomentCon2Prim(T* rc) { 
  
  namespace c = radmoment_cons;  
  namespace p = radmoment_prim;  
  namespace i = radmoment_internal;  
  
  auto *pm = rc->GetParentPointer().get(); 
  
  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsI(IndexDomain::entire);

  PackIndexMap imap;
  auto v = rc->PackVariables(std::vector<std::string>{c::E, c::F, p::J, p::H}, imap);
  
  auto cE = imap.GetFlatIdx(c::E);  
  auto cJ = imap.GetFlatIdx(p::J);  
  auto cF = imap.GetFlatIdx(c::F);  
  auto cH = imap.GetFlatIdx(p::H);  
  auto specB = cE.GetBounds(1);
  auto dirB = cH.GetBounds(1);
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Con2Prim", DevExecSpace(), 
      0, v.GetDim(5)-1, // Loop over meshblocks
      specB.s, specB.e, // Loop over species 
      kb.s, kb.e, // z-loop  
      jb.s, jb.e, // y-loop 
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) { 
        // TODO: Replace this placeholder zero velocity con2prim 
        v(b, cJ(ispec), k, j, i) = v(b, cE(ispec), k, j, i);
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { 
          v(b, cH(idir, ispec), k, j, i) = v(b, cF(idir, ispec), k, j, i);
        }
      });

  return TaskStatus::complete;
}

template <class T> 
TaskStatus ReconstructEdgeStates(T* rc) {

  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = ( pmb->pmy_mesh->ndim > 0 ? 1 : 0);
  
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = ( pmb->pmy_mesh->ndim > 1 ? 1 : 0);
  
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = ( pmb->pmy_mesh->ndim > 2 ? 1 : 0);
  
  namespace c = radmoment_cons;  
  namespace p = radmoment_prim;  
  namespace i = radmoment_internal;  
  
  PackIndexMap imap_ql, imap_qr, imap;
  VariablePack<Real> ql_base = rc->PackVariables(std::vector<std::string>{i::ql}, imap_qr); 
  VariablePack<Real> qr_base = rc->PackVariables(std::vector<std::string>{i::qr}, imap_ql); 
  VariablePack<Real> v = rc->PackVariables(std::vector<std::string>{p::J, p::H}, imap) ;
  
  auto qIdx = imap_qr.GetFlatIdx(i::ql);
  
  const int nspec = qIdx.DimSize(2);
  const int nrecon = 4*nspec;

  const int offset = imap_ql[i::ql].first; 
  
  const int nblock = ql_base.GetDim(5); 
  
  PARTHENON_REQUIRE(nrecon == v.GetDim(4), "Issue with number of reconstruction variables in moments.");

  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(), 
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions for reconstruction
      0, nblock-1, // Loop over reconstructed variables
      kb.s - dk, kb.e + dk, // z-loop  
      jb.s - dj, jb.e + dj, // y-loop 
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir, const int b, const int k, const int j, const int i) { 
        ReconstructionWrapper<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionWrapper<VariablePack<Real>> qr(qr_base, nrecon, offset, b);
        for (int ivar = 0; ivar<nrecon; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(idir, ivar, k, j, i, v, ql, qr);
        }
      });
  return TaskStatus::complete;  
}

// This really only works for MeshBlockData right now since fluxes don't have a block index 
template <class T> 
TaskStatus CalculateFluxes(T* rc) {

  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const int di = ( pmb->pmy_mesh->ndim > 0 ? 1 : 0);
  
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const int dj = ( pmb->pmy_mesh->ndim > 1 ? 1 : 0);
  
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int dk = ( pmb->pmy_mesh->ndim > 2 ? 1 : 0);
  
  namespace c = radmoment_cons;  
  namespace p = radmoment_prim;  
  namespace i = radmoment_internal;  
  
  PackIndexMap imap_ql, imap_qr, imap;
  auto ql = rc->PackVariables(std::vector<std::string>{i::ql}, imap_qr); 
  auto qr = rc->PackVariables(std::vector<std::string>{i::qr}, imap_ql); 
  auto v = rc->PackVariablesAndFluxes(std::vector<std::string>{}, 
          std::vector<std::string>{c::E, c::F}, imap) ;
  
  auto idx_q = imap_qr.GetFlatIdx(i::ql);
  auto idx_Ef = imap.GetFlatIdx(c::E);
  auto idx_Ff = imap.GetFlatIdx(c::F);
  
  const int nspec = idx_q.DimSize(2);
  const int nrecon = 4*nspec;

  const int offset = imap_ql[i::ql].first; 
  
  const int nblock = ql.GetDim(5); 
  
  PARTHENON_REQUIRE(nrecon == v.GetDim(4), "Issue with number of reconstruction variables in moments.");
  
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Fluxes", DevExecSpace(), 
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions 
      //0, nblock-1, // Loop over reconstructed variables
      kb.s - dk, kb.e + dk, // z-loop  
      jb.s - dj, jb.e + dj, // y-loop 
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir_in, const int k, const int j, const int i) { 
        for (int ispec = 0; ispec<nspec; ++ispec) { 
          const int idir = idir_in - 1; // TODO (LFR): Fix indexing so everything starts on a consistent index
        
          const Real& Jl = ql(idx_q(0, ispec, idir), k, j, i);
          const Real& Jr = qr(idx_q(0, ispec, idir), k, j, i);
          const Real Hl[3] = {ql(idx_q(1, ispec, idir), k, j, i), 
                              ql(idx_q(2, ispec, idir), k, j, i), 
                              ql(idx_q(3, ispec, idir), k, j, i)};
          const Real Hr[3] = {qr(idx_q(1, ispec, idir), k, j, i), 
                              qr(idx_q(2, ispec, idir), k, j, i), 
                              qr(idx_q(3, ispec, idir), k, j, i)}; 

          // TODO (LFR): This should all get replaced with real flux calculation, be careful about densitization 
          v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.5*(Hl[idir] + Hr[idir]) + (Jl - Jr); 
        
          v.flux(idir_in, idx_Ff(0, ispec), k, j, i) = (Hl[0] - Hr[0]); 
          v.flux(idir_in, idx_Ff(1, ispec), k, j, i) = (Hl[1] - Hr[1]); 
          v.flux(idir_in, idx_Ff(2, ispec), k, j, i) = (Hl[2] - Hr[2]);
          v.flux(idir_in, idx_Ff(idir, ispec), k, j, i) += 0.5*(Jl/3.0 + Jr/3.0);
        } 
      });

  return TaskStatus::complete;  
}

template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src) { 

  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto geom = Geometry::GetCoordinateSystem(rc);
  // TODO (LFR): Actually build the tasks   
  return TaskStatus::complete;
}

// TODO(LFR): Add implicit source term task (Trickiest, since there is no path to follow)

} //namespace radiationMoments

#endif //RADIATION_MOMENTS_HPP_