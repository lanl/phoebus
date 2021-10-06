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
  
  auto *pm = rc->GetParentPoint().get(); 
  
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

// TODO(LFR): Add geometric source term task 

// TODO(LFR): Add implicit source term task (Trickiest, since there is no path to follow)

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
  
  const int nrecon = qIdx.DimSize(1); 
  const int nblock = ql_base.GetDim(5); 
  const int offset = qIdx.DimSize(1)*qIdx.DimSize(2);
  
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

// TODO(LFR): Add calculate flux task 

} //namespace radiationMoments