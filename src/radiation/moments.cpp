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
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

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
        /// TODO: (LFR) Replace this placeholder zero velocity con2prim 
        v(b, cJ(ispec), k, j, i) = v(b, cE(ispec), k, j, i);
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          v(b, cH(ispec, idir), k, j, i) = v(b, cF(ispec, idir), k, j, i);
        }
      });

  return TaskStatus::complete;
}
//template TaskStatus MomentCon2Prim<MeshData<Real>>(MeshData<Real> *);
template TaskStatus MomentCon2Prim<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T>
TaskStatus MomentPrim2Con(T* rc, IndexDomain domain) { 
  
  namespace c = radmoment_cons;  
  namespace p = radmoment_prim;  
  namespace i = radmoment_internal;  
  
  auto *pm = rc->GetParentPointer().get(); 
  
  IndexRange ib = pm->cellbounds.GetBoundsI(domain);
  IndexRange jb = pm->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pm->cellbounds.GetBoundsK(domain);

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
        /// TODO: (LFR) Replace this placeholder zero velocity prim2con 
        v(b, cE(ispec), k, j, i) = v(b, cJ(ispec), k, j, i);
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { 
          v(b, cF(ispec, idir), k, j, i) = v(b, cH(ispec, idir), k, j, i);
        }
      });

  return TaskStatus::complete;
}
//template TaskStatus MomentPrim2Con<MeshData<Real>>(MeshData<Real> *, IndexDomain);
template TaskStatus MomentPrim2Con<MeshBlockData<Real>>(MeshBlockData<Real> *, IndexDomain);

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
  VariablePack<Real> ql_base = rc->PackVariables(std::vector<std::string>{i::ql}, imap_ql); 
  VariablePack<Real> qr_base = rc->PackVariables(std::vector<std::string>{i::qr}, imap_qr); 
  VariablePack<Real> v = rc->PackVariables(std::vector<std::string>{p::J, p::H}, imap) ;
  
  auto qIdx = imap_ql.GetFlatIdx(i::ql);
  
  const int nspec = qIdx.DimSize(1);
  const int nrecon = 4*nspec;

  const int offset = imap_ql[i::ql].first; 
  
  const int nblock = ql_base.GetDim(5); 
  
  //PARTHENON_REQUIRE(nrecon == v.GetDim(4), "Issue with number of reconstruction variables in moments.");
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(), 
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions for reconstruction
      0, nblock-1, // Loop over blocks
      kb.s - dk, kb.e + dk, // z-loop  
      jb.s - dj, jb.e + dj, // y-loop 
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir, const int b, const int k, const int j, const int i) { 
        ReconstructionWrapper<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionWrapper<VariablePack<Real>> qr(qr_base, nrecon, offset, b);
        for (int ivar = 0; ivar<nrecon; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(idir, ivar, k, j, i, v, ql, qr);
        }
        /* Piecewise constant for simple tests
        const int di = (idir == X1DIR ? 1 : 0);
        const int dj = (idir == X2DIR ? 1 : 0);
        const int dk = (idir == X3DIR ? 1 : 0);
        
        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {
          ql_base(b, qIdx(0, ispec, idir-1), k+dk, j+dj, i+di) = v(b, idxJ(ispec), k, j, i); 
          ql_base(b, qIdx(1, ispec, idir-1), k+dk, j+dj, i+di) = v(b, idxH(0, ispec), k, j, i); 
          ql_base(b, qIdx(2, ispec, idir-1), k+dk, j+dj, i+di) = v(b, idxH(1, ispec), k, j, i); 
          ql_base(b, qIdx(3, ispec, idir-1), k+dk, j+dj, i+di) = v(b, idxH(2, ispec), k, j, i); 
          
          qr_base(b, qIdx(0, ispec, idir-1), k, j, i) = v(b, idxJ(ispec), k, j, i); 
          qr_base(b, qIdx(1, ispec, idir-1), k, j, i) = v(b, idxH(0, ispec), k, j, i); 
          qr_base(b, qIdx(2, ispec, idir-1), k, j, i) = v(b, idxH(1, ispec), k, j, i); 
          qr_base(b, qIdx(3, ispec, idir-1), k, j, i) = v(b, idxH(2, ispec), k, j, i); 
        }
        */
      });
  return TaskStatus::complete;  
}
template TaskStatus ReconstructEdgeStates<MeshBlockData<Real>>(MeshBlockData<Real> *);

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
  auto v = rc->PackVariablesAndFluxes(std::vector<std::string>{i::ql, i::qr}, 
          std::vector<std::string>{c::E, c::F}, imap) ;
  
  auto idx_ql = imap.GetFlatIdx(i::ql);
  auto idx_qr = imap.GetFlatIdx(i::qr);
  auto idx_Ff = imap.GetFlatIdx(c::F);
  auto idx_Ef = imap.GetFlatIdx(c::E);
  
  const int nspec = idx_ql.DimSize(1);

  const int nblock = 1; //v.GetDim(5); 
  
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Fluxes", DevExecSpace(), 
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions 
      //0, nblock-1, // Loop over reconstructed variables
      kb.s - dk, kb.e + dk, // z-loop  
      jb.s - dj, jb.e + dj, // y-loop 
      ib.s - di, ib.e + di, // x-loop
      KOKKOS_LAMBDA(const int idir_in, const int k, const int j, const int i) { 
        for (int ispec = 0; ispec<nspec; ++ispec) { 
          /// TODO: (LFR) Fix indexing so everything starts on a consistent index
          const int idir = idir_in - 1; 
        
          const Real& Jl = v(idx_ql(ispec, 0, idir), k, j, i);
          const Real& Jr = v(idx_qr(ispec, 0, idir), k, j, i);
          const Real Hl[3] = {v(idx_ql(ispec, 1, idir), k, j, i), 
                              v(idx_ql(ispec, 2, idir), k, j, i), 
                              v(idx_ql(ispec, 3, idir), k, j, i)};
          const Real Hr[3] = {v(idx_qr(ispec, 1, idir), k, j, i), 
                              v(idx_qr(ispec, 2, idir), k, j, i), 
                              v(idx_qr(ispec, 3, idir), k, j, i)}; 

          /// TODO: (LFR) This should all get replaced with real flux calculation, be careful about densitization 
          v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.5*(Hl[idir] + Hr[idir]) + (Jl - Jr); 
        
          v.flux(idir_in, idx_Ff(ispec, 0), k, j, i) = (Hl[0] - Hr[0]); 
          v.flux(idir_in, idx_Ff(ispec, 1), k, j, i) = (Hl[1] - Hr[1]); 
          v.flux(idir_in, idx_Ff(ispec, 2), k, j, i) = (Hl[2] - Hr[2]);
          v.flux(idir_in, idx_Ff(ispec, idir), k, j, i) += 0.5*(Jl/3.0 + Jr/3.0);
        } 
      });

  return TaskStatus::complete;  
}
template TaskStatus CalculateFluxes<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src) { 

  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  auto *pmb = rc->GetParentPointer().get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto geom = Geometry::GetCoordinateSystem(rc);
  /// TODO: (LFR) Actually build the source 
  return TaskStatus::complete;
}
template TaskStatus CalculateGeometricSource<MeshBlockData<Real>>(MeshBlockData<Real> *, MeshBlockData<Real> *);

template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid) { 
  
  auto *pmb = rc->GetParentPointer().get();
   
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real DENSITY = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMPERATURE = unit_conv.GetTemperatureCodeToCGS();

  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  
  namespace cr = radmoment_cons;  
  namespace pr = radmoment_prim;  
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E, cr::F, p::density, p::temperature, p::ye}; 
  if (update_fluid) {
    vars.push_back(c::energy);
    vars.push_back(c::momentum);
    vars.push_back(c::ye);
  }

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E); 
  auto idx_F = imap.GetFlatIdx(cr::F);

  int prho = imap[p::density].first; 
  int pT = imap[p::temperature].first; 
  int pYe = imap[p::ye].first; 

  int ceng(-1), cmom_lo(-1), cmom_hi(-1), cy(-1); 
  if (update_fluid) { 
    int ceng = imap[c::energy].first;
    int cmom_lo = imap[c::momentum].first;
    int cmom_hi = imap[c::momentum].second;
    int cye = imap[c::ye].first;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  /// TODO: (LFR) Couple this to singularity-opac  
  const auto B_fake = rad->Param<Real>("B_fake");
  const auto use_B_fake = rad->Param<bool>("use_B_fake");

  // Get the device opacity object
  using namespace singularity::neutrinos; 
  const auto d_opacity = opac->Param<Opacity>("d.opacity");
  
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
          /// TODO: (LFR) Need to make a grid variable holding the energy integrated opacity so that we can 
          ///             create a task to fill the opacity based on MoCMC or some other rule.
          const Real enu = 10.0; // Assume we are gray for now or can take the peak opacity at enu = 10 MeV 
          const Real rho_cgs =  v(iblock, prho, k, j, i) * DENSITY;
          const Real T_cgs =  v(iblock, pT, k, j, i) * TEMPERATURE;
          const Real Ye = v(iblock, pYe, k, j, i);

          Real kappa = d_opacity.AbsorptionCoefficientPerNu(rho_cgs, T_cgs, Ye, species[ispec], enu);
          const Real emis = d_opacity.Emissivity(rho_cgs, T_cgs, Ye, species[ispec]); 
          Real B = emis/kappa; 
          if (use_B_fake) B = B_fake; 
          kappa = 1.0;
          B = 0.5; 

          // This will be replaced with the rest frame calculation
          const Real lam = (kappa*dt)/(1 + kappa*dt);
          const Real dE = (B - v(iblock, idx_E(ispec), k, j, i))*lam;
          const Real dF[3] = {-v(iblock, idx_F(ispec, 0), k, j, i)*lam,
                              -v(iblock, idx_F(ispec, 1), k, j, i)*lam,
                              -v(iblock, idx_F(ispec, 2), k, j, i)*lam};

          // Add source corrections to conserved radiation variables
          v(iblock, idx_E(ispec), k, j, i) += dE; 
          for (int idir=0; idir<3; ++idir) {
            v(iblock, idx_F(ispec, idir), k, j, i) += dF[idir];
          }
          
          // Add source corrections to conserved fluid variables 
          if (update_fluid) {
            v(iblock, ceng, k, j, i) -= dE; 
            v(iblock, cmom_lo + 0, k, j, i) -= dF[0]; 
            v(iblock, cmom_lo + 1, k, j, i) -= dF[1]; 
            v(iblock, cmom_lo + 2, k, j, i) -= dF[2]; 
          }

        } 
      });

  return TaskStatus::complete;
}
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real, bool);
} //namespace radiationMoments
