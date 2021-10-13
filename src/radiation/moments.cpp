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
#include "closure.hpp"

namespace radiation {

struct Vec { 
  Real data[Geometry::NDSPACE]; 
  KOKKOS_FORCEINLINE_FUNCTION
  Real& operator()(const int idx){return data[idx];}
  KOKKOS_FORCEINLINE_FUNCTION
  const Real& operator()(const int idx) const {return data[idx];}
};

struct Tens2 { 
  Real data[Geometry::NDSPACE][Geometry::NDSPACE]; 
  KOKKOS_FORCEINLINE_FUNCTION 
  Real& operator()(const int i, const int j){return data[i][j];} 
  KOKKOS_FORCEINLINE_FUNCTION
  const Real& operator()(const int i, const int j) const {return data[i][j];} 
};

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
  auto pJ = imap.GetFlatIdx(p::J);  
  auto cF = imap.GetFlatIdx(c::F);  
  auto pH = imap.GetFlatIdx(p::H);  
  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Prim2Con", DevExecSpace(), 
      0, v.GetDim(5)-1, // Loop over meshblocks
      specB.s, specB.e, // Loop over species 
      kb.s, kb.e, // z-loop  
      jb.s, jb.e, // y-loop 
      ib.s, ib.e, // x-loop
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) { 
        /// TODO: (LFR) Replace this placeholder zero velocity prim2con 
        v(b, cE(ispec), k, j, i) = v(b, pJ(ispec), k, j, i);
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { 
          v(b, cF(ispec, idir), k, j, i) = v(b, pH(ispec, idir), k, j, i);
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
          const Vec Hl = {v(idx_ql(ispec, 1, idir), k, j, i), 
                              v(idx_ql(ispec, 2, idir), k, j, i), 
                              v(idx_ql(ispec, 3, idir), k, j, i)};
          const Vec Hr = {v(idx_qr(ispec, 1, idir), k, j, i), 
                              v(idx_qr(ispec, 2, idir), k, j, i), 
                              v(idx_qr(ispec, 3, idir), k, j, i)}; 
          
          const Real speed = 1.0;
          Tens2 con_tilPi;
          
          /// TODO: (LFR) need to pull out actual values for these 
          Vec con_vr{{0,0,0}};
          Vec con_vl{{0,0,0}};
          Tens2 cov_gamma{{{1,0,0},{0,1,0},{0,0,1}}};

          
          // Calculate the observer frame quantities on the left side of the interface  
          Closure<Vec, Tens2> cl(con_vl, cov_gamma); 
          Real El; 
          Vec covFl, conFl;
          Tens2 Pl;
          cl.Prim2ConM1(Jl, Hl, &El, &covFl, &con_tilPi);
          cl.raise3Vector(covFl, &conFl);
          cl.getConCovPFromPrim(Jl, Hl, con_tilPi, &Pl);
          
          // Calculate the observer frame quantities on the right side of the interface  
          Closure<Vec, Tens2> cr(con_vr, cov_gamma); 
          Real Er; 
          Vec covFr, conFr;
          Tens2 Pr;
          cr.Prim2ConM1(Jr, Hr, &Er, &covFr, &con_tilPi);
          cr.raise3Vector(covFr, &conFr);
          cr.getConCovPFromPrim(Jr, Hr, con_tilPi, &Pr);

          // Everything below should be independent of the assumed closure, just calculating the LLF flux
          /// TODO: (LFR) Include diffusion limit   
          v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.5*(conFl(idir) + conFr(idir)) + speed*(El - Er); 
        
          v.flux(idir_in, idx_Ff(ispec, 0), k, j, i) = 0.5*(Pl(idir, 0) + Pr(idir, 0)) 
                                                       + speed*(covFl(0) - covFr(0));

          v.flux(idir_in, idx_Ff(ispec, 1), k, j, i) = 0.5*(Pl(idir, 1) + Pr(idir, 1)) 
                                                       + speed*(covFl(1) - covFr(1)); 

          v.flux(idir_in, idx_Ff(ispec, 2), k, j, i) = 0.5*(Pl(idir, 2) + Pr(idir, 2)) 
                                                       + speed*(covFl(2) - covFr(2));
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
