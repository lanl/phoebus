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
  auto v = rc->PackVariables(std::vector<std::string>{c::E, c::F, p::J, p::H, fluid_prim::velocity, i::xi, i::phi}, imap);
  
  auto cE = imap.GetFlatIdx(c::E);  
  auto pJ = imap.GetFlatIdx(p::J);  
  auto cF = imap.GetFlatIdx(c::F);  
  auto pH = imap.GetFlatIdx(p::H); 
  auto iXi = imap.GetFlatIdx(i::xi); 
  auto iPhi = imap.GetFlatIdx(i::phi); 
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);
  auto specB = cE.GetBounds(1);
  auto dirB = pH.GetBounds(1);
  
  auto geom = Geometry::GetCoordinateSystem(rc);

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
        geom.Metric(CellLocation::Cent, k, j, i, cov_gamma.data);
        Closure<Vec, Tens2> c(con_v, cov_gamma); 
        
        Real J; 
        Vec covH;
        Tens2 conTilPi;
        Real E = v(b, cE(ispec), k, j, i);
        Vec covF ={{v(b, cF(ispec, 0), k, j, i), 
                    v(b, cF(ispec, 1), k, j, i), 
                    v(b, cF(ispec, 2), k, j, i)}};
        
        Real xi = v(b, iXi(), k, j, i);
        Real phi = 1.0001*v(b, iPhi(), k, j, i);

        auto result = c.Con2PrimM1(E, covF, xi, phi, &J, &covH, &conTilPi);
        
        v(b, iXi(), k, j, i) = result.xi;
        v(b, iPhi(), k, j, i) = result.phi;

        v(b, pJ(ispec), k, j, i) = J;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { // Loop over directions
          v(b, pH(ispec, idir), k, j, i) = covH(idir);
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
  auto v = rc->PackVariables(std::vector<std::string>{c::E, c::F, p::J, p::H, fluid_prim::velocity}, imap);
  
  auto cE = imap.GetFlatIdx(c::E);  
  auto pJ = imap.GetFlatIdx(p::J);  
  auto cF = imap.GetFlatIdx(c::F);  
  auto pH = imap.GetFlatIdx(p::H);  
  auto pv = imap.GetFlatIdx(fluid_prim::velocity);

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
        geom.Metric(CellLocation::Cent, k, j, i, cov_gamma.data);
        Closure<Vec, Tens2> c(con_v, cov_gamma); 
        
        Real E; 
        Vec covF;
        Tens2 conTilPi;
        Real J = v(b, pJ(ispec), k, j, i);
        Vec covH ={{v(b, pH(ispec, 0), k, j, i), 
                    v(b, pH(ispec, 1), k, j, i), 
                    v(b, pH(ispec, 2), k, j, i)}}; 
        c.Prim2ConM1(J, covH, &E, &covF, &conTilPi);

        v(b, cE(ispec), k, j, i) = E;
        for (int idir = dirB.s; idir <= dirB.e; ++idir) { 
          v(b, cF(ispec, idir), k, j, i) = covF(idir);
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
  VariablePack<Real> v = rc->PackVariables(std::vector<std::string>{p::J, p::H, i::dJ}, imap);
  auto idx_J = imap.GetFlatIdx(p::J);
  auto idx_dJ = imap.GetFlatIdx(i::dJ);

  ParArrayND<Real> ql_v = rc->Get(i::ql_v).data; 
  ParArrayND<Real> qr_v = rc->Get(i::qr_v).data; 
  VariablePack<Real> v_vel = rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
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
        ReconstructionIndexer<VariablePack<Real>> ql(ql_base, nrecon, offset, b);
        ReconstructionIndexer<VariablePack<Real>> qr(qr_base, nrecon, offset, b);
        // Reconstruct radiation
        for (int ivar = 0; ivar<nrecon; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(idir, ivar, k, j, i, v, ql, qr);
        }
        // Reconstruct velocity for radiation 
        for (int ivar = 0; ivar<3; ++ivar) {
          PhoebusReconstruction::PiecewiseLinear(idir, ivar, k, j, i, v_vel, ql_v, qr_v);
        }
        // Calculate spatial derivative of J at zone edges for diffusion limit  
        const int off_k = (idir == 3 ? 1 : 0);
        const int off_j = (idir == 2 ? 1 : 0);
        const int off_i = (idir == 1 ? 1 : 0);
        const Real dx = pmb->coords.Dx(idir, k, j, i);
        for (int ispec=0; ispec<nspec; ++ispec) {
          v(b, idx_dJ(ispec, idir-1), k, j, i) = (v(b, idx_J(ispec), k, j, i) 
                                                - v(b, idx_J(ispec), k-off_k, j-off_j, i-off_i))/dx; 
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
  auto v = rc->PackVariablesAndFluxes(std::vector<std::string>{i::ql, i::qr, i::ql_v, i::qr_v, i::dJ}, 
          std::vector<std::string>{c::E, c::F}, imap) ;
  
  auto idx_qlv = imap.GetFlatIdx(i::ql_v);
  auto idx_qrv = imap.GetFlatIdx(i::qr_v);
  auto idx_ql = imap.GetFlatIdx(i::ql);
  auto idx_qr = imap.GetFlatIdx(i::qr);
  auto idx_Ff = imap.GetFlatIdx(c::F);
  auto idx_Ef = imap.GetFlatIdx(c::E);
  auto idx_dJ = imap.GetFlatIdx(i::dJ);
  
  const int nspec = idx_ql.DimSize(1);

  const int nblock = 1; //v.GetDim(5); 
  
  auto geom = Geometry::GetCoordinateSystem(rc); 

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
          
          const Real speed = 1.0/sqrt(3.0);
          Tens2 con_tilPi;
          
          Vec con_vl{{v(idx_qlv(0, idir), k, j, i), 
                      v(idx_qlv(1, idir), k, j, i),
                      v(idx_qlv(2, idir), k, j, i)}};
          Vec con_vr{{v(idx_qrv(0, idir), k, j, i), 
                      v(idx_qrv(1, idir), k, j, i),
                      v(idx_qrv(2, idir), k, j, i)}};

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

          // Calculate the observer frame quantities on the left side of the interface  
          Closure<Vec, Tens2> cl(con_vl, cov_gamma); 
          Real El; 
          Vec covFl, conFl;
          Tens2 Pl; // P^i_j on the left side of the interface
          cl.Prim2ConM1(Jl, Hl, &El, &covFl, &con_tilPi);
          cl.raise3Vector(covFl, &conFl);
          cl.getConCovPFromPrim(Jl, Hl, con_tilPi, &Pl);
          
          // Calculate the observer frame quantities on the right side of the interface  
          Closure<Vec, Tens2> cr(con_vr, cov_gamma); 
          Real Er; 
          Vec covFr, conFr;
          Tens2 Pr; // P^i_j on the right side of the interface
          cr.Prim2ConM1(Jr, Hr, &Er, &covFr, &con_tilPi);
          cr.raise3Vector(covFr, &conFr);
          cr.getConCovPFromPrim(Jr, Hr, con_tilPi, &Pr);
          
          // Correct the fluxes with the shift terms 
          conFl(idir) -= con_beta(idir)*El;
          conFr(idir) -= con_beta(idir)*Er;
          
          Pl(idir, 0) -= con_beta(idir)*covFl(0);
          Pl(idir, 1) -= con_beta(idir)*covFl(1);
          Pl(idir, 2) -= con_beta(idir)*covFl(2);

          Pr(idir, 0) -= con_beta(idir)*covFr(0);
          Pr(idir, 1) -= con_beta(idir)*covFr(1);
          Pr(idir, 2) -= con_beta(idir)*covFr(2);

          // Everything below should be independent of the assumed closure, just calculating the LLF flux
          v.flux(idir_in, idx_Ef(ispec), k, j, i) = 0.5*(conFl(idir) + conFr(idir)) + speed*(El - Er); 
        
          v.flux(idir_in, idx_Ff(ispec, 0), k, j, i) = 0.5*(Pl(idir, 0) + Pr(idir, 0)) 
                                                       + speed*(covFl(0) - covFr(0));

          v.flux(idir_in, idx_Ff(ispec, 1), k, j, i) = 0.5*(Pl(idir, 1) + Pr(idir, 1)) 
                                                       + speed*(covFl(1) - covFr(1)); 

          v.flux(idir_in, idx_Ff(ispec, 2), k, j, i) = 0.5*(Pl(idir, 2) + Pr(idir, 2)) 
                                                       + speed*(covFl(2) - covFr(2));
        
          /// TODO: (LFR) Include diffusion limit  
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
  namespace ir = radmoment_internal;  
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E, cr::F, p::density, p::temperature, p::ye, p::velocity, 
                                ir::kappaJ, ir::kappaH, ir::JBB}; 
  if (update_fluid) {
    vars.push_back(c::energy);
    vars.push_back(c::momentum);
    vars.push_back(c::ye);
  }

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  auto idx_E = imap.GetFlatIdx(cr::E); 
  auto idx_F = imap.GetFlatIdx(cr::F);

  auto idx_kappaJ = imap.GetFlatIdx(ir::kappaJ);
  auto idx_kappaH = imap.GetFlatIdx(ir::kappaH);
  auto idx_JBB = imap.GetFlatIdx(ir::JBB);
  
  auto pv = imap.GetFlatIdx(p::velocity);

  int prho = imap[p::density].first; 
  int pT = imap[p::temperature].first; 
  int pYe = imap[p::ye].first; 

  int ceng(-1), cmom_lo(-1), cmom_hi(-1), cye(-1); 
  if (update_fluid) { 
    ceng = imap[c::energy].first;
    cmom_lo = imap[c::momentum].first;
    cmom_hi = imap[c::momentum].second;
    cye = imap[c::ye].first;
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
          /*
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
          */          

          // Set up the background state 
          Vec con_v{{v(iblock, pv(0), k, j, i),
                     v(iblock, pv(1), k, j, i),
                     v(iblock, pv(2), k, j, i)}};
          Tens2 cov_gamma; 
          geom.Metric(CellLocation::Cent, k, j, i, cov_gamma.data);
          Real alpha = geom.Lapse(CellLocation::Cent, k, j, i); 
          Closure<Vec, Tens2> c(con_v, cov_gamma); 
          
          Real Estar = v(iblock, idx_E(ispec), k, j, i); 
          Vec cov_Fstar{v(iblock, idx_F(ispec, 0), k, j, i),
                        v(iblock, idx_F(ispec, 1), k, j, i),
                        v(iblock, idx_F(ispec, 2), k, j, i)};
          
          Real dE;
          Vec cov_dF;

          /// TODO: (LFR) Move beyond Eddington for this update
          Tens2 con_tilPi{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};  
          
          Real B = v(iblock, idx_JBB(ispec), k, j, i); 
          Real tauJ = alpha*dt*v(iblock, idx_kappaJ(ispec), k, j, i);  
          Real tauH = alpha*dt*v(iblock, idx_kappaH(ispec), k, j, i);  
        
          c.LinearSourceUpdate(Estar, cov_Fstar, con_tilPi, B, 
                               tauJ, tauH, &dE, &cov_dF); 
          //printf("dE (new) = %e ", dE); 
          // This is the zero velocity limit for testing
          //const Real lam = (kappa*dt)/(1 + kappa*dt);
          //dE = (B - v(iblock, idx_E(ispec), k, j, i))*lam;
          //cov_dF(0) = -v(iblock, idx_F(ispec, 0), k, j, i)*lam;
          //cov_dF(0) = -v(iblock, idx_F(ispec, 1), k, j, i)*lam;
          //cov_dF(0) = -v(iblock, idx_F(ispec, 2), k, j, i)*lam;

          //printf("dE (old) = %e Estar = %e tau = %e\n", dE, Estar, kappa*dt); 
          // Add source corrections to conserved radiation variables
          v(iblock, idx_E(ispec), k, j, i) += dE; 
          for (int idir=0; idir<3; ++idir) {
            v(iblock, idx_F(ispec, idir), k, j, i) += cov_dF(idir);
          }
          
          // Add source corrections to conserved fluid variables 
          if (update_fluid) {
            v(iblock, ceng, k, j, i) -= dE; 
            v(iblock, cmom_lo + 0, k, j, i) -= cov_dF(0); 
            v(iblock, cmom_lo + 1, k, j, i) -= cov_dF(1); 
            v(iblock, cmom_lo + 2, k, j, i) -= cov_dF(2); 
          }

        } 
      });

  return TaskStatus::complete;
}
template TaskStatus MomentFluidSource<MeshBlockData<Real>>(MeshBlockData<Real> *, Real, bool);

template <class T>
TaskStatus MomentCalculateOpacities(T *rc) { 
  
  auto *pmb = rc->GetParentPointer().get();
   
  StateDescriptor *eos = pmb->packages.Get("eos").get();
  auto &unit_conv = eos->Param<phoebus::UnitConversions>("unit_conv");
  const Real DENSITY = unit_conv.GetMassDensityCodeToCGS();
  const Real TEMPERATURE = unit_conv.GetTemperatureCodeToCGS();

  StateDescriptor *opac = pmb->packages.Get("opacity").get();
  StateDescriptor *rad = pmb->packages.Get("radiation").get();
  
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

  // Get the device opacity object
  using namespace singularity::neutrinos; 
  const auto d_opacity = opac->Param<Opacity>("d.opacity");

  // Get the background geometry 
  auto geom = Geometry::GetCoordinateSystem(rc);

  int nblock = v.GetDim(5); 
  int nspec = idx_kappaJ.DimSize(1);

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

          v(iblock, idx_JBB(ispec), k, j, i) = B;  
          v(iblock, idx_kappaJ(ispec), k, j, i) = kappa;  
          v(iblock, idx_kappaH(ispec), k, j, i) = kappa;  
          
        } 
      });

  return TaskStatus::complete;
}
template TaskStatus MomentCalculateOpacities<MeshBlockData<Real>>(MeshBlockData<Real> *);
} //namespace radiationMoments
