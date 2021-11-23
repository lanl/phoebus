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

KOKKOS_FORCEINLINE_FUNCTION 
Vec operator+(Vec a, Vec b) {Vec out; SPACELOOP(i) out(i) = a(i) + b(i); return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Vec operator-(Vec a, Vec b) {Vec out; SPACELOOP(i) out(i) = a(i) - b(i); return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Vec operator-(Vec a) {Vec out; SPACELOOP(i) out(i) = -a(i); return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Vec operator*(Vec a, Real b) {Vec out; SPACELOOP(i) out(i) = a(i)*b; return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Vec operator*(Real a, Vec b) {Vec out; SPACELOOP(i) out(i) = a*b(i); return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Vec operator/(Vec a, Real b) {Vec out; SPACELOOP(i) out(i) = a(i)/b; return out;} 

struct Tens2 { 
  Real data[Geometry::NDSPACE][Geometry::NDSPACE]; 
  KOKKOS_FORCEINLINE_FUNCTION 
  Real& operator()(const int i, const int j){return data[i][j];} 
  KOKKOS_FORCEINLINE_FUNCTION
  const Real& operator()(const int i, const int j) const {return data[i][j];} 
};

KOKKOS_FORCEINLINE_FUNCTION 
Tens2 operator+(Tens2 a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j) + b(i,j);} return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Tens2 operator-(Tens2 a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j) - b(i,j);} return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Tens2 operator*(Real a, Tens2 b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a*b(i,j);} return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Tens2 operator*(Tens2 a, Real b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j)*b;} return out;} 
KOKKOS_FORCEINLINE_FUNCTION 
Tens2 operator/(Tens2 a, Real b) {Tens2 out; SPACELOOP2(i,j) {out(i,j) = a(i,j)/b;} return out;}

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
  
  namespace cr = radmoment_cons;  
  namespace pr = radmoment_prim;  
  namespace ir = radmoment_internal;  
  
  auto *pm = rc->GetParentPointer().get(); 
  
  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap imap;
  auto v = rc->PackVariables(std::vector<std::string>{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity, ir::xi, ir::phi}, imap);
  
  auto cE = imap.GetFlatIdx(cr::E);  
  auto pJ = imap.GetFlatIdx(pr::J);  
  auto cF = imap.GetFlatIdx(cr::F);  
  auto pH = imap.GetFlatIdx(pr::H); 
  auto iXi = imap.GetFlatIdx(ir::xi); 
  auto iPhi = imap.GetFlatIdx(ir::phi); 
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
        
        Real xi = v(b, iXi(ispec), k, j, i);
        Real phi = 1.0001*v(b, iPhi(ispec), k, j, i);

        auto result = c.Con2PrimM1(E, covF, xi, phi, &J, &covH, &conTilPi);
        
        v(b, iXi(ispec), k, j, i) = result.xi;
        v(b, iPhi(ispec), k, j, i) = result.phi;

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
  
  namespace cr = radmoment_cons;  
  namespace pr = radmoment_prim;  
  namespace ir = radmoment_internal;  
  
  auto *pm = rc->GetParentPointer().get(); 
  
  IndexRange ib = pm->cellbounds.GetBoundsI(domain);
  IndexRange jb = pm->cellbounds.GetBoundsJ(domain);
  IndexRange kb = pm->cellbounds.GetBoundsK(domain);

  PackIndexMap imap;
  auto v = rc->PackVariables(std::vector<std::string>{cr::E, cr::F, pr::J, pr::H, fluid_prim::velocity}, imap);
  
  auto cE = imap.GetFlatIdx(cr::E);  
  auto pJ = imap.GetFlatIdx(pr::J);  
  auto cF = imap.GetFlatIdx(cr::F);  
  auto pH = imap.GetFlatIdx(pr::H);  
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
  
  namespace cr = radmoment_cons;  
  namespace pr = radmoment_prim;  
  namespace ir = radmoment_internal;  
  
  PackIndexMap imap_ql, imap_qr, imap;
  VariablePack<Real> ql_base = rc->PackVariables(std::vector<std::string>{ir::ql}, imap_ql); 
  VariablePack<Real> qr_base = rc->PackVariables(std::vector<std::string>{ir::qr}, imap_qr); 
  VariablePack<Real> v = rc->PackVariables(std::vector<std::string>{pr::J, pr::H, ir::dJ}, imap);
  auto idx_J = imap.GetFlatIdx(pr::J);
  auto idx_dJ = imap.GetFlatIdx(ir::dJ);

  ParArrayND<Real> ql_v = rc->Get(ir::ql_v).data; 
  ParArrayND<Real> qr_v = rc->Get(ir::qr_v).data; 
  VariablePack<Real> v_vel = rc->PackVariables(std::vector<std::string>{fluid_prim::velocity});
  auto qIdx = imap_ql.GetFlatIdx(ir::ql);
  
  const int nspec = qIdx.DimSize(1);
  const int nrecon = 4*nspec;

  const int offset = imap_ql[ir::ql].first; 
  
  const int nblock = ql_base.GetDim(5); 
  
  //PARTHENON_REQUIRE(nrecon == v.GetDim(4), "Issue with number of reconstruction variables in moments.");
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "RadMoments::Reconstruct", DevExecSpace(), 
      X1DIR, pmb->pmy_mesh->ndim, // Loop over directions for reconstruction
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
          for (int idir = X1DIR; idir <= pmb->pmy_mesh->ndim; ++idir) {
            // Calculate the derivatives in the plane of the face (and put junk in the derivative perpendicular to the face) 
            const Real dy = pmb->coords.Dx(idir, k, j, i); 
            v(b, idx_dJ(ispec, idir-1, iface-1), k, j, i) = (v(b, idx_J(ispec), k+st_k[idir-1], j+st_j[idir-1], i+st_i[idir-1])  
                                                            -v(b, idx_J(ispec), k-st_k[idir-1], j-st_j[idir-1], i-st_i[idir-1])
                                                            +v(b, idx_J(ispec), k+st_k[idir-1]-off_k, j+st_j[idir-1]-off_j, i+st_i[idir-1]-off_i)
                                                            -v(b, idx_J(ispec), k-st_k[idir-1]-off_k, j-st_j[idir-1]-off_j, i-st_i[idir-1]-off_i))/(4*dy);
          }
          // Overwrite the derivative perpendicular to the face
          const Real dx = pmb->coords.Dx(iface, k, j, i); 
          v(b, idx_dJ(ispec, iface-1, iface-1), k, j, i) =  (v(b, idx_J(ispec), k, j, i) - v(b, idx_J(ispec), k-off_k, j-off_j, i-off_i))/dx; 
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
        
        for (int ispec = 0; ispec<nspec; ++ispec) { 
          /// TODO: (LFR) Fix indexing so everything starts on a consistent index
        
          const Real& Jl = v(idx_ql(ispec, 0, idir), k, j, i);
          const Real& Jr = v(idx_qr(ispec, 0, idir), k, j, i);
          const Vec Hl = {v(idx_ql(ispec, 1, idir), k, j, i), 
                              v(idx_ql(ispec, 2, idir), k, j, i), 
                              v(idx_ql(ispec, 3, idir), k, j, i)};
          const Vec Hr = {v(idx_qr(ispec, 1, idir), k, j, i), 
                              v(idx_qr(ispec, 2, idir), k, j, i), 
                              v(idx_qr(ispec, 3, idir), k, j, i)}; 
          
          Tens2 con_tilPi;
          
          Vec con_vl{{v(idx_qlv(0, idir), k, j, i), 
                      v(idx_qlv(1, idir), k, j, i),
                      v(idx_qlv(2, idir), k, j, i)}};
          Vec con_vr{{v(idx_qrv(0, idir), k, j, i), 
                      v(idx_qrv(1, idir), k, j, i),
                      v(idx_qrv(2, idir), k, j, i)}};
          
          Vec cov_dJ{{v(idx_dJ(ispec, 0, idir), k, j, i), 
                      v(idx_dJ(ispec, 1, idir), k, j, i),
                      v(idx_dJ(ispec, 2, idir), k, j, i)}};

          Real kappaH = 0.5*(v(idx_kappaH(ispec), k, j, i) + v(idx_kappaH(ispec), k - koff, j - joff, i - ioff));
          
          Vec con_beta; 
          Tens2 cov_gamma;
          geom.Metric(face, k, j, i, cov_gamma.data); 
          geom.ContravariantShift(face, k, j, i, con_beta.data);
          
          const Real dx = pmb->coords.Dx(idir_in, k, j, i)*sqrt(cov_gamma(idir, idir));  
          const Real a = tanh(1/std::max(kappaH*dx, 1.e-20));
          //printf ("a : %e dJ/dx^i = (%e %e %e) \n", a, cov_dJ(0), cov_dJ(1), cov_dJ(2));

          // Calculate the observer frame quantities on the left side of the interface  
          /// TODO: (LFR) Add other contributions to the asymptotic flux 
          Vec HasymL = -cov_dJ/(3*kappaH);   
          Closure<Vec, Tens2> cl(con_vl, cov_gamma); 
          Real El; 
          Vec covFl, conFl, conFl_asym;
          Tens2 Pl, Pl_asym; // P^i_j on the left side of the interface
          
          // Fluxes in the asymptotic limit 
          cl.Prim2ConM1(Jl, HasymL, &El, &covFl, &con_tilPi); 
          cl.raise3Vector(covFl, &conFl_asym); 
          cl.getConCovPFromPrim(Jl, HasymL, con_tilPi, &Pl_asym);

          // Regular fluxes          
          cl.Prim2ConM1(Jl, Hl, &El, &covFl, &con_tilPi);
          cl.raise3Vector(covFl, &conFl);
          cl.getConCovPFromPrim(Jl, Hl, con_tilPi, &Pl);
          
          // Calculate the observer frame quantities on the right side of the interface  
          /// TODO: (LFR) Add other contributions to the asymptotic flux 
          Vec HasymR = -cov_dJ/(3*kappaH); 
          Closure<Vec, Tens2> cr(con_vr, cov_gamma); 
          Real Er; 
          Vec covFr, conFr, conFr_asym;
          Tens2 Pr, Pr_asym; // P^i_j on the right side of the interface
          
          // Fluxes in the asymptotic limit 
          cr.Prim2ConM1(Jr, HasymR, &Er, &covFr, &con_tilPi); 
          cr.raise3Vector(covFr, &conFr_asym); 
          cr.getConCovPFromPrim(Jr, HasymR, con_tilPi, &Pr_asym);
          
          // Regular fluxes 
          cr.Prim2ConM1(Jr, Hr, &Er, &covFr, &con_tilPi);
          cr.raise3Vector(covFr, &conFr);
          cr.getConCovPFromPrim(Jr, Hr, con_tilPi, &Pr);
          
          // Mix the fluxes by the Peclet number
          // TODO: (LFR) Make better choices  
          const Real speed = a*1.0/sqrt(3.0) + (1-a)*std::max(sqrt(cl.v2), sqrt(cr.v2));  
          conFl = a*conFl + (1-a)*conFl_asym; 
          conFr = a*conFr + (1-a)*conFr_asym; 
          Pl = a*Pl + (1-a)*Pl_asym;
          Pr = a*Pr + (1-a)*Pr_asym; 

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
   
  namespace cr = radmoment_cons;  
  namespace pr = radmoment_prim;  
  namespace ir = radmoment_internal;  
  namespace c = fluid_cons;
  namespace p = fluid_prim;
  std::vector<std::string> vars{cr::E, cr::F, p::density, p::temperature, p::ye, p::velocity,
                                pr::J, pr::H, 
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
  auto idx_J = imap.GetFlatIdx(pr::J); 
  auto idx_H = imap.GetFlatIdx(pr::H);

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
          Vec con_tilf;
          Tens2 con_tilPi{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};  

          // Treat the Eddington tensor explicitly  
          Real& J = v(iblock, idx_J(ispec), k, j, i); 
          Vec cov_H{{v(iblock, idx_H(ispec, 0), k, j, i), 
                     v(iblock, idx_H(ispec, 1), k, j, i),
                     v(iblock, idx_H(ispec, 2), k, j, i),
                    }}; 
          c.M1FluidPressureTensor(J, cov_H, &con_tilPi, &con_tilf); 

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
          // Add source corrections to conserved iration variables
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

          Real kappa = d_opacity.AbsorptionCoefficient(rho_cgs, T_cgs, Ye, species[ispec], enu);
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
} //namespace irationMoments
