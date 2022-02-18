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

#include <cmath>
#include <string>

#include "pgen/pgen.hpp"

namespace radiation_equilibration {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE((typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski)),
    "Problem \"radiation_equilibration\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({radmoment_prim::J, 
                              radmoment_prim::H, 
                              radmoment_internal::xi, 
                              radmoment_internal::phi,
                              fluid_prim::density, 
                              fluid_prim::temperature, 
                              fluid_prim::pressure, 
                              fluid_prim::energy,
                              fluid_prim::velocity}, 
                              imap);
  
  auto idJ = imap.GetFlatIdx(radmoment_prim::J);
  auto idH = imap.GetFlatIdx(radmoment_prim::H);
  auto ixi = imap.GetFlatIdx(radmoment_internal::xi);
  auto iphi = imap.GetFlatIdx(radmoment_internal::phi);
  
  const int iRho = imap[fluid_prim::density].first; 
  const int iT = imap[fluid_prim::temperature].first; 
  const int iP = imap[fluid_prim::pressure].first; 
  const int iEps = imap[fluid_prim::energy].first; 
  auto idv = imap.GetFlatIdx(fluid_prim::velocity);
  
  const auto specB = idJ.GetBounds(1);
  const Real J = pin->GetOrAddReal("radiation_equilibration", "J", 0.0);
  
  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
      "Phoebus::ProblemGenerator::radiation_equilibration", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        
        const Real rho = 1.0;
        const Real T = 1.0; 
        const Real P = eos.PressureFromDensityTemperature(rho, T);
        const Real eps = eos.InternalEnergyFromDensityTemperature(rho, T);
        
        v(iRho, k, j, i) = rho; 
        v(iT, k, j, i) = T; 
        v(iP, k, j, i) = P; 
        v(iEps, k, j, i) = eps; 
        SPACELOOP(ii) v(idv(ii), k, j, i) = 0.0; 
        
        for (int ispec = specB.s; ispec<=specB.e; ++ispec) {
          SPACELOOP(ii) v(idH(ii, ispec), k, j, i) = 0.0; 
          v(idJ(ispec), k, j, i) = J; 
        }
      });

  radiation::MomentPrim2Con(rc.get(), IndexDomain::entire);
  fluid::PrimitiveToConserved(rc.get());
}

} // namespace radiation_equilibration
