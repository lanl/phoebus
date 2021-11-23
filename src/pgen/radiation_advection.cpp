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

namespace radiation_advection {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
    "Problem \"advection\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v =
      rc->PackVariables(std::vector<std::string>({radmoment_prim::J, radmoment_prim::H, 
                        fluid_prim::density, fluid_prim::temperature, fluid_prim::velocity, 
                        radmoment_internal::xi, radmoment_internal::phi}),
                        imap);
  
  auto idJ = imap.GetFlatIdx(radmoment_prim::J);
  auto idH = imap.GetFlatIdx(radmoment_prim::H);
  auto idv = imap.GetFlatIdx(fluid_prim::velocity);
  const int prho = imap[fluid_prim::density].first; 
  const int pT = imap[fluid_prim::temperature].first; 
  const int iXi = imap[radmoment_internal::xi].first; 
  const int iPhi = imap[radmoment_internal::phi].first; 

  const auto specB = idJ.GetBounds(1);
  const Real J = pin->GetOrAddReal("radiation_advection", "J", 1.0);
  const Real Hx = pin->GetOrAddReal("radiation_advection", "Hx", 0.0);
  const Real Hy = pin->GetOrAddReal("radiation_advection", "Hy", 0.0);
  const Real Hz = pin->GetOrAddReal("radiation_advection", "Hz", 0.0);
  const Real vx = pin->GetOrAddReal("radiation_advection", "vx", 0.0);
  const Real width = pin->GetOrAddReal("radiation_advection", "width", sqrt(2.0));
  const int shapedim = pin->GetOrAddInteger("radiation_advection", "shapedim", 2);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  //auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  pmb->par_for(
      "Phoebus::ProblemGenerator::radiation_advection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

        Real x = coords.x1v(i);
        Real y = (ndim > 1 && shapedim > 1) ? coords.x2v(j) : 0;
        Real z = (ndim > 2 && shapedim > 2) ? coords.x3v(k) : 0;
        Real r = std::sqrt(x * x + y * y + z * z);
        
        v(prho, k, j, i) = 1.0;
        v(pT, k, j, i) = 1.0;

        v(iXi, k, j, i) = 1.e-5;
        v(iPhi, k, j, i) = acos(-1)*1.0001;

        v(idv(0), k, j, i) = vx; 
        v(idv(1), k, j, i) = 0.0; 
        v(idv(2), k, j, i) = 0.0; 

        for (int ispec = specB.s; ispec<=specB.e; ++ispec) {
          v(idJ(ispec), k, j, i) = J*exp(-std::pow((r - 0.5)/(0.5*(width*width)), 2));
          v(idH(0, ispec), k, j, i) = Hx;
          v(idH(1, ispec), k, j, i) = Hy;
          v(idH(2, ispec), k, j, i) = Hz;
        }
      });

  radiation::MomentPrim2Con(rc.get());
}

} // namespace advection
