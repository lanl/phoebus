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

namespace homogeneous_sphere {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE((typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski)) ||
                        (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski)),
                    "Problem \"homogeneous_sphere\" requires \"SphericalMinkowski\" or "
                    "\"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      std::vector<std::string>({radmoment_prim::J, radmoment_prim::H, fluid_prim::density,
                                fluid_prim::temperature, fluid_prim::velocity,
                                radmoment_internal::xi, radmoment_internal::phi}),
      imap);

  auto idJ = imap.GetFlatIdx(radmoment_prim::J);
  auto idH = imap.GetFlatIdx(radmoment_prim::H);
  auto idv = imap.GetFlatIdx(fluid_prim::velocity);
  auto ixi = imap.GetFlatIdx(radmoment_internal::xi);
  auto iphi = imap.GetFlatIdx(radmoment_internal::phi);
  const int prho = imap[fluid_prim::density].first;
  const int pT = imap[fluid_prim::temperature].first;

  const auto specB = idJ.GetBounds(1);
  const Real sphere_rad = pin->GetOrAddReal("homogeneous_sphere", "radius", 1.0);
  const Real rho_min = pin->GetOrAddReal("homogeneous_sphere", "rho_min", 1.e-10);
  const Real J = pin->GetOrAddReal("homogeneous_sphere", "J", 0.0);
  const Real Hx = pin->GetOrAddReal("homogeneous_sphere", "Hx", 0.0);
  const Real Hy = pin->GetOrAddReal("homogeneous_sphere", "Hy", 0.0);
  const Real Hz = pin->GetOrAddReal("homogeneous_sphere", "Hz", 0.0);
  const Real vx = pin->GetOrAddReal("homogeneous_sphere", "vx", 0.0);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "Phoebus::ProblemGenerator::homogeneous_sphere", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real r = coords.x1v(i);

        if (r < sphere_rad) {
          v(prho, k, j, i) = 1.0;
        } else {
          v(prho, k, j, i) = rho_min;
        }
        v(pT, k, j, i) = 1.0;

        v(idv(0), k, j, i) = vx;
        v(idv(1), k, j, i) = 0.0;
        v(idv(2), k, j, i) = 0.0;

        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {

          v(ixi(ispec), k, j, i) = 0.0;
          v(iphi(ispec), k, j, i) = acos(-1.0) * 1.000001;

          v(idJ(ispec), k, j, i) = J;
          printf("i = %i r = %e J = %e rho = %e\n", i, r, v(idJ(ispec), k, j, i),
                 v(prho, k, j, i));
          v(idH(0, ispec), k, j, i) = Hx;
          v(idH(1, ispec), k, j, i) = Hy;
          v(idH(2, ispec), k, j, i) = Hz;
        }
      });

  radiation::MomentPrim2Con(rc.get(), IndexDomain::entire);
}

} // namespace homogeneous_sphere
