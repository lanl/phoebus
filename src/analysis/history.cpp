// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#include "history.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "history_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"

namespace History {

Real ReduceMassAccretionRate(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  auto pmb = md->GetParentPointer();
  auto &pars = pmb->packages.Get("geometry")->AllParams();
  const Real xh = pars.Get<Real>("xh");

  namespace p = fluid_prim;
  const std::vector<std::string> vars({p::density, p::velocity});

  PackIndexMap imap;
  auto pack = md->PackVariables(vars, imap);

  const int prho = imap[p::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;

  auto geom = Geometry::GetCoordinateSystem(md);

  Real result = 0.0;
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for Mass Accretion Rate",
      DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = pack.GetCoords(b);
        if (coords.x1f(i) <= xh && xh < coords.x1f(i + 1)) {
          const Real gdet = geom.DetGamma(CellLocation::Cent, k, j, i);

          const Real dx1 = coords.Dx(X1DIR, k, j, i);
          const Real dx2 = coords.Dx(X2DIR, k, j, i);
          const Real dx3 = coords.Dx(X3DIR, k, j, i);

          // interp to make sure we're getting the horizon correct
          auto m = (CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i + 1) -
                    CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i - 1)) /
                   (2.0 * dx1);
          auto flux = (CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i) +
                       (xh - coords.x1v(i)) * m) *
                      dx2 * dx3;

          lresult += -flux;
        } else {
          lresult += 0.0;
        }
      },
      result);
  return result;
} // mass accretion

} // namespace History
