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

#include "closure.hpp"
#include "radiation/radiation.hpp"
#include "reconstruction.hpp"

namespace radiation {

template <class T>
TaskStatus MOCMCTransport(T *rc) {
  return TaskStatus::complete;
}
template TaskStatus MOCMCTransport<MeshBlockData<Real>>(MeshBlockData<Real> *);

template <class T>
TaskStatus MOCMCReconstruction(T *rc) {

  namespace ir = radmoment_internal;

  auto *pm = rc->GetParentPointer().get();

  IndexRange ib = pm->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pm->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pm->cellbounds.GetBoundsK(IndexDomain::entire);

  std::vector<std::string> variables{ir::tilPi};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto iTilPi = imap.GetFlatIdx(ir::tilPi);
  auto specB = iTilPi.GetBounds(1);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::Reconstruction", DevExecSpace(),
      0, v.GetDim(5)-1,
      specB.s, specB.e,
      kb.s, kb.e,
      jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int ispec, const int k, const int j, const int i) {
        SPACELOOP2(ii, jj) {
          v(b, iTilPi(ispec, ii, jj), k, j, i) = 0.;
}
} // namespace radiation
);

return TaskStatus::complete;
}
template TaskStatus MOCMCReconstruction<MeshBlockData<Real>>(MeshBlockData<Real> *);

} // namespace radiation
