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

namespace pf = fluid_prim;
namespace cr = radmoment_cons;
namespace pr = radmoment_prim;
namespace ir = radmoment_internal;
namespace im = mocmc_internal;

KOKKOS_INLINE_FUNCTION
int get_nsamp_per_zone(const int &k, const int &j, const int &i,
                       const Geometry::CoordSysMeshBlock &geom, const Real &rho,
                       const Real &T, const Real &Ye, const Real &J) {

  return 0;
}

template <class T>
void MOCMCInitSamples(T *rc) {

  auto *pmb = rc->GetParentPointer().get();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("mocmc");

  // Meshblock geometry
  const auto geom = Geometry::GetCoordinateSystem(rc);
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // auto &dnsamp = rc->Get(mocmc_internal::dnsamp);
  std::vector<std::string> variables{pr::J, pf::density, pf::temperature, pf::ye};
  PackIndexMap imap;
  auto v = rc->PackVariables(variables, imap);

  auto pJ = imap.GetFlatIdx(pr::J);
  auto pdens = imap[pf::density].first;
  auto pT = imap[pf::temperature].first;
  auto pye = imap[pf::ye].first;
  auto dn = imap[im::dnsamp].first;

  const int nblock = v.GetDim(5);
  PARTHENON_THROW(nblock == 1, "Packing not currently supported for swarms");

  ParArray1D<int> nsamptot("Total samples per meshblock", nblock);

  // Fill nsamp per zone per species and sum over zones
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MOCMC::Reconstruction", DevExecSpace(), 0, nblock - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real Jtot = 0.;
        for (int s = 0; s < 3; s++) {
          Jtot += v(b, pJ(s), k, j, i);
        }
        v(b, dn, k, j, i) =
            get_nsamp_per_zone(k, j, i, geom, v(b, pdens, k, j, i), v(b, pT, k, j, i),
                               v(b, pye, k, j, i), Jtot);
        Kokkos::atomic_add(&(nsamptot(b)), static_cast<int>(v(b, dn, k, j, i)));
      });

  // for (int b = 0; b < nblock; b++) {
  //}
}

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
template void MOCMCInitSamples<MeshBlockData<Real>>(MeshBlockData<Real> *);

} // namespace radiation
