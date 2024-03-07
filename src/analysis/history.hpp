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

#ifndef ANALYSIS_HISTORY_HPP_
#define ANALYSIS_HISTORY_HPP_

#include <string>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;

/*
 * History Template to build off of. Applies Reducer to MeshData on
 * variable with varname. If variable is multi-d use idx to offset
 * from first element. You should capture this object in a lambda to
 * pass it to the Parthenon history machinery.
 *
 * Works on max/min and on sums over densitized variables, which
 * include the determinant of the metric.
 */

namespace History {

Real ReduceMassAccretionRate(MeshData<Real> *md);
Real ReduceJetEnergyFlux(MeshData<Real> *md);
Real ReduceJetMomentumFlux(MeshData<Real> *md);
Real ReduceMagneticFluxPhi(MeshData<Real> *md);
Real ReduceCentralDensitySN(MeshData<Real> *md);

template <typename Reducer_t>
Real ReduceOneVar(MeshData<Real> *md, const std::string &varname, int idx = 0) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  std::vector<std::string> vars = {varname};
  const auto pack = md->PackVariables(vars, imap);
  const auto ivar = imap[varname];
  PARTHENON_REQUIRE(ivar.first >= 0, "Var must exist");
  PARTHENON_REQUIRE(ivar.second >= ivar.first + idx, "Var must exist");

  // We choose to apply volume weighting when using the sum reduction.
  // This assumes that for "Sum" variables, the variable is densitized, meaning
  // it already contains a factor of the measure: sqrt(det(gamma))
  const bool volume_weighting =
      std::is_same<Reducer_t, Kokkos::Sum<Real, HostExecSpace>>::value;

  Real result = 0.0;
  Reducer_t reducer(result);
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for " + varname, DevExecSpace(),
      0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        // join is a Kokkos construct
        // that automatically does the
        // reduction operation locally
        const auto &coords = pack.GetCoords(b);
        const Real vol = volume_weighting ? coords.CellVolume(k, j, i) : 1.0;
        reducer.join(lresult, pack(b, ivar.first + idx, k, j, i) * vol);
      },
      reducer);
  return result;
}

} // namespace History

#endif // ANALYSIS_HISTORY_HPP_
