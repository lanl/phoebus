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

#ifndef PHOEBUS_UTILS_HISTORY_HPP_
#define PHOEBUS_UTILS_HISTORY_HPP_

#include <string>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

using namespace parthenon::package::prelude;

/*
 * History Template to build off of. Applies Reducer to MeshData on
 * variable with varname. If variable is multi-d use idx to offset
 * from first element. You should capture this object in a lambda to
 * pass it to the Parthenon history machinery.
 *
 * TODO(JMM): This one is very simple. No volume weighting, no metric
 * information. So it only works for min and max basically.  This can
 * be made more sophisticated if needed, but I think this is fine for
 * now.
 */

namespace History {

template <typename Reducer_t>
Real ReduceOneVar(MeshData<Real> *md, const std::string &varname, int idx = 0) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  std::vector<std::string> vars = {varname};
  const auto pack = md->PackVariables(vars, imap);
  int ivar = imap[vars];
  PARTHENON_REQUIRE(ivar.first >= 0, "Var must exist");
  PARTHENON_REQUIRE(ivar.second >= ivar.first + idx, "Var must exist");

  Real result = 0.0;
  Reducer_t reducer(result);
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Phoebus History for " + varname, 0, pack.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        // join is a Kokkos construct
        // that automatically does the
        // reduction operation locally
        reducer.join(lresult, pack(b, idx, k, j, i));
      },
      reducer);
  return result;
}

} // namespace History

#endif // PHOEBUS_UTILS_HISTORY_HPP_
