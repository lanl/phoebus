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

#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/variables.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

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
void ReduceLocalizationFunction(MeshData<Real> *md);
Real CalculateMdot(MeshData<Real> *md, Real rc, bool gain);

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
  Real result = 0.0;
  Reducer_t reducer(result);

  const bool volume_weighting =
      std::is_same<Reducer_t, Kokkos::Sum<Real, HostExecSpace>>::value ||
      std::is_same<Reducer_t, Kokkos::Sum<Real, DevExecSpace>>::value ||
      std::is_same<Reducer_t, Kokkos::Sum<Real, Kokkos::DefaultExecutionSpace>>::value;

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

template <typename Varname>
Real ReduceInGain(MeshData<Real> *md, bool is_conserved, int idx = 0) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  namespace iv = internal_variables;
  using parthenon::MakePackDescriptor;
  auto *pmb = md->GetParentPointer();
  Mesh *pmesh = md->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  const int ndim = pmesh->ndim;
  static auto desc =
      MakePackDescriptor<Varname, fluid_prim::entropy, iv::GcovHeat, iv::GcovCool>(
          resolved_pkgs.get());
  auto v = desc.GetPack(md);
  PARTHENON_REQUIRE_THROWS(v.ContainsHost(0, iv::GcovHeat(), iv::GcovCool()),
                           "Must be doing SN simulation");
  const int nblocks = v.GetNBlocks();

  Real result = 0.0;

  auto geom = Geometry::GetCoordinateSystem(md);
  auto analysis = pmb->packages.Get("analysis").get();
  const Real outside_pns_threshold = analysis->Param<Real>("outside_pns_threshold");

  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(),
      "Phoebus History for integrals in gain region (SN)", DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        Real gdet = geom.DetGamma(CellLocation::Cent, 0, k, j, i);
        bool is_netheat = (v(b, iv::GcovHeat(), k, j, i) - v(b, iv::GcovCool(), k, j, i) >
                           1.e-8); // checks that in the gain region
        bool is_outside_pns = (v(b, fluid_prim::entropy(), k, j, i) >
                               outside_pns_threshold); // checks that outside PNS
        const auto &coords = v.GetCoordinates(b);
        const Real vol = coords.CellVolume(k, j, i);
        if (is_conserved) {
          lresult += vol * is_netheat * is_outside_pns * v(b, Varname(idx), k, j, i);
        } else {
          lresult +=
              gdet * vol * is_netheat * is_outside_pns * v(b, Varname(idx), k, j, i);
        }
      },
      Kokkos::Sum<Real>(result));
  return result;
}

using namespace parthenon::package::prelude;
template <typename F>
KOKKOS_INLINE_FUNCTION Real ComputeDivInPillbox(int ndim, int b, int k, int j, int i,
                                                const parthenon::Coordinates_t &coords,
                                                const F &f) {
  Real div_mass_flux_integral;
  div_mass_flux_integral =
      (f(b, 1, k, j, i + 1) - f(b, 1, k, j, i)) * coords.FaceArea<X1DIR>(k, j, i);

  if (ndim >= 2) {
    div_mass_flux_integral +=
        (f(b, 2, k, j + 1, i) - f(b, 2, k, j, i)) * coords.FaceArea<X2DIR>(k, j, i);
  }
  if (ndim >= 3) {
    div_mass_flux_integral +=
        (f(b, 3, k + 1, j, i) - f(b, 3, k, j, i)) * coords.FaceArea<X3DIR>(k, j, i);
  }
  return div_mass_flux_integral;
}

} // namespace History

#endif // ANALYSIS_HISTORY_HPP_
