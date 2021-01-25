//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef FLUID_HPP_
#define FLUID_HPP_

#include <memory>

#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include <eos/eos.hpp>
#include "con2prim.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/*template <typename T>
TaskStatus PrimitiveToConserved(T *rc);

template <typename T>
TaskStatus ConservedToPrimitive(T *rc);

*/
template <typename T>
TaskStatus PrimitiveToConserved(T *rc) {
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure"});

  PackIndexMap imap;
  auto &v = rc->PackVariables(vars, imap);

  const int prho = imap["p.density"].first;
  const int crho = imap["c.density"].first;
  const int pvel_lo = imap["p.velocity"].first;
  const int pvel_hi = imap["p.velocity"].second;
  const int cmom_lo = imap["c.momentum"].first;
  const int cmom_hi = imap["c.momentum"].second;
  const int peng = imap["p.energy"].first;
  const int ceng = imap["c.energy"].first;
  const int prs = imap["pressure"].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto geom = Geometry::GetCoordinateSystem(rc);

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "PrimToCons", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      //todo(jcd): make these real
      Real gcov[3][3];
      geom.Metric(CellLocation::Cent, b, k, j, i, gcov);
      Real gdet = geom.DetGamma(CellLocation::Cent, b, k, j, i);
      Real vsq = 0.0;
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          vsq += gcov[m][n] * v(b,pvel_lo+m, k, j, i) * v(b, pvel_lo+n, k, j, i);
        }
      }
      Real W = 1.0/sqrt(1.0 - vsq);
      // conserved density D = \sqrt{\gamma} \rho W
      v(b, crho, k, j, i) = gdet * v(b, prho, k, j, i) * W;

      Real rhoh = v(b, prho, k, j, i) + v(b, peng, k, j, i) + v(b, prs, k, j, i);
      for (int m = 0; m < 3; m++) {
        Real vcov = 0.0;
        for (int n = 0; n < 3; n++) {
          vcov += gcov[m][n]*v(b, pvel_lo+n, k, j, i);
        }
        v(b, cmom_lo+m, k, j, i) = gdet*rhoh*W*W*vcov;
      }

      v(b, ceng, k, j, i) = gdet*(rhoh*W*W - v(b, prs, k, j, i)) - v(b, crho, k, j, i);
    });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitive(T *rc) {
  using namespace con2prim;
  auto *pmb = rc->GetParentPointer();

  std::vector<std::string> vars({"p.density", "c.density",
                                 "p.velocity", "c.momentum",
                                 "p.energy", "c.energy",
                                 "pressure", "temperature"});

  using namespace singularity;
  auto &eos_pkg = pmb->packages.Get("EOS");
  auto &eos = eos_pkg->Param<EOS>("d.EOS");

  PackIndexMap imap;
  auto &v = rc->PackVariables(vars, imap);
  auto geom = Geometry::GetCoordinateSystem(rc);
  ConToPrim invert(v, imap, eos, geom);

  const int ifail;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  int fail_cnt;
  parthenon::par_reduce(DEFAULT_LOOP_PATTERN, "ConToPrim", DevExecSpace(),
    0, v.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &fail) {
      auto status = invert(b, k, j, i);
      if (status == ConToPrimStatus::failure) fail++;
    }, Kokkos::Sum<int>(fail_cnt));
  return TaskStatus::complete;
}




} // namespace fluid

#endif // FLUID_HPP_
