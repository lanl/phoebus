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

#ifndef CON2PRIM_HPP_
#define CON2PRIM_HPP_

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace con2prim {

enum class ConToPrimStatus {success, failure};

class VarAccessor {
 public:
  KOKKOS_FUNCTION
  VarAccessor(const VariablePack<Real> &var, const int k, const int j, const int i)
              : var_(var), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int n) const {
    return var_(n,k_,j_,i_);
  }
 private:
  const VariablePack<Real> &var_;
  const int i_, j_, k_;
};

struct CellGeom {
  CellGeom(const Geometry::CoordinateSystem &geom,
               const int k, const int j, const int i)
               : gdet(geom.MetricDeterminant(loc,k,j,i)) {
    geom.Metric(CellLocation::Center, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Center, k, j, i, gcon);
  }
  Real gcov[3][3];
  Real gcon[3][3];
  const Real gdet;
};

template <typename T>
class ConToPrim {
 public:
  ConToPrim(const T &pack, PackIndexMap &imap,
            const singularity::EOS &eos_obj,
            const Geometry::CoordinateSystem &geom_obj)
            : var(pack), eos(eos_obj), geom(geom_obj),
              prho(imap["p.density"].first),
              crho(imap["c.density"].first),
              pvel_lo(imap["p.velocity"].first),
              pvel_hi(imap["p.velocity"].second),
              cmom_lo(imap["c.momentum"].first),
              cmom_hi(imap["c.momentum"].second),
              peng(imap["p.energy"].first),
              ceng(imap["c.energy"].first),
              prs(imap["pressure"].first),
              tmp(imap["temperature"].first) { }

  template <class... Args>
  KOKKOS_FUNCTION
  ConToPrimStatus operator()(Args &&... args) const {
    VarAccessor v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    return Solve(v, g);
  }

  KOKKOS_FUNCTION
  ConToPrimStatus operator()(const int b, const int k, const int j, const int i) const {
    VarAccessor v()
  }

  KOKKOS_FUNCTION
  ConToPrimStatus Solve(const VarAccessor &v, const CellGeom &g) const;

 private:
  const T var;
  const singularity::EOS eos;
  const Geometry::CoordinateSystem geom;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int prs, tmp;
};

} // namespace con2prim

#endif // CON2PRIM_HPP_
