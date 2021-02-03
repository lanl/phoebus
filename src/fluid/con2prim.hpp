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

// parthenon provided headers
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

// singulaarity
#include <eos/eos.hpp>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"

namespace con2prim {

enum class ConToPrimStatus {success, failure};

template <typename T>
class VarAccessor {
 public:
  KOKKOS_FUNCTION
  VarAccessor(const T &var, const int k, const int j, const int i)
              : var_(var), b_(0), k_(k), j_(j), i_(i) {}
  VarAccessor(const T &var, const int b, const int k, const int j, const int i)
              : var_(var), b_(b), k_(k), j_(j), i_(i) {}
  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(const int n) const {
    return var_(b_, n, k_, j_, i_);
  }
 private:
  const T &var_;
  const int b_, i_, j_, k_;
};

struct CellGeom {
  CellGeom(const Geometry::CoordinateSystem &geom,
               const int k, const int j, const int i)
               : gdet(geom.DetGamma(CellLocation::Cent,k,j,i)) {
    geom.Metric(CellLocation::Cent, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);
  }
  CellGeom(const Geometry::CoordinateSystem &geom,
           const int b, const int k, const int j, const int i)
           : gdet(geom.DetGamma(CellLocation::Cent,b,k,j,i)) {
    geom.Metric(CellLocation::Cent, b, k, j, i, gcov);
    geom.MetricInverse(CellLocation::Cent, b, k, j, i, gcon);
  }
  Real gcov[3][3];
  Real gcon[3][3];
  const Real gdet;
};

template <typename Data_t, typename T>
class ConToPrim {
 public:
  ConToPrim(Data_t *rc, const Real tol, const int max_iterations)
    : var(rc->PackVariables(Vars(), imap)),
      eos(SetEOS(rc)),
      geom(Geometry::GetCoordinateSystem(rc)),
      prho(imap["p.density"].first),
      crho(imap["c.density"].first),
      pvel_lo(imap["p.velocity"].first),
      pvel_hi(imap["p.velocity"].second),
      cmom_lo(imap["c.momentum"].first),
      cmom_hi(imap["c.momentum"].second),
      peng(imap["p.energy"].first),
      ceng(imap["c.energy"].first),
      pye(imap["p.ye"].second),
      cye(imap["c.ye"].second),
      prs(imap["pressure"].first),
      tmp(imap["temperature"].first),
      cs(imap["cs"].first),
      gm1(imap["gamma1"].first),
      rel_tolerance(tol),
      max_iter(max_iterations) {}

  const singularity::EOS& SetEOS(MeshBlockData<Real> *rc) {
    return rc->GetBlockPointer()->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  }
  const singularity::EOS& SetEOS(MeshData<Real> *rc) {
    return rc->GetMeshPointer()->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  }

  std::vector<std::string> Vars() {
    return std::vector<std::string>({"p.density", "c.density",
                                    "p.velocity", "c.momentum",
                                    "p.energy", "c.energy",
                                    "p.ye", "c.ye",
                                    "pressure", "temperature",
                                    "cs", "gamma1"});
  }

  template <class... Args>
  KOKKOS_FUNCTION
  ConToPrimStatus operator()(Args &&... args) const {
    VarAccessor<T> v(var, std::forward<Args>(args)...);
    CellGeom g(geom, std::forward<Args>(args)...);
    return Solve(v, g);
  }

  KOKKOS_FUNCTION
  ConToPrimStatus Solve(const VarAccessor<T> &v, const CellGeom &g, const bool print=false) const;

  int NumBlocks() {
    return var.GetDim(5);
  }

 private:
  PackIndexMap imap;
  const T var;
  const singularity::EOS eos;
  const Geometry::CoordinateSystem geom;
  const int prho, crho;
  const int pvel_lo, pvel_hi;
  const int cmom_lo, cmom_hi;
  const int peng, ceng;
  const int pye, cye;
  const int prs, tmp, cs, gm1;
  const Real rel_tolerance;
  const int max_iter;
};



using C2P_Block_t = ConToPrim<MeshBlockData<Real>,VariablePack<Real>>;
using C2P_Mesh_t = ConToPrim<MeshData<Real>,MeshBlockPack<Real>>;

inline C2P_Block_t ConToPrimSetup(MeshBlockData<Real> *rc, const Real tol, const int max_iter) {
  return C2P_Block_t(rc, tol, max_iter);
}
/*inline C2P_Mesh_t ConToPrimSetup(MeshData<Real> *rc) {
  return C2P_Mesh_t(rc);
}*/

} // namespace con2prim

#endif // CON2PRIM_HPP_
