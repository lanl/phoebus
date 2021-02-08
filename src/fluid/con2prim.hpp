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
#include "phoebus_utils/variables.hpp"

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
      prho(imap[primitive_variables::density].first),
      crho(imap[conserved_variables::density].first),
      pvel_lo(imap[primitive_variables::velocity].first),
      pvel_hi(imap[primitive_variables::velocity].second),
      cmom_lo(imap[conserved_variables::momentum].first),
      cmom_hi(imap[conserved_variables::momentum].second),
      peng(imap[primitive_variables::energy].first),
      ceng(imap[conserved_variables::energy].first),
      pb_lo(imap[primitive_variables::bfield].first),
      pb_hi(imap[primitive_variables::bfield].second),
      cb_lo(imap[conserved_variables::bfield].first),
      cb_hi(imap[conserved_variables::bfield].second),
      pye(imap[primitive_variables::ye].second),
      cye(imap[conserved_variables::ye].second),
      prs(imap[primitive_variables::pressure].first),
      tmp(imap[primitive_variables::temperature].first),
      cs(imap[primitive_variables::cs].first),
      gm1(imap[primitive_variables::gamma1].first),
      rel_tolerance(tol),
      max_iter(max_iterations) {}

  const singularity::EOS& SetEOS(MeshBlockData<Real> *rc) {
    return rc->GetBlockPointer()->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  }
  const singularity::EOS& SetEOS(MeshData<Real> *rc) {
    return rc->GetMeshPointer()->packages.Get("eos")->Param<singularity::EOS>("d.EOS");
  }

  std::vector<std::string> Vars() {
    return std::vector<std::string>({primitive_variables::density, conserved_variables::density,
                                    primitive_variables::velocity, conserved_variables::momentum,
                                    primitive_variables::energy, conserved_variables::energy,
                                    primitive_variables::bfield, conserved_variables::bfield,
                                    primitive_variables::ye, conserved_variables::ye,
                                    primitive_variables::pressure, primitive_variables::temperature,
                                    primitive_variables::cs, primitive_variables::gamma1});
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
  const int pb_lo, pb_hi;
  const int cb_lo, cb_hi;
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
