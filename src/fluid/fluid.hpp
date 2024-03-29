//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "phoebus_utils/variables.hpp"

namespace fluid {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus PrimitiveToConserved(MeshBlockData<Real> *rc);
TaskStatus PrimitiveToConservedRegion(MeshBlockData<Real> *rc, const IndexRange &ib,
                                      const IndexRange &jb, const IndexRange &kb);
// JMM: Not sure how the templated value here worked in the first
// place. But proper solution is need to tell the linker it's
// available elsewhere.
template <typename T>
TaskStatus ConservedToPrimitiveRegion(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb);
extern template TaskStatus
ConservedToPrimitiveRegion<MeshData<Real>>(MeshData<Real> *rc, const IndexRange &ib,
                                           const IndexRange &jb, const IndexRange &kb);
extern template TaskStatus ConservedToPrimitiveRegion<MeshBlockData<Real>>(
    MeshBlockData<Real> *rc, const IndexRange &ib, const IndexRange &jb,
    const IndexRange &kb);

template <typename T>
TaskStatus ConservedToPrimitive(T *rc);
template <typename T>
TaskStatus ConservedToPrimitiveRobust(T *rc, const IndexRange &ib, const IndexRange &jb,
                                      const IndexRange &kb);
template <typename T>
TaskStatus ConservedToPrimitiveClassic(T *rc, const IndexRange &ib, const IndexRange &jb,
                                       const IndexRange &kb);
template <typename T>
TaskStatus ConservedToPrimitiveVanDerHolst(T *rc, const IndexRange &ib,
                                           const IndexRange &jb, const IndexRange &kb);
TaskStatus CalculateFluidSourceTerms(MeshData<Real> *rc, MeshData<Real> *rc_src);
TaskStatus CalculateFluxes(MeshBlockData<Real> *rc);
TaskStatus FluxCT(MeshBlockData<Real> *rc);
TaskStatus CalculateDivB(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

template <class T>
TaskStatus CopyFluxDivergence(T *rc);

template <typename T>
using c2p_type = TaskStatus (*)(T *, const IndexRange &, const IndexRange &,
                                const IndexRange &);
using c2p_meshblock_type = c2p_type<MeshBlockData<Real>>;
using c2p_mesh_type = c2p_type<MeshData<Real>>;

#if SET_FLUX_SRC_DIAGS
template <class T>
TaskStatus CopyFluxDivergence(T *rc) {
  Mesh *pm = rc->GetMeshPointer();
  auto &fluid = pm->packages.Get("fluid");
  const Params &params = fluid->AllParams();
  if (!params.Get<bool>("active")) return TaskStatus::complete;

  std::vector<std::string> vars({fluid_cons::density::name(),
                                 fluid_cons::momentum::name(),
                                 fluid_cons::energy::name()});
  vars.push_back(radmoment_cons::E::name());
  vars.push_back(radmoment_cons::F::name());
  PackIndexMap imap;
  auto divf = rc->PackVariables(vars, imap);
  const int crho = imap[fluid_cons::density::name()].first;
  const int cmom_lo = imap[fluid_cons::momentum::name()].first;
  const int cmom_hi = imap[fluid_cons::momentum::name()].second;
  const int ceng = imap[fluid_cons::energy::name()].first;
  auto idx_E = imap.GetFlatIdx(radmoment_cons::E::name(), false);
  auto idx_F = imap.GetFlatIdx(radmoment_cons::F::name(), false);
  std::vector<std::string> diag_vars(
      {diagnostic_variables::divf::name(), diagnostic_variables::r_divf::name()});
  PackIndexMap imap_diag;
  auto diag = rc->PackVariables(diag_vars, imap_diag);
  auto idx_r_divf = imap_diag.GetFlatIdx(diagnostic_variables::r_divf::name(), false);

  StateDescriptor *rad = pm->packages.Get("radiation").get();
  int num_species = 0;
  if (idx_E.IsValid()) {
    num_species = rad->Param<int>("num_species");
  }

  // TODO(JMM): If we expose a way to get cellbounds from the mesh or
  // meshdata object, that would be better.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CopyDivF", DevExecSpace(), 0, divf.GetDim(5) - 1, 0,
      divf.GetDim(3) - 1, 0, divf.GetDim(2) - 1, 0, divf.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        diag(b, 0, k, j, i) = divf(b, crho, k, j, i);
        diag(b, 1, k, j, i) = divf(b, cmom_lo, k, j, i);
        diag(b, 2, k, j, i) = divf(b, cmom_lo + 1, k, j, i);
        diag(b, 3, k, j, i) = divf(b, cmom_lo + 2, k, j, i);
        diag(b, 4, k, j, i) = divf(b, ceng, k, j, i);
        for (int ispec = 0; ispec < num_species; ispec++) {
          if (idx_E.IsValid()) {
            diag(b, idx_r_divf(ispec, 0), k, j, i) = divf(b, idx_E(ispec), k, j, i);
            diag(b, idx_r_divf(ispec, 1), k, j, i) = divf(b, idx_F(ispec, 0), k, j, i);
            diag(b, idx_r_divf(ispec, 2), k, j, i) = divf(b, idx_F(ispec, 1), k, j, i);
            diag(b, idx_r_divf(ispec, 3), k, j, i) = divf(b, idx_F(ispec, 2), k, j, i);
          }
        }
      });
  return TaskStatus::complete;
}

#endif

} // namespace fluid

#endif // FLUID_HPP_
