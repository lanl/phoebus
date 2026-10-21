// © 2026. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

// Constrained transport for the face-centered magnetic field.

#ifndef FLUID_B_CT_HPP_
#define FLUID_B_CT_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

namespace b_ct {

// Face-centered B -> cell-centered primitive/conserved B.
TaskStatus BlockFaceToCell(MeshBlockData<Real> *rc, IndexDomain domain);
TaskStatus MeshFaceToCell(MeshData<Real> *md, IndexDomain domain);

// Edge EMF from Riemann fluxes. In 1D this is the direct X1 Riemann EMF.
TaskStatus CalculateEMF(MeshData<Real> *md);

// Apply physical EMF boundary conditions.
TaskStatus BoundaryEMF(MeshData<Real> *md, bool coarse);

// EMF circulation -> face-field source.
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt, IndexDomain domain);

// RK mixing and update for face fields.
TaskStatus AverageFaceField(MeshData<Real> *mc0, MeshData<Real> *mbase, Real beta);
TaskStatus UpdateFaceField(MeshData<Real> *mc0, MeshData<Real> *mdudt, Real beta_dt,
                           MeshData<Real> *mc1);

// Face-field divergence diagnostics.
Real MaxDivB(MeshData<Real> *md);
TaskStatus CalcDivB(MeshBlockData<Real> *rc);

// EMF boundary treatments called by BoundaryEMF.
void ZeroBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);
void AverageBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);

// Fill face-field ghosts; reflection zeros normal B on the boundary.
void OutflowFaceB(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);
void ReflectFaceB(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);

} // namespace b_ct

#endif // FLUID_B_CT_HPP_
