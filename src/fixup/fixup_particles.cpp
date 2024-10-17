// © 2021-2023. Triad National Security, LLC. All rights reserved.
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

#include <cmath>

#include "fixup.hpp"

#include "geometry/analytic_system.hpp"
#include "geometry/cached_system.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "geometry/mckinney_gammie_ryan.hpp"
#include "geometry/modified_system.hpp"
#include "geometry/spherical_kerr_schild.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/linear_algebra.hpp"
#include "phoebus_utils/variables.hpp"

namespace fixup {

/**
 * PurgeParticles machinery to remove particles when criteria are met.
 * Current criteria: particle falls into event horizon.
 **/
TaskStatus PurgeParticles(MeshBlockData<Real> *rc, const std::string swarmName) {

  /* only do this when FMKS is used. */
  if constexpr (std::is_same<PHOEBUS_GEOMETRY, Geometry::FMKS>::value) {
    auto *pmb = rc->GetParentPointer();
    // auto &swarm = rc->swarm_data.Get()->Get(swarmName);
    auto &swarm = rc->GetSwarmData()->Get(swarmName);

    auto &x = swarm->Get<Real>("x").Get();
    auto &y = swarm->Get<Real>("y").Get();
    auto &z = swarm->Get<Real>("z").Get();

    auto swarm_d = swarm->GetDeviceContext();

    auto &pars = pmb->packages.Get("geometry")->AllParams();
    const Real Rh = pars.Get<Real>("Rh");
    const Real xh = std::log(Rh);

    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "fixup::PurgeParticles", 0, max_active_index - 1, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {

            if (x(n) <= xh) {
              swarm_d.MarkParticleForRemoval(n);
            }

            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          }
        }); // par for

    swarm->RemoveMarkedParticles();
  }
  return TaskStatus::complete;
} // PurgeParticles

} // namespace fixup
