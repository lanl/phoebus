// © 2021. Triad National Security, LLC. All rights reserved.  This
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

#ifndef RADIATION_MOCMC_HPP_
#define RADIATION_MOCMC_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace radiation {

class MOCMCBoundNoWork : public parthenon::ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void
  Apply(const int n, double &x, double &y, double &z,
        const parthenon::SwarmDeviceContext &swarm_d) const override {}
};

inline std::unique_ptr<parthenon::ParticleBound,
                       parthenon::DeviceDeleter<parthenon::DevMemSpace>>
SetSwarmNoWorkBC() {
  return parthenon::DeviceAllocate<MOCMCBoundNoWork>();
}

}; // namespace radiation

#endif // RADIATION_MOCMC_HPP_
