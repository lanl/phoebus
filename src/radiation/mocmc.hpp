#ifndef RADIATION_MOCMC_HPP_
#define RADIATION_MOCMC_HPP_

#include <interface/swarm.hpp>

namespace radiation {

class ParticleBoundNoWork : public parthenon::ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                    const SwarmDeviceContext &swarm_d) const override {}
};

inline std::unique_ptr<parthenon::ParticleBound, parthenon::DeviceDeleter<parthenon::DevMemSpace>>
SetSwarmNoWorkBC() {
  return parthenon::DeviceAllocate<ParticleBoundNoWork>();
}

} // namespace radiation

#endif // RADIATION_MOCMC_HPP_
