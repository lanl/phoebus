#include "radiation.hpp"

namespace radiation {

TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, const double t0) {
  return TaskStatus::complete;
}

TaskStatus MonteCarloTransport(MeshBlock *pmb, const double dt,
                               const double t0) {
  return TaskStatus::complete;
}
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks) {
  return TaskStatus::complete;
}

} // namespace radiation
