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

#ifndef RADIATION_HPP_
#define RADIATION_HPP_

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

#include "compile_constants.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/constants.hpp"

#include "geometry/geometry.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

using namespace phoebus;

namespace radiation {

enum class ParticleResolution { emitted = 0, absorbed = 1, scattered = 2, total = 3 };

enum class MOCMCRecon { kdgrid };

enum class MOCMCBoundaries { outflow, fixed_temp, periodic };

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
using singularity::RadiationType;

constexpr RadiationType species[3] = {
    RadiationType::NU_ELECTRON, RadiationType::NU_ELECTRON_ANTI, RadiationType::NU_HEAVY};
constexpr int NumRadiationTypes = 1;

KOKKOS_INLINE_FUNCTION
Real LogLinearInterp(Real x, int sidx, int k, int j, int i, ParArrayND<Real> table,
                     Real lx_min, Real dlx) {
  Real lx = log(x);
  Real dn = (lx - lx_min) / dlx;
  int n = static_cast<int>(dn);
  dn = dn - n;
  return (1. - dn) * table(n, sidx, k, j, i) + dn * table(n + 1, sidx, k, j, i);
}

// Choice of RNG
typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const Real dt);

Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

// Cooling function tasks
TaskStatus CoolingFunctionCalculateFourForce(MeshBlockData<Real> *rc, const Real dt);

// Monte Carlo tasks
TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                     SwarmContainer *sc, const Real t0, const Real dt);
TaskStatus MonteCarloTransport(MeshBlock *pmb, MeshBlockData<Real> *rc,
                               SwarmContainer *sc, const Real dt);
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks);

TaskStatus MonteCarloUpdateTuning(Mesh *pmesh, std::vector<Real> *resolution,
                                  const Real t0, const Real dt);

TaskStatus MonteCarloUpdateParticleResolution(Mesh *pmesh, std::vector<Real> *tuning);

TaskStatus MonteCarloEstimateParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                       SwarmContainer *sc, const Real t0, const Real dt,
                                       Real *dNtot);

TaskStatus MonteCarloCountCommunicatedParticles(MeshBlock *pmb,
                                                int *particles_outstanding);

TaskStatus InitializeCommunicationMesh(const std::string swarmName,
                                       const BlockList_t &blocks);

Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

// Moment tasks
template <class T>
TaskStatus MomentCon2Prim(T *rc);

template <class T>
TaskStatus MomentPrim2Con(T *rc, IndexDomain domain = IndexDomain::entire);

template <class T>
TaskStatus ReconstructEdgeStates(T *rc);

template <class T>
TaskStatus CalculateFluxes(T *rc);

template <class T>
TaskStatus CalculateGeometricSource(T *rc, T *rc_src);

template <class T>
TaskStatus MomentFluidSource(T *rc, Real dt, bool update_fluid);

template <class T>
TaskStatus MomentCalculateOpacities(T *rc);

// MOCMC tasks
template <class T>
void MOCMCInitSamples(T *rc);

template <class T>
TaskStatus MOCMCTransport(T *rc, const Real dt);

template <class T>
TaskStatus MOCMCSampleBoundaries(T *rc);

template <class T>
TaskStatus MOCMCReconstruction(T *rc);

template <class T>
TaskStatus MOCMCEddington(T *rc);

template <class T>
TaskStatus MOCMCFluidSource(T *rc, const Real dt, const bool update_fluid);

TaskStatus MOCMCUpdateParticleCount(Mesh *pmesh, std::vector<Real> *resolution);

} // namespace radiation

#endif // RADIATION_HPP_
