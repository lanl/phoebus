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

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include <singularity-opac/neutrinos/opac_neutrinos.hpp>

#include "compile_constants.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/constants.hpp"

#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

namespace radiation {

using pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>;
using singularity::RadiationType;

constexpr RadiationType species[3] = {RadiationType::NU_ELECTRON, RadiationType::NU_ELECTRON_ANTI, RadiationType::NU_HEAVY};

// TODO(BRR) Utilities that should be moved
#define SMALL (1.e-200)
KOKKOS_INLINE_FUNCTION Real GetLorentzFactor(Real v[4],
                                             const Geometry::CoordSysMeshBlock &system,
                                             CellLocation loc, const int k, const int j,
                                             const int i) {
  Real W = 1;
  Real gamma[Geometry::NDSPACE][Geometry::NDSPACE];
  system.Metric(loc, k, j, i, gamma);
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    for (int m = 1; m < Geometry::NDFULL; ++m) {
      W -= v[l] * v[m] * gamma[l - 1][m - 1];
    }
  }
  W = 1. / std::sqrt(std::abs(W) + SMALL);
  return W;
}

KOKKOS_INLINE_FUNCTION void GetFourVelocity(Real v[4],
                                            const Geometry::CoordSysMeshBlock &system,
                                            CellLocation loc, const int k, const int j,
                                            const int i, Real u[Geometry::NDFULL]) {
  Real beta[Geometry::NDSPACE];
  Real W = GetLorentzFactor(v, system, loc, k, j, i);
  Real alpha = system.Lapse(loc, k, j, i);
  system.ContravariantShift(loc, k, j, i, beta);
  u[0] = W / (std::abs(alpha) + SMALL);
  for (int l = 1; l < Geometry::NDFULL; ++l) {
    u[l] = W * v[l - 1] - u[0] * beta[l - 1];
  }
}

KOKKOS_INLINE_FUNCTION
Real LinearInterpLog(Real x, int k, int j, int i, ParArrayND<Real> table, Real lx_min,
                     Real dlx) {
  Real lx = log(x);
  Real dn = (lx - lx_min) / dlx;
  int n = static_cast<int>(dn);
  dn = dn - n;
  return (1. - dn) * table(n, k, j, i) + dn * table(n + 1, k, j, i);
}

// Choice of RNG
typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

enum class NeutrinoSpecies { Electron, ElectronAnti, Heavy };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt);

// Optically thin cooling function
TaskStatus CoolingFunctionCalculateFourForce(MeshBlockData<Real> *rc, const double dt);

// Monte Carlo transport
TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, MeshBlockData<Real> *rc,
                                     SwarmContainer *sc, const double t0,
                                     const double dt);
TaskStatus MonteCarloTransport(MeshBlock *pmb, MeshBlockData<Real> *rc,
                               SwarmContainer *sc, const double t0, const double dt);
TaskStatus MonteCarloStopCommunication(const BlockList_t &blocks);

// Mark all MPI requests as NULL / initialize boundary flags.
TaskStatus InitializeCommunicationMesh(const std::string swarmName,
                                       const BlockList_t &blocks);

Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

} // namespace radiation

#endif // RADIATION_HPP_
