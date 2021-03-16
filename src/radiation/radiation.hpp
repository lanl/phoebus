//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef RADIATION_HPP_
#define RADIATION_HPP_

#include "Kokkos_Random.hpp"

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "compile_constants.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/constants.hpp"

#include "geometry/geometry.hpp"
#include "phoebus_utils/variables.hpp"

#define CoolingFunction (0)
#define Diffusion (1)
#define M1 (2)
#define MonteCarlo (3)
#define MOCMC (4)

namespace radiation {

// TODO(BRR) Utilities that should be moved
#define SMALL (1.e-200)
KOKKOS_INLINE_FUNCTION Real GetLorentzFactor(Real v[4],
                                             const Geometry::CoordinateSystem &system,
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
                                            const Geometry::CoordinateSystem &system,
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
typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

extern parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus ApplyRadiationFourForce(MeshBlockData<Real> *rc, const double dt);

//TaskStatus CalculateRadiationFourForce(MeshBlockData<Real> *rc, const double dt);

// Optically thin cooling function
TaskStatus CalculateCoolingFunctionFourForce(MeshBlockData<Real> *rc, const double dt);

// Monte Carlo transport
TaskStatus MonteCarloSourceParticles(MeshBlock *pmb, const double t0);


enum class NeutrinoSpecies { Electron, ElectronAnti, Heavy };

// TODO(BRR) temporary opacity here
KOKKOS_INLINE_FUNCTION Real Getyf(Real Ye, NeutrinoSpecies s) {
  if (s == NeutrinoSpecies::Electron) {
    return 2. * Ye;
  } else if (s == NeutrinoSpecies::ElectronAnti) {
    return 1. - 2. * Ye;
  } else {
    return 0.;
  }
}

} // namespace radiation

#endif // RADIATION_HPP_
