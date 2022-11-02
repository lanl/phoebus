// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#include <string>

#include "geometry/geometry_utils.hpp"
#include "pgen/pgen.hpp"

// A simplified toy cosmology for testing time-dependent metrics
// Assumes ds^2 = -dt^2 + a^2 delta_{ij} dx^i dx^j for delta = Chronicker Delta
// We also assume da/dt = const = C, i.e., d^2a/dt^2 = 0
// and hubble param H = (1/a)(da/dt) = C/a
// This also implies vanishing spatial curvature... i.e., flat spacetime.
// This implies d rho/dt = -3 (rho + P) H
// where here rho is the rho_0 + u, i.e., the total energy density.
// Beware this spacetime has cosmological horizons and a singularity.
// For example, t = -a0/dadt produces a = 0, i.e., the Big Bang.
// Treatment taken from Chapter 8 of Carroll

// friedmann here to avoid name collisions with FLRW geometry object
namespace friedmann {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FLRW),
                    "Problem \"friedmann\" requires \"FLRW\" geometry!");
  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density, fluid_prim::velocity, fluid_prim::energy, fluid_prim::bfield,
       fluid_prim::ye, fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1},
      imap);
  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ieng = imap[fluid_prim::energy].first;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  const Real rho = pin->GetOrAddReal("friedmann", "rho", 1.0);
  const Real eps = pin->GetOrAddReal("friedmann", "sie", 1.0);
  const Real u = rho * eps;

  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = rc->GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "Phoebus::ProblemGenerator::friedmann", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real eos_lambda[2];

        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }

        const Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, eos_lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(rho, eps, eos_lambda);

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = u;
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                           v(iprs, k, j, i);

        SPACELOOP(i) { v(ivlo + i, k, j, i) = 0; }
      });
  fluid::PrimitiveToConserved(rc.get());
}

} // namespace friedmann
