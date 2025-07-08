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

#include "pgen/pgen.hpp"
#include "phoebus_utils/relativity_utils.hpp"

// namespace phoebus {

namespace p2c2p {

KOKKOS_INLINE_FUNCTION
Real rho_of_x(const Real x) { return 1.0 + 0.3 * sin(x); }

KOKKOS_INLINE_FUNCTION
Real u_of_x(const Real x) { return 1.0 + 0.3 * cos(x); }

KOKKOS_INLINE_FUNCTION
Real v1_of_x(const Real x) { return 0.5 + 0.2 * sin(x - 0.3); }

KOKKOS_INLINE_FUNCTION
Real v2_of_x(const Real x) { return 0.3 + 0.2 * cos(x - 0.2); }

KOKKOS_INLINE_FUNCTION
Real v3_of_x(const Real x) { return 0.4 + 0.3 * sin(x + 0.1); }

KOKKOS_INLINE_FUNCTION
Real b1_of_x(const Real x) { return 0.3 + 0.1 * sin(x); }

KOKKOS_INLINE_FUNCTION
Real b2_of_x(const Real x) { return 0.4 + 0.2 * cos(x); }

KOKKOS_INLINE_FUNCTION
Real b3_of_x(const Real x) { return 0.5 + 0.3 * sin(x - 0.1); }


void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  namespace p = fluid_prim;
  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<p::density, p::velocity, p::energy,
                        p::bfield, p::ye, p::pressure, 
                        p::temperature, p::gamma1>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc.get());
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  pmb->par_for(
      "Phoebus::ProblemGenerator::Sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        v(0, p::density(), k, j, i) = rho_of_x(x);
        v(0, p::energy(), k, j, i) = u_of_x(x);
        v(0, fluid_prim::velocity(0), k, j, i) = v1_of_x(x);
        v(0, fluid_prim::velocity(1), k, j, i) = v2_of_x(x);
        v(0, fluid_prim::velocity(2), k, j, i) = v3_of_x(x);
        v(0, fluid_prim::bfield(0), k, j, i) = b1_of_x(x);
        v(0, fluid_prim::bfield(1), k, j, i) = b2_of_x(x);
        v(0, fluid_prim::bfield(2), k, j, i) = b3_of_x(x);
        v(0, p::pressure(), k, j, i) = eos.PressureFromDensityInternalEnergy(
            v(0, p::density(), k, j, i), v(0, p::energy(), k, j, i) / v(0, p::density(), k, j, i));
        v(0, p::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            v(0, p::density(), k, j, i), v(0, p::energy(), k, j, i) / v(0, p::density(), k, j, i));
        v(0, p::gamma1(), k, j, i) =
            eos.BulkModulusFromDensityTemperature(v(0, p::density(), k, j, i), v(0, p::temperature(), k, j, i)) /
            v(0, p::pressure(), k, j, i);
      });

  for (int i = 0; i < 100; i++) {
    fluid::PrimitiveToConserved(rc.get());
    fluid::ConservedToPrimitive(rc.get());
  }

  ReportErrorP2C2P(coords, v, p::density(0), rho_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::velocity(0), v1_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::velocity(1), v2_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::velocity(2), v3_of_x);
  ReportErrorP2C2P(coords, v, p::energy(0), u_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::bfield(0), b1_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::bfield(1), b2_of_x);
  ReportErrorP2C2P(coords, v, fluid_prim::bfield(2), b3_of_x);

  exit(1);
}

} // namespace p2c2p
