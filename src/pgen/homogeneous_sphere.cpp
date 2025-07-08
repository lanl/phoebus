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

#include <cmath>
#include <string>

#include "pgen/pgen.hpp"

namespace homogeneous_sphere {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE((typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalMinkowski)) ||
                        (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski)),
                    "Problem \"homogeneous_sphere\" requires \"SphericalMinkowski\" or "
                    "\"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  Mesh *pmesh = rc->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<fluid_prim::density, fluid_prim::velocity, fluid_prim::temperature, 
                         radmoment_prim::J, radmoment_prim::H, radmoment_internal::xi, radmoment_internal::phi>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc.get());

  auto rad_pkg = pmb->packages.Get("radiation");
  auto num_species = rad_pkg->Param<int>("num_species");
  const Real sphere_rad = pin->GetOrAddReal("homogeneous_sphere", "radius", 1.0);
  const Real rho_min = pin->GetOrAddReal("homogeneous_sphere", "rho_min", 1.e-10);
  const Real J = pin->GetOrAddReal("homogeneous_sphere", "J", 0.0);
  const Real Hx = pin->GetOrAddReal("homogeneous_sphere", "Hx", 0.0);
  const Real Hy = pin->GetOrAddReal("homogeneous_sphere", "Hy", 0.0);
  const Real Hz = pin->GetOrAddReal("homogeneous_sphere", "Hz", 0.0);
  const Real vx = pin->GetOrAddReal("homogeneous_sphere", "vx", 0.0);

  auto &coords = pmb->coords;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "Phoebus::ProblemGenerator::homogeneous_sphere", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real r = coords.Xc<1>(i);

        if (r < sphere_rad) {
          v(0, fluid_prim::density(), k, j, i) = 1.0;
        } else {
          v(0, fluid_prim::density(), k, j, i) = rho_min;
        }
        v(0, fluid_prim::temperature(), k, j, i) = 1.0;

        v(0, fluid_prim::velocity(0), k, j, i) = vx;
        v(0, fluid_prim::velocity(1), k, j, i) = 0.0;
        v(0, fluid_prim::velocity(2), k, j, i) = 0.0;

        for (int ispec = 0; ispec < num_species; ++ispec) {

          v(0, radmoment_internal::xi(ispec), k, j, i) = 0.0;
          v(0, radmoment_internal::phi(ispec), k, j, i) = acos(-1.0) * 1.000001;

          v(0, radmoment_prim::J(ispec), k, j, i) = J;
          printf("i = %i r = %e J = %e rho = %e\n", i, r, v(0, radmoment_prim::J(ispec), k, j, i),
                 v(0, fluid_prim::density(), k, j, i));
          v(0, radmoment_prim::H(ispec, 0), k, j, i) = Hx;
          v(0, radmoment_prim::H(ispec, 1), k, j, i) = Hy;
          v(0, radmoment_prim::H(ispec, 2), k, j, i) = Hz;
        }
      });

  radiation::MomentPrim2Con(rc.get(), IndexDomain::entire);
}

} // namespace homogeneous_sphere
