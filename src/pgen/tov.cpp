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
#include <typeinfo>

#include <utils/error_checking.hpp>

#include "geometry/geometry.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "tov/tov.hpp"

// Tohlmann-Oppenheimer-Volkov star in hydrostatic equilibrium

namespace tov {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const bool is_monopole_cart =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleCart));
  const bool is_monopole_sph =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph));

  // Velocity perturbation
  // of form v = a*r*exp(-(r-mu)^2/(2 sigma)^2)
  // amplitude
  const Real vpert_a = pin->GetOrAddReal("tov", "vpert_amp", 0);
  // center of vel pert
  const Real vpert_r = pin->GetOrAddReal("tov", "vpert_radius", 5);
  // standard deviation
  const Real vpert_s = pin->GetOrAddReal("tov", "vpert_sigma", 1);

  auto tov_pkg = pmb->packages.Get("tov");
  auto monopole_pkg = pmb->packages.Get("monopole_gr");
  auto eos_pkg = pmb->packages.Get("eos");
  const auto enable_tov = tov_pkg->Param<bool>("enabled");
  const auto enable_monopole = monopole_pkg->Param<bool>("enable_monopole_gr");
  if (!(enable_tov && enable_monopole)) {
    PARTHENON_THROW("TOV problem generator requires tov and monopole GR packages");
  }
  if (!(is_monopole_cart || is_monopole_sph)) {
    PARTHENON_THROW("TOV problem generator requires a MonopoleGR metric.");
  }

  auto &rc = pmb->meshblock_data.Get();

  auto rgrid = monopole_pkg->Param<MonopoleGR::Radius>("radius");
  auto intrinsic = tov_pkg->Param<TOV::State_t>("tov_intrinsic");

  auto coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");

  const auto pmin = tov_pkg->Param<Real>("pmin");
  const Real rhomin = pin->GetOrAddReal("tov", "rhomin", 1e-12);
  const Real epsmin = pin->GetOrAddReal("tov", "epsmin", 1e-12);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto geom = Geometry::GetCoordinateSystem(rc.get());

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity,
                              fluid_prim::energy, fluid_prim::bfield, fluid_prim::ye,
                              fluid_prim::pressure, fluid_prim::temperature,
                              fluid_prim::gamma1},
                             imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int iblo = imap[fluid_prim::bfield].first;
  const int ibhi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  pmb->par_for(
      "Phoebus::ProblemGenerator::TOV", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real r;
        if (is_monopole_sph) {
          r = coords.Xc<1>(k, j, i);
        } else { // Cartesian
          const Real x1 = coords.Xc<1>(k, j, i);
          const Real x2 = coords.Xc<2>(k, j, i);
          const Real x3 = coords.Xc<3>(k, j, i);
          r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        }
        Real rho =
            std::max(rhomin, MonopoleGR::Interpolate(r, intrinsic, rgrid, TOV::RHO0));
        Real eps =
            std::max(epsmin, MonopoleGR::Interpolate(r, intrinsic, rgrid, TOV::EPS));
        Real P = std::max(pmin, eos.PressureFromDensityInternalEnergy(rho, eps));

        // TODO(JMM): Add lambdas, Ye, etc
        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = eps * rho;
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, eps);
        v(igm1, k, j, i) =
            eos.BulkModulusFromDensityTemperature(v(irho, k, j, i), v(itmp, k, j, i)) /
            v(iprs, k, j, i);
        for (int d = 0; d < 3; ++d) {
          v(ivlo + d, k, j, i) = 0.0;
        }
        // Perturbation in velocity for testing
        if (vpert_a > 0) {
          v(ivlo, k, j, i) =
              vpert_a * r *
              std::exp(-(r - vpert_r) * (r - vpert_r) / (4 * vpert_s * vpert_s));
        }
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace tov
