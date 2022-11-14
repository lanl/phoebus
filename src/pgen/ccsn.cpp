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

#include "ccsn/ccsn.hpp"
#include "geometry/geometry.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"

// CCSN explosion model from a 1-d stellar model

namespace ccsn {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const bool is_monopole_cart =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleCart));
  const bool is_monopole_sph =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph));

  auto ccsn_pkg = pmb->packages.Get("ccsn");
  auto monopole_pkg = pmb->packages.Get("monopole_gr");
  auto eos_pkg = pmb->packages.Get("eos");
  const auto enable_ccsn = ccsn_pkg->Param<bool>("enabled");
  const auto enable_monopole = monopole_pkg->Param<bool>("enable_monopole_gr");
  if (!(enable_ccsn && enable_monopole)) {
    PARTHENON_THROW("CCSN problem generator requires ccsn and monopole GR packages");
  }
  if (!(is_monopole_cart || is_monopole_sph)) {
    PARTHENON_THROW("CCSN problem generator requires a MonopoleGR metric.");
  }

  auto &rc = pmb->meshblock_data.Get();

  auto rgrid = monopole_pkg->Param<MonopoleGR::Radius>("radius");
  auto base = ccsn_pkg->Param<CCSN::State_t>("ccsn_state_interp_d");
  auto coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS");

  // floors from pin
  const auto pmin = ccsn_pkg->Param<Real>("pmin");
  const auto rhomin = ccsn_pkg->Param<Real>("rhomin");
  const auto epsmin = ccsn_pkg->Param<Real>("epsmin");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto geom = Geometry::GetCoordinateSystem(rc.get());

  PackIndexMap imap;

  auto v = rc->PackVariables({fluid_prim::density, fluid_prim::velocity,
                              fluid_prim::energy, fluid_prim::bfield, fluid_prim::ye,
                              fluid_prim::pressure, fluid_prim::temperature},
                             imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int iblo = imap[fluid_prim::bfield].first;
  const int ibhi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;

  pmb->par_for(
      "Phoebus::ProblemGenerator::CCSN", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real r;
        if (is_monopole_sph) {
          r = coords.x1v(k, j, i);
        } else { // Cartesian
          const Real x1 = coords.x1v(k, j, i);
          const Real x2 = coords.x2v(k, j, i);
          const Real x3 = coords.x3v(k, j, i);
          r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        }
        // interp per mesh block here? or pull data from model_1d
        Real rho = std::max(rhomin, MonopoleGR::Interpolate(r, base, rgrid, CCSN::RHO));
        Real eps = std::max(epsmin, MonopoleGR::Interpolate(r, base, rgrid, CCSN::EPS));
        Real P = std::max(pmin, MonopoleGR::Interpolate(r, base, rgrid, CCSN::PRES));

        // TODO(JMM): Add lambdas, Ye, etc
        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = eps * rho;
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(rho, eps);
        for (int d = 0; d < 3; ++d) {
          v(ivlo + d, k, j, i) = 0.0;
        }
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace ccsn
