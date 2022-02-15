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

#ifndef MONOPOLE_GR_INTERP_3D_TO_1D_HPP_
#define MONOPOLE_GR_INTERP_3D_TO_1D_HPP_

// stdlib
#include <memory>
#include <typeinfo>

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Phoebus
#include "fluid/tmunu.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "monopole_gr/monopole_gr_base.hpp"
#include "monopole_gr/monopole_gr_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;

namespace MonopoleGR {

namespace impl {
// Gets ADM mass rho0, adm momentum in r, jr, the rr component of S^i_j, and the trace
// of the spatial stress tensor. Applies coordinate transforms if necessary.
template <bool IS_CART, typename EnergyMomentum, typename Geometry, typename Pack>
KOKKOS_INLINE_FUNCTION void
GetMonopoleVarsHelper(const EnergyMomentum &tmunu, const Geometry &geom, const Pack &p,
                      const int b, const int k, const int j, const int i, Real &rho0,
                      Real &jr, Real &srr, Real &trcs);
// Gets the radius r of a cell, as well as its width in radius dr, and
// its volume element dv. Computes this for both Cartesian and
// spherical coords.
template <bool IS_CART, typename Pack>
PORTABLE_INLINE_FUNCTION void
GetCoordsAndCellWidthsHelper(const Pack &p, const int b,
                             const int k, const int j, const int i, Real &r, Real &th,
                             Real &ph, Real &dr, Real &dth, Real &dph, Real &dv);

} // namespace impl

// TODO(JMM): Try hierarchical parallelism too. Not sure how to write
// it, but the most performant version of this function is DEFINITELY
// not what I've written here.
template <typename Data>
TaskStatus InterpolateMatterTo1D(Data *rc) {
  using namespace Interp3DTo1D;
  using namespace impl;

  // Available in both mesh and meshblock
  std::shared_ptr<StateDescriptor> const &pkg =
      rc->GetParentPointer()->packages.Get("monopole_gr");
  auto &params = pkg->AllParams();

  auto enabled = params.Get<bool>("enable_monopole_gr");
  if (!enabled) return TaskStatus::complete;

  constexpr bool is_monopole_sph =
      std::is_same<PHOEBUS_GEOMETRY, Geometry::MonopoleSph>::value;
  constexpr bool is_monopole_cart =
      std::is_same<PHOEBUS_GEOMETRY, Geometry::MonopoleCart>::value;
  PARTHENON_REQUIRE(is_monopole_sph || is_monopole_cart,
                    "Monopole solver requires monopole geometry");

  auto matter = params.Get<Matter_t>("matter_cells");
  auto vols = params.Get<Volumes_t>("integration_volumes");
  auto radius1d = params.Get<Radius>("radius");
  auto npoints = params.Get<int>("npoints");

  // Initialize 1D arrays to zero
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "monopole_gr prepare for interp",
      parthenon::DevExecSpace(), 0, npoints - 1, KOKKOS_LAMBDA(const int i) {
        for (int v = 0; v < NMAT; ++v) {
          matter(v, i) = 0;
        }
        vols(i) = 0;
      });

  // Templated on container type
  auto tmunu = fluid::BuildStressEnergyTensor(rc);
  auto geom = Geometry::GetCoordinateSystem(rc);

  // I just need the pack for the coords object,
  // but I may want these quantities in a future
  // iteration, so I ask for them here.
  std::vector<std::string> vars({fluid_cons::density,
      fluid_cons::energy,
      fluid_cons::momentum});
  // PackIndexMap imap;
  auto pack = rc->PackVariables(vars);

  // Available in all Container types
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  const int nblocks = pack.GetDim(5);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MonpoleGR::InterpolateMatterTo1D", DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // TODO(JMM): Should some of this be precomputed? Or should it
        // be computed here?  With liberal use of scratch arrays this
        // kernel could be broken up and maybe written in a way that
        // vectorizes better. I don't know what the right thing is
        // though. Just gotta try stuff.

        // First compute the relevant conserved/prim vars
        Real matter_loc[4];
        GetMonopoleVarsHelper<is_monopole_cart>(
          tmunu, geom, pack, b, k, j, i, matter_loc[Matter::RHO],
          matter_loc[Matter::J_R], matter_loc[Matter::Srr],
	  matter_loc[Matter::trcS]);
        // Next get coords and grid spacing
        Real r, th, ph, dr, dth, dph, dv;
        GetCoordsAndCellWidthsHelper<is_monopole_cart>(pack,
						       b, k, j, i, r, th, ph,
                                                       dr, dth, dph, dv);

        // Bounds in the 1d grid We're wasteful here because I'm
        // paranoid. Need to make sure we don't need miss any cells in
        // the 1d grid.
        // No correctness issue from making this too big,
        int i1dleft = radius1d.index(r - dr) - 1;
        int i1dright = radius1d.index(r + dr) + 1;

        // Loop through the 1d grid, and do the thing
        // TODO(JMM): Thanks I hate it.
        Real dr1d = radius1d.dx();
        for (int i1d = i1dleft; i1d <= i1dright; ++i1d) {
          Real r1d = radius1d.x(i1d);
          Real weight = GetVolIntersectHelper(r1d, dr1d, r, dr) * dv;
          // Yucky atomics
          Kokkos::atomic_add(&vols(i1d), weight);
          for (int v = 0; v < NMAT; ++v) {
            Kokkos::atomic_add(&matter(v, i1d), matter_loc[v] * weight);
          }
        }
      });
  return TaskStatus::complete;  
}

// TODO(JMM): This doesn't really work with the hierarchical parallelism model,
// at least not without some changes? Is memory locality shot?
// Can we vectorize this better?
// Technically IS_CART is not necessary. This could be resolved
// with type resolution and a constexpr if based on other templated types
// Geometry_t is the geometry class in this case. Geometry is a namespace.
namespace impl {
template <bool IS_CART, typename EnergyMomentum, typename Geometry_t, typename Pack>
KOKKOS_INLINE_FUNCTION void
GetMonopoleVarsHelper(const EnergyMomentum &tmunu, const Geometry_t &geom, const Pack &p,
                      const int b, const int k, const int j, const int i, Real &rho0,
                      Real &jr, Real &srr, Real &trcs) {
  // Tmunu and gcov
  constexpr int ND = Geometry::NDFULL;
  constexpr int NS = Geometry::NDSPACE;
  constexpr auto loc = CellLocation::Cent;
  Real Tcon[ND][ND];
  tmunu(Tcon, b, k, j, i);
  Real gcov[ND][ND];
  geom.SpacetimeMetric(loc, b, k, j, i, gcov);
  Real gamcon[NS][NS];
  geom.MetricInverse(loc, b, k, j, i, gamcon);
  Real beta[NS];
  geom.ContravariantShift(loc, b, k, j, i, beta);
  const Real alpha = geom.Lapse(loc, b, k, j, i);
  // const Real gdet = geom.DetGamma(loc, b, k, j, i);

  // Get rho and S before lowering T
  Real Scon[NS];
  rho0 = alpha * alpha * Tcon[0][0];
  SPACELOOP(d) { Scon[d] = -alpha * Tcon[0][d+1] + beta[d] * Tcon[0][0]; }

  // Lower Tmunu
  Real TConCov[ND][ND] = {0};
  SPACETIMELOOP(mu) {
    SPACETIMELOOP2(nu, nup) { TConCov[mu][nu] += gcov[nu][nup] * Tcon[mu][nup]; }
  }

  trcs = 0; // initialize for summing

  // Only one branch is resolved at compile time
  // Thanks to the magic of templates and C++11.
  // constexpr if would be better though
  if (IS_CART) {
    Real ssph[NS][NS] = {0};
    Real s2c[ND][ND], c2s[ND][ND];
    Real r, th, ph;

    parthenon::Coordinates_t coords = p.GetCoords(b);
    Real X1 = coords.x1v(i);
    Real X2 = coords.x1v(j);
    Real X3 = coords.x1v(k);
    // TODO(JMM): I kinda don't like this design. Should I redesign
    // the coordinate transforms in the monopole class to be
    // self-contained?
    Geometry::MonopoleCoordTransforms::Cart2Sph(X1, X2, X3, r, th, ph);
    Geometry::MonopoleCoordTransforms::S2C(r, th, ph, s2c);
    Geometry::MonopoleCoordTransforms::C2S(X1, X2, X3, r, c2s);

    // J
    jr = 0;
    SPACELOOP(d) { jr += c2s[1][d + 1] * Scon[d]; }
    // S
    SPACELOOP2(l, ip) {
      SPACELOOP2(m, jp) {
        ssph[ip][jp] += c2s[l + 1][ip + 1] * s2c[m + 1][jp + 1] *
                        (TConCov[ip + 1][jp + 1] + beta[ip] * TConCov[0][jp + 1]);
      }
    }
    srr = ssph[0][0];
    SPACELOOP(d) { trcs += ssph[d][d]; }
  } else {
    jr = Scon[0];
    srr = TConCov[1][1] + beta[0] * TConCov[0][1];
    SPACELOOP(d) { trcs += (TConCov[d + 1][d + 1] + beta[d] * TConCov[0][d + 1]); }
  }
}

template <bool IS_CART, typename Pack>
PORTABLE_INLINE_FUNCTION void
GetCoordsAndCellWidthsHelper(const Pack &p, const int b,
                             const int k, const int j, const int i, Real &r, Real &th,
                             Real &ph, Real &dr, Real &dth, Real &dph, Real &dv) {
  const parthenon::Coordinates_t &coords = p.GetCoords(b);
  if (IS_CART) {
    Geometry::MonopoleCoordTransforms::Cart2Sph(
      coords.x1v(k, j, i), coords.x2v(k, j, i), coords.x3v(k, j, i),
      coords.Dx(1, k, j, i), coords.Dx(2, k, j, i), coords.Dx(3, k, j, i), r,
      th, ph, dr, dth, dph);
    dv = coords.Volume(k, j, i);
  } else {
    Interp3DTo1D::GetCoordsAndDerivsSph(k, j, i, coords,
      r, th, ph, dr, dth, dph, dv);
  }
}

} // namespace impl
} // namespace MonopoleGR

#endif // MONOPOLE_GR_INTERP_3D_TO_1D_HPP_
