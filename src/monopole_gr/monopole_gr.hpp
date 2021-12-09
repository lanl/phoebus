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

#ifndef MONOPOLE_GR_MONOPOLE_GR_HPP_
#define MONOPOLE_GR_MONOPOLE_GR_HPP_

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
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"

using namespace parthenon::package::prelude;

// TODO(JMM): Clean up notation? Or is it fine?
/*
  ASSUMPTIONS:
  -------------
  ADM split:

  ds^2 = (-alpha^2 + beta_i beta^i) dt^2 + 2 beta_i dt dx^i + gamma_{ij} dx^i dx^j

  where alpha is lapse, beta is shift, gamma is 3-metric

  with spherically symmetric ansatz:

  ^(3)ds^2 = gamma_{ij} dx^i dx^j
           = a^2 dr^2 + b^2 r^2 dOmega^2

  Areal coordinates, meaning surface area of sphere = pi r^2
  => b = 1

  Extrinsic curvature defines time-evolution of metric:

  (partial_t - Lie_beta) gamma_{ij} = -2 alpha K_{ij}

  We assume trace of extrinsic curvature vanishes:

  K = K^i_i = 0

  This is so-called "maximal slicing" and provides
  singularity-avoiding coordinates.

  Note that:

  Lie_beta gamma_{ij} = gamma_{kj} \partial_i beta^k
                        + beta^k \partial_k gamma_{ij}
                        + gamma_{ki} \partial_j beta^k

  Trace-free + spherical symmetry implies that
  only non-vanishing components of extrinsic curvature are:

  K^th_th = K^ph_ph = -(1/2) K^r_r

  so we work with K^r_r


  GOVERNING EQUATIONS, SUMMARY:
  ------------------------------
  gamm_{ij} = diag(a^2, r^2, r^2 sin^2(th))

  K^i_j = K^r_r diag(1, -1/2, -1/2)

  da/dr = a (4 + a^2 (-4 + r^2 (-3 (K^r_r)^2 + 32 pi rho)))/(8 r);

  d K^r_r/dr = 8 pi a^2 j^r - (3/r) K^r_r

  r d^2 alpha/dr^2 = a^2 r alpha ((3/2) (K^r_r)^2 + 4 pi (rho + S)) +
  ((1/a)(da/dr)r-2)dalpha/dr

  beta^r = -(1/2) alpha r K^r_r


  BOUNDARY CONDITIONS
  -------------------
  at r = 0:
     da/dr = 0
     d K^r_r /dr = 0
     dalpha/dr = 0

  Examination of the equations shows that this translates to:
     a = 1
     K = 0
     dalpha/dr = 0

  at outer boundary (idealy r = infinity):
     da/dr = (a - a^3) / 2 r OR a = 1
     K^r_r = 0
     alpha = 1 - r (dalpha/dr) OR alpha = 1


  TIME DERIVATIVES
  ----------------

  The ADM constraint equations are sufficient to determine a, K,
  alpha, beta. However, the Christoffel coefficients depend on
  derivatives of these quantities. We can determine the spatial
  derivatives by taking gradients or using the above equations. For
  time derivatives, we rely on the Einstein evolution equations:

  (\partial_t - Lie_beta) g_{mu nu} = - 2 alpha K_{mu nu}

  (\partial_t - Lie_beta) K_{i j} = -D_i D_j alpha + alpha [^{(3)}R_{ij} + K K_{ij} - 2
  K_{ik} K^k_j] + 4 pi alpha [gamma_{ij} (S - rho) - 2 S_{ij}]

  where here "D" is the covariant derivative operator on a hypersurface.

  In particular, this allows us to derive that

  da/dt = (da/dr) beta^r + a (d beta^r/dr) - alpha a K^r_r

  d alpha/dr = beta^r [ a beta^r (da/dt)/alpha + a^2 K^r_r beta^r
                        + 2 (da/dr) (1 - (beta^r)^2) - 2 a^2 beta^r (d beta^r/dr)/alpha ]

  d K^r_r/dt = beta^r (d K^r_r/dr) - (1/a^2) (d^2 alpha/dr^2) + alpha [(2/(a^3 r)) (da/dr)
                - r (K^r_r)^2] + 4 pi alpha [S - rho - 2 S^r_r]

  d K^r_r/dt = beta^r (d K^r_r/dr) - (d^2 alpha/dr^2)/a^2
              + a [2 (da/dr)/(a^2 r) - 4 (K^r_r)^2 ] + 4 pi alpha (s - rho - 2 S^r_r)


  MATTER COMPONENTS:
  ------------------

  In ADM formulation:

  rho    = TOTAL mass-energy.
  j^i    = CONSERVED 3-momentum vector
  S_{ij} = 3-stress tensor
  ---------------------------------------
  rho    = n^mu n^nu T_{mu nu} = tau + D
  j^i    = - P^{i mu} n^nu T_{mu nu} = S^i
  S_{ij} = P^mu_i P^nu_i T_{munu}

  where n^mu is unit normal of hypersurface, P^{i mu} is projector

  Translates to (in Valencia variables):

  rho    = tau + D
  j^i    = S^i
  Trc(S) = (rho_0 h + b^2)W^2 + 3(P + b^2/2) - b^i b_i

  In case of no B fields,

  trc(S) = (tau + D) + 4 P - rho_0 h

  where rho_0 here is primitive density and h is enthalpy.

  We also need S^r_r = T^r_r for the Christoffel symbols:
  T^r_r = (rho + u + P + b^2) u^r u_r + (P + (1/2) b^2) - b^r b_r
 */

namespace MonopoleGR {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus MatterToHost(StateDescriptor *pkg);

TaskStatus IntegrateHypersurface(StateDescriptor *pkg);

TaskStatus LinearSolveForAlpha(StateDescriptor *pkg);

TaskStatus SpacetimeToDevice(StateDescriptor *pkg);

void DumpToTxt(const std::string &filename, StateDescriptor *pkg);

namespace impl {
template <bool IS_CART, typename EnergyMomentum, typename Geometry, typename Pack>
void GetMonopoleVarsHelper(const EnergyMomentum &tmunu, const Geometry &geom,
                           const Pack &p, const int b, const int k, const int j,
                           const int i, const int crho, const int cmom_lo, const int ceng,
                           Real &rho0, Real &jr, Real &srr, Real &trcs);
} // namespace impl

// TODO(JMM): Try hierarchical parallelism too
template <typename Data>
TaskStatus InterpolateMatterTo1D(Data *rc) {
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
  PARTHENON_DEBUG_REQUIRE(is_monopole_sph || is_monopole_cart,
                          "Monopole solver requires monopole geometry");

  auto matter = params.Get<Matter_t>("matter");

  // Templated on container type
  auto tmunu = fluid::BuildStressEnergyTensor(rc);
  auto geom = Geometry::GetCoordinateSystem(rc);

  std::vector<std::string> vars(
      {fluid_cons::density, fluid_cons::momentum, fluid_cons::energy});

  // Available in all Container types
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int crho = imap[fluid_cons::density].first;
  const int cmom_lo = imap[fluid_cons::momentum].first;
  const int ceng = imap[fluid_cons::energy].first;

  const int nblocks = pack.GetDim(5);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MonpoleGR::InterpolateMatterTo1D", DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // First compute the relevant conserved/prim vars

        const Real D = pack(b, crho, k, j, i);
        const Real S[] = {pack(b, cmom_lo, k, j, i), pack(b, cmom_lo + 1, k, j, i),
                          pack(b, cmom_lo + 2, k, j, i)};
        const Real tau = pack(b, ceng, k, j, i);
        const Real rho0 = tau + D;
      });
}

// TODO(JMM): This doesn't really work with the hierarchical parallelism model,
// at least not without some changes? Is memory locality shot?
// Can we vectorize this better?
// Technically IS_CART is not necessary. This could be resolved
// with type resolution and a constexpr if based on other templated types
namespace impl {
template <bool IS_CART, typename EnergyMomentum, typename Geometry, typename Pack>
void GetMonopoleVarsHelper(const EnergyMomentum &tmunu, const Geometry &geom,
                           const Pack &p, const int b, const int k, const int j,
                           const int i, const int crho, const int cmom_lo, const int ceng,
                           Real &rho0, Real &jr, Real &srr, Real &trcs) {
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

  // D, S, tau in base coordinates
  const Real D = p(b, crho, k, j, i);
  const Real Scov[] = {p(b, cmom_lo, k, j, i), p(b, cmom_lo + 1, k, j, i),
    p(b, cmom_lo + 2, k, j, i)};
  const Real tau = p(b, ceng, k, j, i);

  // Lower Tmunu
  Real TConCov[ND][ND] = {0};
  SPACETIMELOOP(mu) {
    SPACETIMELOOP2(nu, nup) {
      TConCov[mu][nu] += gcov[nu][nup]*Tcon[mu][nup];
    }
  }
  // Raise S
  Real Scon[NS] = {0};
  SPACELOOP2(i, ip) {
    Scon[i] += gamcon[i,ip]*Scov[ip];
  }

  trcs = 0; // initialize for summing
  rho0 = tau + D; // scalar so coord indep

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
    geom.Cart2Sph(X1, X2, X3, r, th, ph);
    geom.S2C(r, th, ph, s2c);
    geom.C2S(X1, X2, X3, c2s);

    // J
    jr = 0;
    SPACELOOP(i) {
      jr += c2s[1][i+1]*Scon[i];
    }
    // S
    SPACELOOP2(i,ip) {
      SPACELOOP2(j, jp) {
        ssph[ip][jp] += c2s[i+1][ip+1]*s2c[j+1][jp+1]*TConCov[ip+1][jp+1];
      }
    }
    srr = ssph[1][1];
    SPACELOOP(i) {
      trcs += ssph[i][i];
    }
  } else {
    jr = Scon[0];
    srr = TConCov[1][1];
    SPACELOOP(i) {
      trcs += TConCov[i+1][i+1];
    }
  }
}
} // namespace impl

} // namespace MonopoleGR

#endif // MONOPOLE_GR_MONOPOLE_GR_HPP_
