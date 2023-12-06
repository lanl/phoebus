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

void ReportError(Coordinates_t &coords, VariablePack<Real> &v, const int vindex,
                 Real f(const Real)) {

  Real max_error = 0.0;
  Real x0, val0, v0;
  /*parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "ReportError",
    DevExecSpace(), 0, v.GetDim(3)-1, 0, v.GetDim(2)-1, 0, v.GetDim(1)-1,
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &merr) {*/
  int k = 0;
  int j = 0;
  for (int i = 0; i < v.GetDim(1); i++) {
    const Real x = coords.Xc<1>(i);
    const Real val = f(x);
    const Real err = std::abs(val - v(vindex, k, j, i)) / val;
    if (err > max_error) {
      max_error = err;
      x0 = x;
      val0 = val;
      v0 = v(vindex, k, j, i);
    }
    // merr = (err > merr ? err : merr);
  } //, Kokkos::Max<Real>(max_error));

  printf("Max error [%d] = %g    %g %g %g\n", vindex, max_error, x0, val0, v0);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                              fluid_prim::energy::name(), fluid_prim::bfield::name(),
                              fluid_prim::ye::name(), fluid_prim::pressure::name(),
                              fluid_prim::temperature::name(), fluid_prim::gamma1},
                             imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iprs = imap[fluid_prim::pressure::name()].first;
  const int itmp = imap[fluid_prim::temperature::name()].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

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
        v(irho, k, j, i) = rho_of_x(x);
        v(ieng, k, j, i) = u_of_x(x);
        v(ivlo, k, j, i) = v1_of_x(x);
        v(ivlo + 1, k, j, i) = v2_of_x(x);
        v(ivlo + 2, k, j, i) = v3_of_x(x);
        v(ib_lo, k, j, i) = b1_of_x(x);
        v(ib_lo + 1, k, j, i) = b2_of_x(x);
        v(ib_lo + 2, k, j, i) = b3_of_x(x);
        v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(
            v(irho, k, j, i), v(ieng, k, j, i) / v(irho, k, j, i));
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            v(irho, k, j, i), v(ieng, k, j, i) / v(irho, k, j, i));
        v(igm1, k, j, i) =
            eos.BulkModulusFromDensityTemperature(v(irho, k, j, i), v(itmp, k, j, i)) /
            v(iprs, k, j, i);
      });

  for (int i = 0; i < 100; i++) {
    fluid::PrimitiveToConserved(rc.get());
    fluid::ConservedToPrimitive(rc.get());
  }

  ReportError(coords, v, irho, rho_of_x);
  ReportError(coords, v, ivlo, v1_of_x);
  ReportError(coords, v, ivlo + 1, v2_of_x);
  ReportError(coords, v, ivlo + 2, v3_of_x);
  ReportError(coords, v, ieng, u_of_x);
  ReportError(coords, v, ib_lo, b1_of_x);
  ReportError(coords, v, ib_lo + 1, b2_of_x);
  ReportError(coords, v, ib_lo + 2, b3_of_x);

  exit(1);
}

} // namespace p2c2p
