// © 2026. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
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

// Magnetic-loop advection divB-preservation test, ported from AthenaPK.
// REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
// constrained transport", JCP, 205, 509 (2005).

#include "pgen/pgen.hpp"

namespace field_loop {

// Vector potential for spherical or cylindrical loops.
KOKKOS_INLINE_FUNCTION
void FieldLoopPotential(const bool sphere, const Real nx, const Real ny, const Real nz,
                        const Real wrap_period, const Real rad, const Real amp,
                        const Real x, const Real y, const Real z, Real &ax, Real &ay,
                        Real &az) {
  ax = 0.0;
  ay = 0.0;
  az = 0.0;
  if (sphere) {
    const Real r = std::sqrt(x * x + y * y + z * z);
    if (r < rad) {
      ay = amp * (rad - r);
      az = amp * (rad - r);
    }
    return;
  }

  Real r;
  if (wrap_period > 0.0) {
    Real xx = x * nz - z * nx; // in-plane coordinate perpendicular to the (x1-x3-plane)
                               // axis; see comment above.
    while (xx > 0.5 * wrap_period)
      xx -= wrap_period;
    while (xx < -0.5 * wrap_period)
      xx += wrap_period;
    const Real yy = y;
    r = std::sqrt(xx * xx + yy * yy);
  } else {
    const Real xdotn = x * nx + y * ny + z * nz;
    r = std::sqrt(std::max(0.0, x * x + y * y + z * z - xdotn * xdotn));
  }
  if (r < rad) {
    const Real A = amp * (rad - r);
    ax = A * nx;
    ay = A * ny;
    az = A * nz;
  }
}

// Components used to construct B as a discrete curl of A.
KOKKOS_INLINE_FUNCTION
Real FieldLoopAx(const bool sphere, const Real nx, const Real ny, const Real nz,
                 const Real wrap_period, const Real rad, const Real amp, const Real x,
                 const Real y, const Real z) {
  Real ax, ay, az;
  FieldLoopPotential(sphere, nx, ny, nz, wrap_period, rad, amp, x, y, z, ax, ay, az);
  return ax;
}
KOKKOS_INLINE_FUNCTION
Real FieldLoopAy(const bool sphere, const Real nx, const Real ny, const Real nz,
                 const Real wrap_period, const Real rad, const Real amp, const Real x,
                 const Real y, const Real z) {
  Real ax, ay, az;
  FieldLoopPotential(sphere, nx, ny, nz, wrap_period, rad, amp, x, y, z, ax, ay, az);
  return ay;
}
KOKKOS_INLINE_FUNCTION
Real FieldLoopAz(const bool sphere, const Real nx, const Real ny, const Real nz,
                 const Real wrap_period, const Real rad, const Real amp, const Real x,
                 const Real y, const Real z) {
  Real ax, ay, az;
  FieldLoopPotential(sphere, nx, ny, nz, wrap_period, rad, amp, x, y, z, ax, ay, az);
  return az;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"field_loop\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  auto fluid_pkg = pmb->packages.Get("fluid");
  const bool mhd = fluid_pkg->Param<bool>("mhd");
  PARTHENON_REQUIRE_THROWS(!mhd || pmb->pmy_mesh->ndim >= 2,
                           "field_loop: mhd requires ndim >= 2 (a field loop needs at "
                           "least 2 spatial dimensions)");

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density::name(), fluid_prim::velocity::name(),
       fluid_prim::energy::name(), fluid_prim::ye::name(), fluid_prim::pressure::name(),
       fluid_prim::temperature::name(), fluid_prim::gamma1::name()},
      imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure::name()].first;
  const int itmp = imap[fluid_prim::temperature::name()].first;
  const int igm1 = imap[fluid_prim::gamma1::name()].first;

  const Real rad = pin->GetReal("field_loop", "rad");
  const Real amp = pin->GetReal("field_loop", "amp");
  const Real drat = pin->GetOrAddReal("field_loop", "drat", 1.0);
  const Real P0 = pin->GetOrAddReal("field_loop", "p0", 1.0);
  PARTHENON_REQUIRE_THROWS(rad > 0.0, "field_loop/rad must be positive");

  // Flow velocity, set directly rather than derived from a single diagonal-flow
  // magnitude -- any sub-luminal direction is valid.
  const Real vx = pin->GetOrAddReal("field_loop", "vx", 0.0);
  const Real vy = pin->GetOrAddReal("field_loop", "vy", 0.0);
  const Real vz = pin->GetOrAddReal("field_loop", "vz", 0.0);
  PARTHENON_REQUIRE_THROWS(vx * vx + vy * vy + vz * vz < 1.0,
                           "field_loop: vx^2+vy^2+vz^2 must be < 1 (sub-luminal flow)");

  // Loop geometry: either a cylinder with an arbitrary axis direction, or a sphere.
  const std::string shape = pin->GetOrAddString("field_loop", "shape", "cylinder");
  PARTHENON_REQUIRE_THROWS(shape == "cylinder" || shape == "sphere",
                           "field_loop/shape must be \"cylinder\" or \"sphere\"");
  const bool sphere = (shape == "sphere");

  Real nx = 0.0, ny = 0.0, nz = 1.0;
  Real wrap_period = 0.0;
  if (!sphere) {
    Real axis_x = pin->GetOrAddReal("field_loop", "axis_x", 0.0);
    Real axis_y = pin->GetOrAddReal("field_loop", "axis_y", 0.0);
    Real axis_z = pin->GetOrAddReal("field_loop", "axis_z", 1.0);
    const Real axis_norm = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
    PARTHENON_REQUIRE_THROWS(axis_norm > 0.0,
                             "field_loop: (axis_x, axis_y, axis_z) must be nonzero");
    nx = axis_x / axis_norm;
    ny = axis_y / axis_norm;
    nz = axis_z / axis_norm;

    wrap_period = pin->GetOrAddReal("field_loop", "wrap_period", 0.0);
    if (wrap_period > 0.0) {
      // Periodic tiling of a tilted loop (see FieldLoopPotential) is only implemented
      // for an axis confined to the x1-x3 plane.
      PARTHENON_REQUIRE_THROWS(
          std::abs(ny) < 1.0e-13,
          "field_loop: wrap_period > 0 requires axis_y == 0 (axis confined to the "
          "x1-x3 plane)");
    }
  }

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  const Real dx = coords.Dxc<1>();
  const Real dy = coords.Dxc<2>();
  const Real dz = two_d ? 1.0 : coords.Dxc<3>(); // unused (aydz/axdz=0) when two_d

  pmb->par_for(
      "Phoebus::ProblemGenerator::FieldLoop", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = two_d ? 0.0 : coords.Xc<3>(k);

        const Real rho = (x * x + y * y + z * z < rad * rad) ? drat : 1.0;
        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P0;

        Real eos_lambda[2] = {0.5, 0.0};
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }
        v(ieng, k, j, i) =
            phoebus::energy_from_rho_P(eos, rho, P0, emin, emax, eos_lambda[0]);
        v(itmp, k, j, i) = eos.TemperatureFromDensityInternalEnergy(
            rho, v(ieng, k, j, i) / rho, eos_lambda);
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), eos_lambda) /
                           v(iprs, k, j, i);

        // Raw 3-velocity -> phoebus's "u^i = W v^i" primitive velocity. Flat Minkowski
        // Cartesian (required geometry above), so vsq is just the flat dot product.
        const Real vsq = vx * vx + vy * vy + vz * vz;
        const Real W = 1.0 / sqrt(1.0 - vsq);
        v(ivlo, k, j, i) = W * vx;
        v(ivlo + 1, k, j, i) = W * vy;
        v(ivlo + 2, k, j, i) = W * vz;
      });

  if (mhd) {
    // Initialize the CT face field as the discrete curl of A.
    using TE = parthenon::TopologicalElement;
    auto geom = Geometry::GetCoordinateSystem(rc.get());
    auto Bf = rc->PackVariables({fluid_cons::fbfield::name()});
    pmb->par_for(
        "Phoebus::ProblemGenerator::FieldLoop::fbfield::F1", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real xf = coords.Xf<1>(i);
          const Real zc = two_d ? 0.0 : coords.Xc<3>(k);
          // Bx = dAz/dy - dAy/dz
          const Real dAzdy = (FieldLoopAz(sphere, nx, ny, nz, wrap_period, rad, amp, xf,
                                          coords.Xf<2>(j + 1), zc) -
                              FieldLoopAz(sphere, nx, ny, nz, wrap_period, rad, amp, xf,
                                          coords.Xf<2>(j), zc)) /
                             dy;
          Real dAydz = 0.0;
          if (!two_d) {
            dAydz = (FieldLoopAy(sphere, nx, ny, nz, wrap_period, rad, amp, xf,
                                 coords.Xc<2>(j), coords.Xf<3>(k + 1)) -
                     FieldLoopAy(sphere, nx, ny, nz, wrap_period, rad, amp, xf,
                                 coords.Xc<2>(j), coords.Xf<3>(k))) /
                    dz;
          }
          Bf(TE::F1, 0, k, j, i) =
              (dAzdy - dAydz) * geom.DetGamma(CellLocation::Face1, k, j, i);
        });
    pmb->par_for(
        "Phoebus::ProblemGenerator::FieldLoop::fbfield::F2", kb.s, kb.e, jb.s, jb.e + 1,
        ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real yf = coords.Xf<2>(j);
          const Real zc = two_d ? 0.0 : coords.Xc<3>(k);
          // By = dAx/dz - dAz/dx
          Real dAxdz = 0.0;
          if (!two_d) {
            dAxdz = (FieldLoopAx(sphere, nx, ny, nz, wrap_period, rad, amp,
                                 coords.Xc<1>(i), yf, coords.Xf<3>(k + 1)) -
                     FieldLoopAx(sphere, nx, ny, nz, wrap_period, rad, amp,
                                 coords.Xc<1>(i), yf, coords.Xf<3>(k))) /
                    dz;
          }
          const Real dAzdx = (FieldLoopAz(sphere, nx, ny, nz, wrap_period, rad, amp,
                                          coords.Xf<1>(i + 1), yf, zc) -
                              FieldLoopAz(sphere, nx, ny, nz, wrap_period, rad, amp,
                                          coords.Xf<1>(i), yf, zc)) /
                             dx;
          Bf(TE::F2, 0, k, j, i) =
              (dAxdz - dAzdx) * geom.DetGamma(CellLocation::Face2, k, j, i);
        });
    if (!two_d) {
      pmb->par_for(
          "Phoebus::ProblemGenerator::FieldLoop::fbfield::F3", kb.s, kb.e + 1, jb.s, jb.e,
          ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real zf = coords.Xf<3>(k);
            // Bz = dAy/dx - dAx/dy
            const Real dAydx = (FieldLoopAy(sphere, nx, ny, nz, wrap_period, rad, amp,
                                            coords.Xf<1>(i + 1), coords.Xc<2>(j), zf) -
                                FieldLoopAy(sphere, nx, ny, nz, wrap_period, rad, amp,
                                            coords.Xf<1>(i), coords.Xc<2>(j), zf)) /
                               dx;
            const Real dAxdy = (FieldLoopAx(sphere, nx, ny, nz, wrap_period, rad, amp,
                                            coords.Xc<1>(i), coords.Xf<2>(j + 1), zf) -
                                FieldLoopAx(sphere, nx, ny, nz, wrap_period, rad, amp,
                                            coords.Xc<1>(i), coords.Xf<2>(j), zf)) /
                               dy;
            Bf(TE::F3, 0, k, j, i) =
                (dAydx - dAxdy) * geom.DetGamma(CellLocation::Face3, k, j, i);
          });
    }
  }

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace field_loop
