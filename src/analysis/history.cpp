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

#include "history.hpp"
#include "analysis/analysis.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "history_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include <interface/sparse_pack.hpp>
#include <parthenon/package.hpp>

namespace History {

Real ReduceMassAccretionRate(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Mesh *pmesh = md->GetMeshPointer();
  auto &pars = pmesh->packages.Get("geometry")->AllParams();
  const Real xh = pars.Get<Real>("xh");

  namespace p = fluid_prim;
  const std::vector<std::string> vars({p::density::name(), p::velocity::name()});

  PackIndexMap imap;
  auto pack = md->PackVariables(vars, imap);

  const int prho = imap[p::density::name()].first;
  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;

  auto geom = Geometry::GetCoordinateSystem(md);

  Real result = 0.0;
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for Mass Accretion Rate",
      DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = pack.GetCoords(b);
        if (coords.Xf<1>(i) <= xh && xh < coords.Xf<1>(i + 1)) {
          const Real dx1 = coords.CellWidthFA(X1DIR, k, j, i);
          const Real dx2 = coords.CellWidthFA(X2DIR, k, j, i);
          const Real dx3 = coords.CellWidthFA(X3DIR, k, j, i);

          // interp to make sure we're getting the horizon correct
          auto m = (CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i + 1) -
                    CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i - 1)) /
                   (2.0 * dx1);
          auto flux = (CalcMassFlux(pack, geom, prho, pvel_lo, pvel_hi, b, k, j, i) +
                       (xh - coords.Xc<1>(i)) * m) *
                      dx2 * dx3;

          lresult += flux;
        } else {
          lresult += 0.0;
        }
      },
      result);
  return result;
} // mass accretion

Real ReduceJetEnergyFlux(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Mesh *pmesh = md->GetMeshPointer();
  auto &pars = pmesh->packages.Get("geometry")->AllParams();
  const Real xh = pars.Get<Real>("xh");

  namespace p = fluid_prim;
  const std::vector<std::string> vars(
      {p::density::name(), p::bfield::name(), p::velocity::name()});

  PackIndexMap imap;
  auto pack = md->PackVariables(vars, imap);

  const int prho = imap[p::density::name()].first;
  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int pb_lo = imap[p::bfield::name()].first;
  const int pb_hi = imap[p::bfield::name()].second;

  auto geom = Geometry::GetCoordinateSystem(md);

  const Real sigma_cutoff = pmesh->packages.Get("fluid")->Param<Real>("sigma_cutoff");

  Real result = 0.0;
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for Jet Energy Flux",
      DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = pack.GetCoords(b);
        const Real sigma = CalcMagnetization(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi,
                                             prho, b, k, j, i);
        if (coords.Xf<1>(i) <= xh && xh < coords.Xf<1>(i + 1) && sigma > sigma_cutoff) {
          const Real dx1 = coords.CellWidthFA(X1DIR, k, j, i);
          const Real dx2 = coords.CellWidthFA(X2DIR, k, j, i);
          const Real dx3 = coords.CellWidthFA(X3DIR, k, j, i);

          // interp to make sure we're getting the horizon correct
          auto m = (CalcEMEnergyFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b, k, j,
                                     i + 1) -
                    CalcEMEnergyFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b, k, j,
                                     i - 1)) /
                   (2.0 * dx1);
          auto flux =
              (CalcEMEnergyFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b, k, j, i) +
               (xh - coords.Xc<1>(i)) * m) *
              dx2 * dx3;

          lresult += flux;
        } else {
          lresult += 0.0;
        }
      },
      result);
  return result;

} // JetEnergyFlux

Real ReduceJetMomentumFlux(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Mesh *pmesh = md->GetMeshPointer();
  auto &pars = pmesh->packages.Get("geometry")->AllParams();
  const Real xh = pars.Get<Real>("xh");

  namespace p = fluid_prim;
  const std::vector<std::string> vars(
      {p::density::name(), p::bfield::name(), p::velocity::name()});

  PackIndexMap imap;
  auto pack = md->PackVariables(vars, imap);

  const int prho = imap[p::density::name()].first;
  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int pb_lo = imap[p::bfield::name()].first;
  const int pb_hi = imap[p::bfield::name()].second;

  auto geom = Geometry::GetCoordinateSystem(md);

  const Real sigma_cutoff = pmesh->packages.Get("fluid")->Param<Real>("sigma_cutoff");

  Real result = 0.0;
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for Jet Momentum Flux",
      DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = pack.GetCoords(b);
        const Real sigma = CalcMagnetization(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi,
                                             prho, b, k, j, i);
        if (coords.Xf<1>(i) <= xh && xh < coords.Xf<1>(i + 1) && sigma > sigma_cutoff) {
          const Real dx1 = coords.CellWidthFA(X1DIR, k, j, i);
          const Real dx2 = coords.CellWidthFA(X2DIR, k, j, i);
          const Real dx3 = coords.CellWidthFA(X3DIR, k, j, i);

          // interp to make sure we're getting the horizon correct
          auto m = (CalcEMMomentumFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b, k,
                                       j, i + 1) -
                    CalcEMMomentumFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b, k,
                                       j, i - 1)) /
                   (2.0 * dx1);
          auto flux = (CalcEMMomentumFlux(pack, geom, pvel_lo, pvel_hi, pb_lo, pb_hi, b,
                                          k, j, i) +
                       (xh - coords.Xc<1>(i)) * m) *
                      dx2 * dx3;

          lresult += flux;
        } else {
          lresult += 0.0;
        }
      },
      result);
  return result;

} // ReduceJetMomentumFlux

Real ReduceMagneticFluxPhi(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  Mesh *pmesh = md->GetMeshPointer();
  auto &pars = pmesh->packages.Get("geometry")->AllParams();
  const Real xh = pars.Get<Real>("xh");

  namespace c = fluid_cons;
  const std::vector<std::string> vars({c::bfield::name()});

  PackIndexMap imap;
  auto pack = md->PackVariables(vars, imap);

  const int cb_lo = imap[c::bfield::name()].first;

  auto geom = Geometry::GetCoordinateSystem(md);

  Real result = 0.0;
  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Phoebus History for Jet Magnetic Flux Phi",
      DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = pack.GetCoords(b);
        if (coords.Xf<1>(i) <= xh && xh < coords.Xf<1>(i + 1)) {
          const Real dx1 = coords.CellWidthFA(X1DIR, k, j, i + 1);
          const Real dx2 = coords.CellWidthFA(X2DIR, k, j, i + 1);
          const Real dx3 = coords.CellWidthFA(X3DIR, k, j, i + 1);

          // interp to make sure we're getting the horizon correct
          auto m = (CalcMagneticFluxPhi(pack, geom, cb_lo, b, k, j, i + 1 + 1) -
                    CalcMagneticFluxPhi(pack, geom, cb_lo, b, k, j, i - 1 + 1)) /
                   (2.0 * dx1);
          auto flux = (CalcMagneticFluxPhi(pack, geom, cb_lo, b, k, j, i + 1) +
                       (xh - coords.Xc<1>(i + 1)) * m) *
                      dx2 * dx3;

          lresult += flux;
        } else {
          lresult += 0.0;
        }
      },
      result);
  return 0.5 * result; // 0.5 \int detg B^r dx2 dx3
} // Phi

// SN analysis
// ReduceLocalizationFunction is not used currently. However this function returns
// normalization function that is used for localizing quantities at the center, or at some
// particular case. For example SN diagnostics oftec computes quantities at 400 km.
void ReduceLocalizationFunction(MeshData<Real> *md) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  namespace diag = diagnostic_variables;
  using parthenon::MakePackDescriptor;
  auto *pmb = md->GetParentPointer();
  Mesh *pmesh = md->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  const int ndim = pmesh->ndim;

  static auto desc = MakePackDescriptor<diag::localization_function>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  const int nblocks = v.GetNBlocks();
  auto geom = Geometry::GetCoordinateSystem(md);

  parthenon::par_for(
      parthenon::LoopPatternMDRange(), "Central Density for SN", DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &coords = v.GetCoordinates(b);
        auto analysis = pmb->packages.Get("analysis").get();
        const Real x[3] = {coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                           coords.Xc<3>(k, j, i)};
        const Real sigma = analysis->Param<Real>("sigma");
        Real gam[3][3];
        Real r2 = 0;
        geom.Metric(CellLocation::Cent, 0, k, j, i, gam);
        for (int n = 0; n < 3; ++n) {
          for (int m = 0; m < 3; ++m) {
            r2 += gam[n][m] * x[n] * x[m];
          }
        }
        v(b, diag::localization_function(), k, j, i) = std::exp(-r2 / sigma / sigma);
      });

} // exp

// This function calculates mass accretion rate which I defined as
// Mdot=Int(dV*d/dx^i{detg*rho*U^i}) where detg is the determinant of four metric, U is
// four-velocity, and dV=d^3x

Real CalculateMdot(MeshData<Real> *md, Real rc, bool gain) {
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  namespace c = fluid_cons;
  using parthenon::MakePackDescriptor;
  auto *pmb = md->GetParentPointer();
  Mesh *pmesh = md->GetMeshPointer();
  auto &resolved_pkgs = pmesh->resolved_packages;
  const int ndim = pmesh->ndim;
  using parthenon::PDOpt;
  static auto desc = MakePackDescriptor<c::density, internal_variables::GcovHeat,
                                        internal_variables::GcovCool>(
      resolved_pkgs.get(), {}, {PDOpt::WithFluxes});
  auto v = desc.GetPack(md);

  const int nblocks = v.GetNBlocks();
  auto geom = Geometry::GetCoordinateSystem(md);
  Real result = 0.0;
  auto rad = pmb->packages.Get("radiation").get();
  const parthenon::AllReduce<bool> *pdo_gain_reducer =
      rad->MutableParam<parthenon::AllReduce<bool>>("do_gain_reducer");
  const bool do_gain = pdo_gain_reducer->val;
  auto analysis = pmb->packages.Get("analysis").get();
  const Real inside_pns_threshold = analysis->Param<Real>("inside_pns_threshold");

  parthenon::par_reduce(
      parthenon::LoopPatternMDRange(), "Calculates mass accretion rate (SN)",
      DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = v.GetCoordinates(b);
        const Real vol = coords.CellVolume(k, j, i);

        Real C[NDFULL];
        geom.Coords(CellLocation::Cent, b, k, j, i, C);
        Real r = std::sqrt(C[1] * C[1] + C[2] * C[2] + C[3] * C[3]);
        if (r <= rc) {
          if (gain) {
            bool is_netheat = (v(b, internal_variables::GcovHeat(), k, j, i) -
                                   v(b, internal_variables::GcovCool(), k, j, i) >
                               1.e-8); // checks that in the gain region
            bool is_inside_pns = (r < inside_pns_threshold); // checks that inside PNS
            if (do_gain && (is_inside_pns || is_netheat)) {

              lresult += -ComputeDivInPillbox(
                  ndim, b, k, j, i, coords, [=](int b, int dir, int k, int j, int i) {
                    return v.flux(b, dir, c::density(), k, j, i);
                  });
            } else {
              lresult += 0;
            }

          } else {
            lresult += -ComputeDivInPillbox(
                ndim, b, k, j, i, coords, [=](int b, int dir, int k, int j, int i) {
                  return v.flux(b, dir, c::density(), k, j, i);
                });
          }

        } else {
          lresult += 0.0;
        }
      },
      Kokkos::Sum<Real>(result));
  return result;
} // mdot

} // namespace History
