// © 2022-2023. Triad National Security, LLC. All rights reserved.
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

#include <cmath>

#include "fixup.hpp"

#include <defs.hpp>

#include "fluid/con2prim_robust.hpp"
#include "fluid/prim2con.hpp"
#include "geometry/geometry.hpp"
#include "geometry/tetrads.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/programming_utils.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"
#include "radiation/closure.hpp"
#include "radiation/closure_m1.hpp"
#include "radiation/closure_mocmc.hpp"
#include "radiation/radiation.hpp"
#include <parthenon/package.hpp>

using radiation::ClosureEquation;
using radiation::ClosureSettings;
using radiation::ClosureVerbosity;
using radiation::Tens2;
using radiation::Vec;
using robust::ratio;
using parthenon::MakePackDescriptor;

namespace fixup {

template <typename T, class CLOSURE>
TaskStatus ConservedToPrimitiveFixupImpl(T *rc) {
  namespace p = fluid_prim;
  namespace c = fluid_cons;
  namespace impl = internal_variables;
  namespace ir = radmoment_internal;
  namespace pr = radmoment_prim;
  namespace cr = radmoment_cons;
  Mesh *pmesh = rc->GetMeshPointer();
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);

  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  StateDescriptor *fluid_pkg = pmesh->packages.Get("fluid").get();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();
  StateDescriptor *eos_pkg = pmesh->packages.Get("eos").get();

  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc =
      MakePackDescriptor<c::density, p::density, p::velocity, 
                         c::momentum, p::energy, c::energy, 
                         p::bfield, p::ye, c::ye, p::pressure, 
                         p::temperature, p::gamma1, impl::cell_signal_speed, 
                         impl::fail, ir::c2pfail, pr::J, pr::H, 
                         cr::E, cr::F, ir::xi, ir::phi, ir::tilPi>(
          resolved_pkgs.get());

  auto v = desc.GetPack(rc);

  const bool rad_active = rad_pkg->Param<bool>("active");
  const int num_species = rad_active ? rad_pkg->Param<int>("num_species") : 0;

  bool enable_c2p_fixup = fix_pkg->Param<bool>("enable_c2p_fixup");
  bool update_fluid = fluid_pkg->Param<bool>("active");
  if (!enable_c2p_fixup || !update_fluid) return TaskStatus::complete;

  bool report_c2p_fails = fix_pkg->Param<bool>("report_c2p_fails");
  if (report_c2p_fails) {
    int nfail_total;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b,  impl::fail(), k, j, i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total nfail: %i\n", nfail_total);
    IndexRange ibi = rc->GetBoundsI(IndexDomain::interior);
    IndexRange jbi = rc->GetBoundsJ(IndexDomain::interior);
    IndexRange kbi = rc->GetBoundsK(IndexDomain::interior);
    nfail_total = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Rad ConToPrim::Solve fixup failures",
        DevExecSpace(), 0, v.GetNBlocks() - 1, kbi.s, kbi.e, jbi.s, jbi.e, ibi.s, ibi.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &nf) {
          if (v(b,  impl::fail(), k, j, i) == con2prim_robust::FailFlags::fail) {
            nf++;
          }
        },
        Kokkos::Sum<int>(nfail_total));
    printf("total interior nfail: %i\n", nfail_total);
  }

  const int ndim = pmesh->ndim;

  auto eos = eos_pkg->Param<Microphysics::EOS::EOS>("d.EOS");
  auto geom = Geometry::GetCoordinateSystem(rc);
  Bounds *pbounds = fix_pkg->MutableParam<Bounds>("bounds");
  Bounds bounds = *pbounds;

  Coordinates_t coords = rc->GetParentPointer()->coords;

  auto fluid_c2p_failure_strategy =
      fix_pkg->Param<FAILURE_STRATEGY>("fluid_c2p_failure_strategy");
  const auto c2p_failure_force_fixup_both =
      fix_pkg->Param<bool>("c2p_failure_force_fixup_both");

  PARTHENON_REQUIRE(!c2p_failure_force_fixup_both,
                    "As currently implemented this is a race condition!");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConToPrim::Solve fixup", DevExecSpace(), 0, v.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real eos_lambda[2]; // use last temp as initial guess
        eos_lambda[0] = 0.5;
        eos_lambda[1] = std::log10(v(b, p::temperature(), k, j, i));

        Real gamma_max, e_max;
        bounds.GetCeilings(coords.Xc<1>(k, j, i), coords.Xc<2>(k, j, i),
                           coords.Xc<3>(k, j, i), gamma_max, e_max);

        if (c2p_failure_force_fixup_both && rad_active) {
          if (v(b,  impl::fail(), k, j, i) == con2prim_robust::FailFlags::fail ||
              v(b, ir::c2pfail(), k, j, i) == radiation::FailFlags::fail) {
            v(b,  impl::fail(), k, j, i) = con2prim_robust::FailFlags::fail;
            v(b, ir::c2pfail(), k, j, i) = radiation::FailFlags::fail;
          }
        }

        // Need to account for not stenciling outside of ghost zones
        // bool is_outer_ghost_layer =
        //    (i == ib.s || i == ib.e || j == jb.s || j == jb.e || k == kb.s || k ==
        //    kb.e);

        auto fail = [&](const int k, const int j, const int i) {
          if (c2p_failure_force_fixup_both) {
            return v(b,  impl::fail(), k, j, i) * v(b, ir::c2pfail(), k, j, i);
          } else {
            return v(b,  impl::fail(), k, j, i);
          }
        };
        auto fixup = [&](auto iv, const Real inv_mask_sum) {
          v(b, iv, k, j, i) = fail(k, j, i - 1) * v(b, iv, k, j, i - 1) +
                              fail(k, j, i + 1) * v(b, iv, k, j, i + 1);
          if (ndim > 1) {
            v(b, iv, k, j, i) += fail(k, j - 1, i) * v(b, iv, k, j - 1, i) +
                                 fail(k, j + 1, i) * v(b, iv, k, j + 1, i);
            if (ndim == 3) {
              v(b, iv, k, j, i) += fail(k - 1, j, i) * v(b, iv, k - 1, j, i) +
                                   fail(k + 1, j, i) * v(b, iv, k + 1, j, i);
            }
          }
          return inv_mask_sum * v(b, iv, k, j, i);
        };
        // When using IndexDomain::entire, can't stencil from e.g. i = 0 because 0 - 1 =
        // -1 is a segfault
        if (v(b,  impl::fail(), k, j, i) == con2prim_robust::FailFlags::fail) {
          Real num_valid = 0;
          num_valid = v(b,  impl::fail(), k, j, i - 1) + v(b,  impl::fail(), k, j, i + 1);
          if (ndim > 1) num_valid += v(b,  impl::fail(), k, j - 1, i) + v(b,  impl::fail(), k, j + 1, i);
          if (ndim == 3) num_valid += v(b,  impl::fail(), k - 1, j, i) + v(b,  impl::fail(), k + 1, j, i);

          // if (num_valid > 0.5 &&
          //    fluid_c2p_failure_strategy == FAILURE_STRATEGY::interpolate && i > ib.s &&
          //    i < ib.e - 1 && j > jb.s && j < jb.e - 1 && k > kb.s && k < kb.e - 1) {
          if (num_valid > 0.5 &&
              fluid_c2p_failure_strategy == FAILURE_STRATEGY::interpolate) {
            const Real norm = 1.0 / num_valid;
            v(b, p::density(), k, j, i) = fixup(p::density(), norm);
            for (int i = 0; i < 3; ++i) {
              v(b, p::velocity(i), k, j, i) = fixup(p::velocity(i), norm);
            }
            v(b, p::energy(), k, j, i) = fixup(p::energy(), norm);

            if (v.Contains(b, p::ye())) v(b, p::ye(), k, j, i) = fixup(p::ye(), norm);

            v(b, p::density(), k, j, i) =
                std::max<Real>(v(b, p::density(), k, j, i), 100. * robust::SMALL());
            v(b, p::energy(), k, j, i) =
                std::max<Real>(v(b, p::energy(), k, j, i), 100. * robust::SMALL());
          } else {
            // No valid neighbors; set fluid mass/energy to near-zero and set primitive
            // velocities to zero

            v(b, p::density(), k, j, i) = 100. * robust::SMALL();
            v(b, p::energy(), k, j, i) = 100. * robust::SMALL();

            // Safe value for ye
            if (v.Contains(b, p::ye())) {
              v(b, p::ye(), k, j, i) = 0.5;
            }

            // Zero primitive velocities
            SPACELOOP(ii) { v(b, p::velocity(ii), k, j, i) = 0.; }
          }

          const Real sdetgam = geom.DetGamma(CellLocation::Cent, k, j, i);
          const Real alpha = geom.Lapse(CellLocation::Cent, k, j, i);
          Real beta[3];
          geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
          Real gcov[4][4];
          geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
          Real gcon[3][3];
          geom.MetricInverse(CellLocation::Cent, k, j, i, gcon);

          // Clamp velocity now (for rad inversion)
          Real vpcon[3] = {v(b, p::velocity(0), k, j, i), v(b, p::velocity(1), k, j, i),
                           v(b, p::velocity(2), k, j, i)};
          Real W = phoebus::GetLorentzFactor(vpcon, gcov);
          if (W > gamma_max) {
            const Real rescale = std::sqrt(gamma_max * gamma_max - 1.) / (W * W - 1.);
            SPACELOOP(ii) { vpcon[ii] *= rescale; }
            SPACELOOP(ii) { v(b, p::velocity(ii), k, j, i) = vpcon[ii]; }
            W = gamma_max;
          }

          // Update dependent primitives
          if (v.Contains(b, p::ye())) eos_lambda[0] = v(b, p::ye(), k, j, i);
          v(b, p::temperature(), k, j, i) = eos.TemperatureFromDensityInternalEnergy(
              v(b, p::density(), k, j, i), ratio(v(b, p::energy(), k, j, i), v(b, p::density(), k, j, i)),
              eos_lambda);
          v(b, p::pressure(), k, j, i) = eos.PressureFromDensityTemperature(
              v(b, p::density(), k, j, i), v(b, p::temperature(), k, j, i), eos_lambda);
          v(b, p::gamma1(), k, j, i) =
              ratio(eos.BulkModulusFromDensityTemperature(v(b, p::density(), k, j, i),
                                                          v(b, p::temperature(), k, j, i), eos_lambda),
                    v(b, p::pressure(), k, j, i));

          // Update conserved variables

          Real S[3];
          Real bcons[3];
          Real bp[3] = {0.0, 0.0, 0.0};
          if (v.Contains(b, p::bfield(2))) {
            bp[0] = v(b, p::bfield(0), k, j, i);
            bp[1] = v(b, p::bfield(1), k, j, i);
            bp[2] = v(b, p::bfield(2), k, j, i);
          }
          Real ye_cons;
          Real ye_prim = 0.5;
          if (v.Contains(b, p::ye())) {
            ye_prim = v(b, p::ye(), k, j, i);
          }
          Real sig[3];
          prim2con::p2c(v(b, p::density(), k, j, i), vpcon, bp, v(b, p::energy(), k, j, i), ye_prim,
                        v(b, p::pressure(), k, j, i), v(b, p::gamma1(), k, j, i), gcov, gcon, beta, alpha,
                        sdetgam, v(b, c::density(), k, j, i), S, bcons, v(b, c::energy(), k, j, i),
                        ye_cons, sig);
          v(b, c::momentum(0), k, j, i) = S[0];
          v(b, c::momentum(1), k, j, i) = S[1];
          v(b, c::momentum(2), k, j, i) = S[2];
          if (v.Contains(b, p::ye())) v(b, c::ye(), k, j, i) = ye_cons;
          for (int m = 0; m < 3; ++m) {
            v(b, impl::cell_signal_speed(m), k, j, i) = sig[m];
          }

          if (v.Contains(b, ir::c2pfail())) {
            // If rad c2p failed, we'll fix that up subsequently
            if (v(b, ir::c2pfail(), k, j, i) == radiation::FailFlags::success) {
              for (int ispec = 0; ispec < num_species; ispec++) {
                typename CLOSURE::LocalGeometryType g(geom, CellLocation::Cent, b, k, j,
                                                      i);
                Vec con_v{vpcon[0] / W, vpcon[1] / W, vpcon[2] / W};
                CLOSURE c(con_v, &g);

                Real E = v(b, cr::E(ispec), k, j, i) / sdetgam;
                Vec cov_F;
                SPACELOOP(ii) { cov_F(ii) = v(b, cr::F(ispec, ii), k, j, i) / sdetgam; }
                Tens2 con_tilPi;
                Real J;
                Vec cov_H;
                if (v.Contains(b, ir::tilPi())) {
                  SPACELOOP2(ii, jj) {
                    con_tilPi(ii, jj) = v(b, ir::tilPi(ispec, ii, jj), k, j, i);
                  }
                } else {
                  Real xi = 0.;
                  Real phi = M_PI;
                  // TODO(BRR) STORE_GUESS
                  c.GetConTilPiFromCon(E, cov_F, xi, phi, &con_tilPi);
                }

                c.Con2Prim(E, cov_F, con_tilPi, &J, &cov_H);

                v(b, pr::J(ispec), k, j, i) = J;
                SPACELOOP(ii) { v(b, pr::H(ispec, ii), k, j, i) = cov_H(ii) / J; }

                // Floors and ceilings will be applied subsequently by ApplyFloors task
              }
            }
          }
        }
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus ConservedToPrimitiveFixup(T *rc) {
  Mesh *pmesh = rc->GetMeshPointer();
  StateDescriptor *rad_pkg = pmesh->packages.Get("radiation").get();
  StateDescriptor *fix_pkg = pmesh->packages.Get("fixup").get();
  const bool enable_rad_floors = fix_pkg->Param<bool>("enable_rad_floors");
  std::string method;
  if (enable_rad_floors) {
    method = rad_pkg->Param<std::string>("method");
  }

  // TODO(BRR) share these settings somewhere else. Set at configure time?
  using settings =
      ClosureSettings<ClosureEquation::energy_conserve, ClosureVerbosity::quiet>;
  if (method == "moment_m1") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureM1<settings>>(rc);
  } else if (method == "moment_eddington") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureEdd<settings>>(rc);
  } else if (method == "mocmc") {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureMOCMC<settings>>(rc);
  } else {
    return ConservedToPrimitiveFixupImpl<T, radiation::ClosureEdd<settings>>(rc);
  }
  return TaskStatus::complete;
}

template TaskStatus
ConservedToPrimitiveFixup<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
// template TaskStatus
template <>
TaskStatus ConservedToPrimitiveFixup<MeshData<Real>>(MeshData<Real> *md) {
  for (const auto &mbd : md->GetAllBlockData()) {
    ConservedToPrimitiveFixup(mbd.get());
  }
  return TaskStatus::complete;
}

} // namespace fixup
