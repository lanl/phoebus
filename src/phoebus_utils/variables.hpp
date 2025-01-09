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

#ifndef PHOEBUS_UTILS_VARIABLES_HPP_
#define PHOEBUS_UTILS_VARIABLES_HPP_

#include <interface/sparse_pack.hpp>
#include <parthenon/package.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

#define VARIABLE_NONS(varname)                                                           \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #varname; }                                       \
  }

#define VARIABLE_CUSTOM(varname, varstring)                                              \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #varstring; }                                     \
  }

namespace fluid_prim {
VARIABLE(p, density);
VARIABLE(p, velocity);
VARIABLE(p, energy);
VARIABLE(p, bfield);
VARIABLE(p, ye);
VARIABLE_NONS(pressure);
VARIABLE_NONS(temperature);
VARIABLE_NONS(entropy);
VARIABLE_NONS(cs);
VARIABLE_NONS(gamma1);
} // namespace fluid_prim

namespace fluid_cons {
VARIABLE(c, density);
VARIABLE(c, momentum);
VARIABLE(c, energy);
VARIABLE(c, bfield);
VARIABLE(c, ye);
} // namespace fluid_cons

namespace radmoment_prim {
VARIABLE(r.p, J);
VARIABLE(r.p, H);
} // namespace radmoment_prim

namespace radmoment_cons {
VARIABLE(r.c, E);
VARIABLE(r.c, F);
} // namespace radmoment_cons

namespace radmoment_internal {
VARIABLE(r.i, xi);
VARIABLE(r.i, phi);
VARIABLE(r.i, ql);
VARIABLE(r.i, qr);
VARIABLE(r.i, ql_v);
VARIABLE(r.i, qr_v);
VARIABLE(r.i, dJ);
VARIABLE(r.i, kappaJ);
VARIABLE(r.i, kappaH);
VARIABLE(r.i, JBB);
VARIABLE(r.i, tilPi);
VARIABLE(r.i, kappaH_mean);
VARIABLE(r.i, c2pfail);
VARIABLE(r.i, srcfail);
} // namespace radmoment_internal

namespace mocmc_internal {
VARIABLE(mocmc.i, dnsamp);
VARIABLE(mocmc.i, Inu0);
VARIABLE(mocmc.i, Inu1);
VARIABLE(mocmc.i, jinvs);
} // namespace mocmc_internal

namespace internal_variables {
VARIABLE_NONS(face_signal_speed);
VARIABLE_NONS(cell_signal_speed);
VARIABLE_NONS(emf);
VARIABLE_NONS(c2p_scratch);
VARIABLE_NONS(ql);
VARIABLE_NONS(qr);
VARIABLE_NONS(fail);
VARIABLE_NONS(Gcov);
VARIABLE_NONS(Gye);
VARIABLE_NONS(c2p_mu);
VARIABLE_NONS(tau);
;
VARIABLE_NONS(GcovHeat);
VARIABLE_NONS(GcovCool);
} // namespace internal_variables

namespace geometric_variables {
VARIABLE_CUSTOM(cell_coords, g.c.coord);
VARIABLE_CUSTOM(node_coords, g.n.coord);
} // namespace geometric_variables

namespace tracer_variables {
SWARM_VARIABLE(Real, tr, rho);
SWARM_VARIABLE(Real, tr, temperature);
SWARM_VARIABLE(Real, tr, ye);
SWARM_VARIABLE(Real, tr, entropy);
SWARM_VARIABLE(Real, tr, pressure);
SWARM_VARIABLE(Real, tr, energy);
SWARM_VARIABLE(Real, tr, vel_x);
SWARM_VARIABLE(Real, tr, vel_y);
SWARM_VARIABLE(Real, tr, vel_z);
SWARM_VARIABLE(Real, tr, lorentz);
SWARM_VARIABLE(Real, tr, lapse);
SWARM_VARIABLE(Real, tr, detgamma);
SWARM_VARIABLE(Real, tr, shift_x);
SWARM_VARIABLE(Real, tr, shift_y);
SWARM_VARIABLE(Real, tr, shift_z);
SWARM_VARIABLE(Real, tr, mass);
SWARM_VARIABLE(Real, tr, bernoulli);
SWARM_VARIABLE(Real, tr, B_x);
SWARM_VARIABLE(Real, tr, B_y);
SWARM_VARIABLE(Real, tr, B_z);
} // namespace tracer_variables

namespace diagnostic_variables {
VARIABLE_NONS(divb);
VARIABLE_NONS(ratio_divv_cs);
VARIABLE_NONS(entropy_z_0);
VARIABLE_NONS(localization_function);
VARIABLE_CUSTOM(divf, flux_divergence);
VARIABLE_NONS(src_terms);
VARIABLE_CUSTOM(r_divf, r.flux_divergence);
VARIABLE_CUSTOM(r_src_terms, r.src_terms);
} // namespace diagnostic_variables

namespace phoebus {
template <typename Data, typename... Ts>
auto MakePackDescriptor(Data *rc) {
  parthenon::Mesh *pm = rc->GetMeshPointer();
  parthenon::StateDescriptor *resolved_pkgs = pm->resolved_packages.get();
  return parthenon::MakePackDescriptor<Ts...>(resolved_pkgs);
}
} // namespace phoebus

#endif // PHOEBUS_UTILS_VARIABLES_HPP_
