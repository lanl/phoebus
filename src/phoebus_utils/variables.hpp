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

namespace fluid_prim {
constexpr char density[] = "p.density";
constexpr char velocity[] = "p.velocity";
constexpr char energy[] = "p.energy";
constexpr char bfield[] = "p.bfield";
constexpr char ye[] = "p.ye";
constexpr char pressure[] = "pressure";
constexpr char temperature[] = "temperature";
constexpr char gamma1[] = "gamma1";
} // namespace fluid_prim

namespace fluid_cons {
constexpr char density[] = "c.density";
constexpr char momentum[] = "c.momentum";
constexpr char energy[] = "c.energy";
constexpr char bfield[] = "c.bfield";
constexpr char ye[] = "c.ye";
} // namespace fluid_cons

namespace radmoment_prim {
constexpr char J[] = "r.p.J";
constexpr char H[] = "r.p.H";
} // namespace radmoment_prim

namespace radmoment_cons {
constexpr char E[] = "r.c.E";
constexpr char F[] = "r.c.F";
} // namespace radmoment_cons

namespace radmoment_internal {
constexpr char xi[] = "r.i.xi";
constexpr char phi[] = "r.i.phi";
constexpr char ql[] = "r.i.ql";
constexpr char qr[] = "r.i.qr";
constexpr char ql_v[] = "r.i.ql_v";
constexpr char qr_v[] = "r.i.qr_v";
constexpr char dJ[] = "r.i.dJ";
constexpr char kappaJ[] = "r.i.kappaJ";
constexpr char kappaH[] = "r.i.kappaH";
constexpr char JBB[] = "r.i.JBB";
constexpr char tilPi[] = "r.i.tilPi";
constexpr char kappaH_mean[] = "r.i.kappaH_mean";
constexpr char c2pfail[] = "r.i.c2p_fail";
constexpr char srcfail[] = "r.i.src_fail";
} // namespace radmoment_internal

namespace mocmc_internal {
constexpr char dnsamp[] = "mocmc.i.dnsamp";
constexpr char Inu0[] = "mocmc.i.inu0";
constexpr char Inu1[] = "mocmc.i.inu1";
constexpr char jinvs[] = "mocmc.i.jinvs";
} // namespace mocmc_internal

namespace internal_variables {
constexpr char face_signal_speed[] = "face_signal_speed";
constexpr char cell_signal_speed[] = "cell_signal_speed";
constexpr char emf[] = "emf";
constexpr char c2p_scratch[] = "c2p_scratch";
constexpr char ql[] = "ql";
constexpr char qr[] = "qr";
constexpr char fail[] = "fail";
constexpr char Gcov[] = "Gcov";
constexpr char Gye[] = "Gye";
constexpr char c2p_mu[] = "c2p_mu";
constexpr char tau[] = "light_bulb_tau";
constexpr char GcovHeat[] = "GcovHeat";
constexpr char GcovCool[] = "GcovCool";
constexpr char compweight[] = "compweight";

} // namespace internal_variables

namespace geometric_variables {
constexpr char cell_coords[] = "g.c.coord";
constexpr char node_coords[] = "g.n.coord";
} // namespace geometric_variables

namespace diagnostic_variables {
constexpr char divb[] = "divb";
constexpr char divf[] = "flux_divergence";
constexpr char src_terms[] = "src_terms";
constexpr char r_divf[] = "r.flux_divergence";
constexpr char r_src_terms[] = "r.src_terms";
} // namespace diagnostic_variables

#endif // PHOEBUS_UTILS_VARIABLES_HPP_
