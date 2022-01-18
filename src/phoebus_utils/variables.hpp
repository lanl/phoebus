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
}

namespace fluid_cons {
  constexpr char density[] = "c.density";
  constexpr char momentum[] = "c.momentum";
  constexpr char energy[] = "c.energy";
  constexpr char bfield[] = "c.bfield";
  constexpr char ye[] = "c.ye";
}

namespace radmoment_prim {
  constexpr char J[] = "r.p.J";
  constexpr char H[] = "r.p.H";
}

namespace radmoment_cons {
  constexpr char E[] = "r.c.E";
  constexpr char F[] = "r.c.F";
}

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
}

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
}

namespace geometric_variables {
  constexpr char cell_coords[] = "g.c.coord";
  constexpr char node_coords[] = "g.n.coord";
}

namespace diagnostic_variables {
  constexpr char divb[] = "divb";
  constexpr char divf[] = "flux_divergence";
  constexpr char src_terms[] = "src_terms";
}

#endif // PHOEBUS_UTILS_VARIABLES_HPP_
