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
}

#endif // PHOEBUS_UTILS_VARIABLES_HPP_
