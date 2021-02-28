#ifndef PHOEBUS_UTILS_VARIABLES_HPP_
#define PHOEBUS_UTILS_VARIABLES_HPP_

namespace primitive_variables {
  constexpr char density[] = "p.density";
  constexpr char velocity[] = "p.velocity";
  constexpr char energy[] = "p.energy";
  constexpr char bfield[] = "p.bfield";
  constexpr char ye[] = "p.ye";
  constexpr char pressure[] = "pressure";
  constexpr char temperature[] = "temperature";
  constexpr char gamma1[] = "gamma1";
}

namespace conserved_variables {
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
}

namespace diagnostic_variables {
  constexpr char divb[] = "divb";
}

#endif // PHOEBUS_UTILS_VARIABLES_HPP_