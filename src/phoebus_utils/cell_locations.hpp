#ifndef PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
#define PHOEBUS_UTILS_CELL_LOCATIONS_HPP_

#include <kokkos_abstraction.hpp>
#include <defs.hpp>

enum class CellLocation{
  Cent, Face1, Face2, Face3, Corn
};

KOKKOS_FORCEINLINE_FUNCTION
CellLocation DirectionToFaceID(int dir) {
  switch(dir) {
    case parthenon::X1DIR:
      return CellLocation::Face1;
    case parthenon::X2DIR:
      return CellLocation::Face2;
    case parthenon::X3DIR:
      return CellLocation::Face3;
  }
  //TODO(JCD): insert the appropriate parthenon fail thing that's safe on device
  return CellLocation::Cent; // return the right type for the compiler
}

#endif // PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
