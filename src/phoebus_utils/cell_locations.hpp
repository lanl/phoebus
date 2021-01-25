#ifndef PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
#define PHOEBUS_UTILS_CELL_LOCATIONS_HPP_

enum class CellLocation{
  Face1, Face2, Face3, Cent, Corn
};

KOKKOS_FORCEINLINE_FUNCITON
CellLocation DirectionToFaceID(int dir) {
  switch(dir) {
    case X1DIR:
      return CellLocation::Face1;
    case X2DIR:
      return CellLocation::Face2;
    case X3DIR:
      return CellLocation::Face3;
  }
  //TODO(JCD): insert the appropriate parthenon fail thing that's safe on device
  return CellLocation::Cent; // return the right type for the compiler
}

#endif // PHOEBUS_UTILS_CELL_LOCATIONS_HPP_
