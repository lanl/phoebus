# © 2021-2024. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S.  Department of Energy/National
# Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is
# granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce,
# prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so.

add_library(radiation_unit_tests OBJECT
  test_closure.cpp
)

target_link_libraries(radiation_unit_tests
  PRIVATE
  phoebuslib
  parthenon
  Catch2::Catch2
  Kokkos::kokkos
  singularity-eos::singularity-eos
  )

target_link_libraries(radiation_unit_tests PRIVATE Catch2::Catch2)
target_sources(phoebus_unit_tests PRIVATE $<TARGET_OBJECTS:radiation_unit_tests>)
