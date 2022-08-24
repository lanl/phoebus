# © 2021. Triad National Security, LLC. All rights reserved.  This
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

add_library(geometry_unit_tests OBJECT
  test_geometry.cpp
  test_tetrads.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/analytic_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/boyer_lindquist.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/boyer_lindquist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/cached_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/flrw.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/flrw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/fmks.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/fmks.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/geometry_defaults.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/geometry_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/mckinney_gammie_ryan.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/mckinney_gammie_ryan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/modified_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/minkowski.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/minkowski.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/snake.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/snake.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_kerr_schild.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_kerr_schild.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_minkowski.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_minkowski.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/phoebus_utils/phoebus_interpolation.hpp

)

target_link_libraries(geometry_unit_tests
  PRIVATE
  parthenon
  singularity-eos::singularity-eos
  Catch2::Catch2
  Kokkos::kokkos)

target_include_directories(geometry_unit_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  )


target_link_libraries(geometry_unit_tests PRIVATE Catch2::Catch2)
target_sources(phoebus_unit_tests PRIVATE $<TARGET_OBJECTS:geometry_unit_tests>)
