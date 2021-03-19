add_library(geometry_unit_tests OBJECT
  test_geometry.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/analytic_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/boyer_lindquist.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/boyer_lindquist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/cached_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/fmks.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/fmks.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/geometry_defaults.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/geometry_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/mckinney_gammie_ryan.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/mckinney_gammie_ryan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/modified_system.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/minkowski.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/minkowski.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_kerr_schild.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_kerr_schild.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_minkowski.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/spherical_minkowski.cpp

)

target_link_libraries(geometry_unit_tests
  PRIVATE
    parthenon
    Catch2::Catch2
    Kokkos::kokkos)

target_include_directories(geometry_unit_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  )


target_link_libraries(geometry_unit_tests PRIVATE Catch2::Catch2)
target_sources(phoebus_unit_tests PRIVATE $<TARGET_OBJECTS:geometry_unit_tests>)
