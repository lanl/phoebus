add_library(geometry_unit_tests OBJECT test_geometry.cpp test_tetrads.cpp)

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
