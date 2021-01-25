include(Catch)

add_executable(phoebus_unit_tests phoebus_unit_test_main.cpp)
target_link_libraries(phoebus_unit_tests
  PRIVATE
    parthenon
    Catch2::Catch2
    Kokkos::kokkos)

catch_discover_tests(phoebus_unit_tests TEST_SPEC "[geometry]" TEST_PREFIX "geometry:")

if(PHOEBUS_ENABLE_UNIT_TESTS)
  message("Building unit tests.")
  add_subdirectory(unit)
endif()
if(PHOEBUS_ENABLE_INTEGRATION_TESTS)
  message("Building integration tests.")
  add_subdirectory(integration)
endif()
if(PHOEBUS_ENABLE_REGRESSION_TESTS)
  message("Building regression tests.")
  add_subdirectory(regression)
endif()
