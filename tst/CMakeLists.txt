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

include(Catch)

add_executable(phoebus_unit_tests phoebus_unit_test_main.cpp)
target_link_libraries(phoebus_unit_tests
  PRIVATE
    parthenon
    Catch2::Catch2
    Kokkos::kokkos
    singularity-eos::singularity-eos
    )
if (PHOEBUS_ENABLE_MPI)
  target_compile_definitions(phoebus_unit_tests
    PRIVATE
    MPI_PARALLEL)
endif()

if(PHOEBUS_ENABLE_UNIT_TESTS)
  message("Building unit tests.")
  add_subdirectory(unit)
endif()

catch_discover_tests(phoebus_unit_tests
  TEST_SPEC "[geometry]" TEST_PREFIX "geometry:")
catch_discover_tests(phoebus_unit_tests
  TEST_SPEC "[MonopoleGR]" TEST_PREFIX "MonopoleGR:")
catch_discover_tests(phoebus_unit_tests
  TEST_SPEC "[relativity_utils]" TEST_PREFIX "relativity_utils:")
catch_discover_tests(phoebus_unit_tests 
  TEST_SPEC "[radiation]" TEST_PREFIX "radiation:")
catch_discover_tests(phoebus_unit_tests 
  TEST_SPEC "[reader_utils]" TEST_PREFIX "reader:")

if(PHOEBUS_ENABLE_DOWNLOADS)
  catch_discover_tests(phoebus_unit_tests 
    TEST_SPEC "[compute_adiabats]" TEST_PREFIX "adiabats:")
endif()

# Currently integration and regression tests 
# are not integrated into the cmake build system 
#if(PHOEBUS_ENABLE_INTEGRATION_TESTS)
#  message("Building integration tests.")
#  add_subdirectory(integration)
#endif()
#if(PHOEBUS_ENABLE_REGRESSION_TESTS)
#  message("Building regression tests.")
#  add_subdirectory(regression)
#endif()
