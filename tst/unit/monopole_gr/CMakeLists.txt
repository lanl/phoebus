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

add_library(monopole_gr_unit_tests OBJECT
  test_monopole_gr.cpp
  test_interp_3d_to_1d.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/monopole_gr/monopole_gr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/monopole_gr/monopole_gr_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/monopole_gr/monopole_gr.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/tov/tov.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/tov/tov.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/phoebus_utils/unit_conversions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/phoebus_utils/unit_conversions.cpp
  
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/microphysics/eos_phoebus/eos_phoebus.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/microphysics/eos_phoebus/eos_phoebus.cpp
  )

target_link_libraries(monopole_gr_unit_tests
  PRIVATE
  parthenon
  Catch2::Catch2
  Kokkos::kokkos
  singularity-eos::singularity-eos
  )

target_include_directories(monopole_gr_unit_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/singularity-eos"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/singularity-eos/utils"
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  )

target_sources(phoebus_unit_tests PRIVATE $<TARGET_OBJECTS:monopole_gr_unit_tests>)
