#=========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

add_executable(phoebus
  main.cpp # first

  # alphabetical

  fluid/con2prim.cpp
  fluid/con2prim.hpp
  fluid/fluid.cpp
  fluid/fluid.hpp
  fluid/tmunu.hpp

  geometry/coordinate_systems.hpp
  geometry/geometry_variant.hpp
  geometry/geometry.cpp
  geometry/geometry.hpp

  initial_conditions.hpp
  initial_conditions.cpp

  microphysics/eos_phoebus/eos_phoebus.hpp
  microphysics/eos_phoebus/eos_phoebus.cpp

  phoebus_driver.cpp
  phoebus_driver.hpp

  phoebus_utils/cell_locations.hpp

  reconstruction.hpp
)

target_include_directories(phoebus PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos"
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/utils/herumi-fmath"
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/utils/ports-of-call"
)

target_compile_features(phoebus PUBLIC cxx_std_14)
if (CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  target_compile_options(riot PUBLIC -std=c++1y -qxflag=disable__cplusplusOverride)
endif()

if (Kokkos_ENABLE_CUDA)
  target_compile_options(phoebus PUBLIC --expt-relaxed-constexpr)
endif()

target_link_libraries(phoebus PRIVATE parthenon
  singularity-eos::flags
  singularity-eos
  $<$<BOOL:${PHOEBUS_ENABLE_HDF5}>:singularity-eos::hdf5>
)
