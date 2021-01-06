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

# Load machine specific defaults such as architecture and mpi launch command.
# Command line argument takes precedence over environment variable.
# Loading this before project definition to allow setting the compiler.
# In fact, for some reason, this needs to be above the `cmake_minimum_required`.
if (MACHINE_CFG)
  if(EXISTS "${MACHINE_CFG}")
    include(${MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration at "
      "${MACHINE_CFG} not found.")
  endif()
elseif (DEFINED ENV{MACHINE_CFG})
  if(EXISTS "$ENV{MACHINE_CFG}")
    include($ENV{MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration from environment variable "
      "MACHINE_CFG at $ENV{MACHINE_CFG} not found.")
  endif()
else()
  message(WARNING "Not using any machine configuration. Consider creating a configuration "
    "file following the examples in ${PROJECT_SOURCE_DIR}/external/parthenon/cmake/machine_cfgs/ and then "
    "point the MACHINE_CFG variable to your custom file."
    "Note, that the machine file can be placed in any directory (also outside the repo).")
endif()

# Boiler plate
cmake_minimum_required(VERSION 3.13)
project(phoebus LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

# TODO(JMM): Tests? Copy what's in riot? Hack something in?
# TODO(JMM): Pgen?
option(PHOEBUS_ENABLE_CUDA "Enable cuda for riot and all dependencies" OFF)
option(PHOEBUS_ENABLE_HDF5 "Enable HDF5 for riot and all dependencies" ON)
option(PHOEBUS_ENABLE_MPI "Enable MPI for riot and all dependencies" ON)
option(PHOEBUS_ENABLE_OPENMP "Enable OpenMP for riot and parthenon" OFF)

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
    "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# append to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# TODO(JMM): do we want parent flags for both singularity and
# parthenon? I dunno.
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)

# CUDA
if(PHOEBUS_ENABLE_CUDA)
  set(SINGULARITY_USE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
endif()

# HDF5
if(PHOEBUS_ENABLE_HDF5)
  set(SINGULARITY_USE_HDF5 ON CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
else()
  set(SINGULARITY_USE_HDF5 OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 ON CACHE BOOL "" FORCE)
endif()

if(PHOEBUS_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

if(PHOEBUS_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
else()
  set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

# TODO(JMM): For some reason, order still maters for including
# parthenon and singularity. Likely has to do with project
# includes other than Kokkos. MPI and OpenMP likely culprits.

# parthenon
# TODO(JMM): This should be renamed PARTHENON_BUILD_TESTING
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(external/parthenon parthenon)

# singularity eos
set(SINGULARITY_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
set(SINGULARITY_SUBMODULE_MODE ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_KOKKOSKERNELS OFF CACHE BOOL "" FORCE)
set(SINGULARITY_BUILD_MIX OFF CACHE BOOL "" FORCE)
add_subdirectory(external/singularity-eos singularity-eos)

# Phoebus src
add_subdirectory(src)

# TODO(JMM): Testing?
# add testing subdirs, etc here
