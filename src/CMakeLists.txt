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
  fixup/fixup.hpp
  fixup/fixup.cpp
  fixup/fixup_c2p.cpp
  fixup/fixup_radc2p.cpp
  fixup/fixup_src.cpp
  fixup/fixup_netfield.cpp

  fluid/con2prim.hpp
  fluid/con2prim_robust.hpp
  fluid/con2prim_statistics.cpp
  fluid/con2prim_statistics.hpp
  fluid/fluid.cpp
  fluid/fluid.hpp
  fluid/riemann.hpp
  fluid/tmunu.cpp
  fluid/tmunu.hpp

  geometry/analytic_system.hpp
  geometry/boosted_minkowski.hpp
  geometry/boosted_minkowski.cpp
  geometry/boyer_lindquist.hpp
  geometry/boyer_lindquist.cpp
  geometry/cached_system.hpp
  geometry/coordinate_systems.hpp
  geometry/flrw.hpp
  geometry/flrw.cpp
  geometry/fmks.hpp
  geometry/fmks.cpp
  geometry/geometry.cpp
  geometry/geometry.hpp
  geometry/geometry_defaults.hpp
  geometry/geometry_utils.hpp
  geometry/sph2cart.cpp
  geometry/sph2cart.hpp
  geometry/tetrads.hpp

  geometry/inchworm.hpp
  geometry/inchworm.cpp
  geometry/mckinney_gammie_ryan.hpp
  geometry/mckinney_gammie_ryan.cpp
  geometry/modified_system.hpp
  geometry/minkowski.hpp
  geometry/minkowski.cpp
  geometry/monopole.hpp
  geometry/monopole.cpp
  geometry/snake.hpp
  geometry/snake.cpp
  geometry/spherical_kerr_schild.hpp
  geometry/spherical_kerr_schild.cpp
  geometry/spherical_minkowski.hpp
  geometry/spherical_minkowski.cpp

  microphysics/eos_phoebus/eos_phoebus.hpp
  microphysics/eos_phoebus/eos_phoebus.cpp
  microphysics/opac_phoebus/opac_phoebus.hpp
  microphysics/opac_phoebus/opac_phoebus.cpp

  monopole_gr/interp_3d_to_1d.hpp
  monopole_gr/monopole_gr.hpp
  monopole_gr/monopole_gr_base.hpp
  monopole_gr/monopole_gr_interface.hpp
  monopole_gr/monopole_gr_utils.hpp
  monopole_gr/monopole_gr.cpp

  pgen/pgen.hpp
  pgen/pgen.cpp
  pgen/advection.cpp
  pgen/blandford_mckee.cpp
  pgen/bondi.cpp
  pgen/check_cached_geom.cpp
  pgen/friedmann.cpp
  pgen/homogeneous_sphere.cpp
  pgen/progenitor.cpp
  pgen/kh.cpp
  pgen/leptoneq.cpp
  pgen/linear_modes.cpp
  pgen/p2c2p.cpp
  pgen/radiation_advection.cpp
  pgen/radiation_equilibration.cpp
  pgen/rhs_tester.cpp
  pgen/rotor.cpp
  pgen/shock_tube.cpp
  pgen/thin_cooling.cpp
  pgen/sedov.cpp
  pgen/torus.cpp
  pgen/tov.cpp

  phoebus_boundaries/phoebus_boundaries.hpp
  phoebus_boundaries/phoebus_boundaries.cpp

  phoebus_driver.cpp
  phoebus_driver.hpp
  phoebus_package.cpp
  phoebus_package.hpp

  phoebus_utils/adiabats.hpp
  phoebus_utils/cell_locations.hpp
  phoebus_utils/unit_conversions.cpp
  phoebus_utils/unit_conversions.hpp
  phoebus_utils/linear_algebra.hpp
  phoebus_utils/phoebus_interpolation.hpp
  phoebus_utils/preprocessor_utils.hpp
  phoebus_utils/reduction.hpp
  phoebus_utils/relativity_utils.hpp
  phoebus_utils/robust.hpp
  phoebus_utils/root_find.hpp
  phoebus_utils/variables.hpp

  progenitor/ascii_reader.cpp
  progenitor/ascii_reader.hpp
  progenitor/progenitordata.hpp
  progenitor/progenitordata.cpp
  
  radiation/radiation.hpp
  radiation/radiation.cpp
  radiation/cooling_function.cpp
  radiation/mocmc.cpp
  radiation/moments.cpp
  radiation/moments_source.cpp
  radiation/monte_carlo.cpp
  radiation/geodesics.hpp

  reconstruction.hpp

  analysis/history.hpp
  analysis/history_utils.hpp
  analysis/history.cpp

  tov/tov.hpp
  tov/tov.cpp
)

set(MAX_NUMBER_CONSERVED_VARS 9 CACHE INTEGER "Maximum number of conserved variables")

configure_file(compile_constants.hpp.in generated/compile_constants.hpp @ONLY)

target_include_directories(phoebus PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos"
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-opac"
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/utils/herumi-fmath"
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/utils/ports-of-call"
)

target_compile_features(phoebus PUBLIC cxx_std_14)
if (CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  target_compile_options(phoebus PUBLIC -std=c++1y -qxflag=disable__cplusplusOverride)
endif()

if (Kokkos_ENABLE_CUDA)
  target_compile_options(phoebus PUBLIC --expt-relaxed-constexpr)
endif()

# target_compile_options(phoebus PUBLIC -G -lineinfo)

target_link_libraries(phoebus PRIVATE parthenon
  singularity-eos::singularity-eos
  singularity-opac::singularity-opac
)
