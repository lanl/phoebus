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

include(FetchContent)

if(PHOEBUS_ENABLE_DOWNLOADS)
  set(FN "Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2")
  message(STATUS "Fetching SFHo table as needed (this may take a moment..)")
  FetchContent_Declare(table
    #URL https://stellarcollapse.org/~evanoc/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2
    URL https://stellarcollapse.org/~evanoc/${FN}
    URL_HASH SHA256=5a76c40dcff0f027fb79a7fd46fa1c867f7e592428ec01796e8e13800f694154
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/table_dir
    DOWNLOAD_NO_EXTRACT true
  )
  FetchContent_MakeAvailable(table)
  
  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/table_dir/${FN})
  message(STATUS "Extracting table (this will take a moment...)")
  execute_process(
    COMMAND bzip2 -d 
    ${CMAKE_CURRENT_BINARY_DIR}/table_dir/${FN}
    #OUTPUT_VARIABLE extract_status
    )
  endif()
endif()

add_library(phoebus_utils_unit_tests OBJECT
  test_relativity_utils.cpp
  test_reader.cpp
  test_adiabats.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/geometry/geometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/progenitor/ascii_reader.cpp
)

target_link_libraries(phoebus_utils_unit_tests
  PRIVATE
    parthenon
    Catch2::Catch2
    Kokkos::kokkos
    singularity-eos::singularity-eos
    )

target_include_directories(phoebus_utils_unit_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/singularity-eos"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/singularity-eos/utils"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  )


target_link_libraries(phoebus_utils_unit_tests PRIVATE Catch2::Catch2)
target_sources(phoebus_unit_tests PRIVATE $<TARGET_OBJECTS:phoebus_utils_unit_tests>)
