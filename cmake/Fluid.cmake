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

option(PHOEBUS_VALENCIA "Use Valencia formulation of GRMHD" ON)

if(PHOEBUS_VALENCIA)
  set(USE_VALENCIA 1 CACHE BOOL "Valencia formulation of GRMHD")
else()
  set(USE_VALENCIA 0 CACHE BOOL "Covariant formulation of GRMHD")
endif()

option(PHOEBUS_FLUX_AND_SRC_DIAGS "Stash flux divergence and sources for possible output" OFF)
if(PHOEBUS_FLUX_AND_SRC_DIAGS)
  set(SET_FLUX_SRC_DIAGS 1 CACHE BOOL "Allocate and set these diagnostics")
else()
  set(SET_FLUX_SRC_DIAGS 0 CACHE BOOL "Skip these diagnostics")
endif()

option(PHOEBUS_CON2PRIM_STATISTICS "Gather and report a histogram of iteration counts" OFF)
if(PHOEBUS_CON2PRIM_STATISTICS)
  set(CON2PRIM_STATISTICS 1 CACHE BOOL "Allocate, set, and report")
else()
  set(CON2PRIM_STATISTICS 0 CACHE BOOL "Skip it")
endif()
