# © 2021-2025. Triad National Security, LLC. All rights reserved.  This
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

option(PHOEBUS_DO_NU_ELECTRON "Evolve electron neutrinos" ON)
option(PHOEBUS_DO_NU_ELECTRON_ANTI "Evolve electron antineutrinos" ON)
option(PHOEBUS_DO_NU_HEAVY "Evolve heavy composite type neutrinos" ON)

if(PHOEBUS_DO_NU_ELECTRON)
  set(DO_NU_ELECTRON 1 CACHE BOOL "Electron neutrinos enabled")
else()
  set(DO_NU_ELECTRON 0 CACHE BOOL "Electron neutrinos disabled")
endif()

if(PHOEBUS_DO_NU_ELECTRON_ANTI)
  set(DO_NU_ELECTRON_ANTI 1 CACHE BOOL "Electron antineutrinos enabled")
else()
  set(DO_NU_ELECTRON_ANTI 0 CACHE BOOL "Electron antineutrinos disabled")
endif()

if(PHOEBUS_DO_NU_HEAVY)
  set(DO_NU_HEAVY 1 CACHE BOOL "Heavy neutrinos enabled")
else()
  set(DO_NU_HEAVY 0 CACHE BOOL "Heavy neutrinos disabled")
endif()
