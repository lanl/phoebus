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

option(PHOEBUS_ANALYTIC_GEOMETRY "Use an analytic geometry" ON) # TODO(JMM): clean up later
option(PHOEBUS_CACHE_GEOMETRY "Cache geometry. Only available for some geometries" OFF)

# Default
set(PHOEBUS_GEOMETRY "Minkowski" CACHE STRING "The metric used by Phoebus")

set(PHOEBUS_ANALYTIC_GEOMETRIES
    "Minkowski"
    "BoostedMinkowski"
    "SphericalMinkowski"
    "BoyerLindquist"
    "SphericalKerrSchild"
    "FMKS"
    "Snake"
    "Inchworm"
    "MonopoleSph"
    "MonopoleCart"
    )
set(PHOEBUS_GEOMETRY_NO_CACHE
    "MonopoleSph"
    "MonopoleCart"
    )

if(PHOEBUS_ANALYTIC_GEOMETRY)
  if (PHOEBUS_GEOMETRY IN_LIST PHOEBUS_ANALYTIC_GEOMETRIES)
    set(GEOMETRY_MESH "Analytic<${PHOEBUS_GEOMETRY}, IndexerMesh>")
    set(GEOMETRY_MESH_BLOCK "Analytic<${PHOEBUS_GEOMETRY}, IndexerMeshBlock>")
    if (PHOEBUS_CACHE_GEOMETRY)
      if (PHOEBUS_GEOMETRY IN_LIST PHOEBUS_GEOMETRY_NO_CACHE)
	message(WARNING "The selected geometry does not support caching. "
	  "You may experience unexpected behaviour.")
      endif()
      set(GEOMETRY_MESH "CachedOverMesh<${GEOMETRY_MESH}>")
      set(GEOMETRY_MESH_BLOCK "CachedOverMeshBlock<${GEOMETRY_MESH_BLOCK}>")
    endif()
  else()
    message(FATAL_ERROR "Unknown geometry")
  endif()
else()
  message(FATAL_ERROR "Only analytic geometries currently supported")
endif()

message("\nGeometry set to ${PHOEBUS_GEOMETRY}")
message(STATUS "On MeshBlocks:     ${GEOMETRY_MESH_BLOCK}")
message(STATUS "On MeshBlockPacks: ${GEOMETRY_MESH}")
