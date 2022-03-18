// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef COMPILE_CONSTANTS_HPP_
#define COMPILE_CONSTANTS_HPP_

#define NCONS_MAX 10

#define PHOEBUS_GEOMETRY Geometry::Minkowski
#define GEOMETRY_MESH Analytic<Minkowski, IndexerMesh>
#define GEOMETRY_MESH_BLOCK Analytic<Minkowski, IndexerMeshBlock>

#define PRINT_RHS 0

#endif // COMPILE_CONSTANTS_HPP_
