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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

/*
 * Test for near-machine-precision equivalence of Real values
 *
 * PARAM[IN] - val - Value to test equivalence of
 * PARAM[IN] - ref - Value against which to test equivalence
 * PARAM[IN] - tol - Tolerance of equivalent test
 * PARAM[IN] - ignore_small - If both val and ref are small compared to tol, return true
 *
 * RETURN - True if val and
 */
KOKKOS_INLINE_FUNCTION
bool SoftEquiv(const Real &val, const Real &ref, const Real tol = 1.e-8,
               const bool ignore_small = true) {
  if (ignore_small) {
    if (fabs(val) < tol && fabs(ref) < tol) {
      return true;
    }
  }

  if (fabs(val - ref) < tol * fabs(ref) / 2) {
    return true;
  } else {
    return false;
  }
}

/* Create a dummy MeshBlock with a single dummy package and field. 1D grid assumed. Useful
 * for constructing other objects e.g. MeshBlockData<Real> and CoordSysMeshBlock
 *
 * PARAM[IN] - nzones - Number of zones in x direction
 *
 * RETURN shared_ptr to trivial MeshBlock
 */
std::shared_ptr<MeshBlock> inline GetDummyMeshBlock(const int nzones = 8) {
  return std::make_shared<MeshBlock>(nzones, 1);
}

/*
 * Create a dummy MeshBlockData with a single dummy package and field. 1D grid assumed.
 * Useful for constructing other objects e.g. CoordSysMeshBlocks.
 *
 * PARAM[IN] - pmb - shared_ptr to existing MeshBlock
 *
 * RETURN Trivial MeshBlockData<Real>
 */
std::shared_ptr<MeshBlockData<Real>> inline GetDummyMeshBlockData(
    std::shared_ptr<MeshBlock> pmb) {
  auto pkg = std::make_shared<StateDescriptor>("Dummy package");
  std::vector<int> scalar_shape{1, 1, 1};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  pkg->AddField("dummy_field", m);

  auto rc = std::make_shared<MeshBlockData<Real>>();
  rc.get()->Initialize(pkg, pmb);
  return rc;
}

#endif // TEST_UTILS_HPP_
