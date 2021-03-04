#include <parthenon/package.hpp>

#include "geometry/coordinate_systems.hpp"

#ifndef GEOMETRY_TETRADS_HPP_
#define GEOMETRY_TETRADS_HPP_

namespace Geometry {

#define MULOOP for (int mu = 0; mu < NDFULL; mu++)
#define NULOOP for (int nu = 0; nu < NDFULL; nu++)

class Tetrads {
public:
  KOKKOS_FUNCTION
  Tetrads(const double UCon[NDFULL], const double Trial[NDFULL],
          const double GCov[NDFULL][NDFULL]) {

    Real X1ness = 0.;
    Real X2ness = 0.;
    Real X3ness = 0.;
    MULOOP {
      X1ness += GCov[1][mu] * Trial[mu] / sqrt(fabs(GCov[1][1]));
      X2ness += GCov[2][mu] * Trial[mu] / sqrt(fabs(GCov[2][2]));
      X3ness += GCov[3][mu] * Trial[mu] / sqrt(fabs(GCov[3][3]));
    }
    X1ness = fabs(X1ness);
    X2ness = fabs(X2ness);
    X3ness = fabs(X3ness);

    // Normalize Trial vector
    double norm = 0.;
    MULOOP NULOOP { norm += Trial[mu] * Trial[nu] * GCov[mu][nu]; }

    // Time component parallel to u^{\mu}
    MULOOP { ECon_[0][mu] = UCon[mu]; }

    // Use X1, X2, X3. Then, whichever is closest to Trial, overwrite.
    MULOOP {
      ECon_[1][mu] = Delta_(mu, 1);
      ECon_[2][mu] = Delta_(mu, 2);
      ECon_[3][mu] = Delta_(mu, 3);
    }

    if (norm > SMALL) {
      // We can use the Trial vector
      if (X1ness > X2ness && X1ness > X3ness) {
        // Trial vector is closest to X1. Overwrite
        MULOOP { ECon_[1][mu] = Trial[mu]; }
      } else if (X2ness >= X1ness && X2ness > X3ness) {
        // Trial vector is closest to X2. Overwrite
        MULOOP { ECon_[2][mu] = Trial[mu]; }
      } else { // Trial vector is closest X3. Overwrite
        MULOOP { ECon_[3][mu] = Trial[mu]; }
      }
    }

    // Gram-Schmidt and normalization
    Normalize_(ECon_[0], GCov);
    ProjectOut_(ECon_[1], ECon_[0], GCov);
    Normalize_(ECon_[1], GCov);
    ProjectOut_(ECon_[2], ECon_[0], GCov);
    ProjectOut_(ECon_[2], ECon_[1], GCov);
    Normalize_(ECon_[2], GCov);
    ProjectOut_(ECon_[3], ECon_[0], GCov);
    ProjectOut_(ECon_[3], ECon_[1], GCov);
    ProjectOut_(ECon_[3], ECon_[2], GCov);
    Normalize_(ECon_[3], GCov);

    // Make covariant version
    MULOOP { Lower_(ECon_[mu], GCov, ECov_[mu]); }
    MULOOP { ECov_[0][mu] *= -1.; }
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCon(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    MULOOP {
      VTetrad[mu] = 0.;
      NULOOP { VTetrad[mu] += ECov_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCov(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    MULOOP {
      VTetrad[mu] = 0.;
      NULOOP { VTetrad[mu] += ECon_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCon(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    MULOOP {
      VCoord[mu] = 0.;
      NULOOP { VCoord[mu] += ECon_[nu][mu] * VTetrad[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCov(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    MULOOP {
      VCoord[mu] = 0.;
      NULOOP { VCoord[mu] += ECov_[nu][mu] * VTetrad[nu]; }
    }
  }

private:
  KOKKOS_INLINE_FUNCTION
  void Normalize_(double VCon[NDFULL], const double GCov[NDFULL][NDFULL]) {
    double norm = 0.;
    MULOOP NULOOP { norm += VCon[mu] * VCon[nu] * GCov[mu][nu]; }

    norm = sqrt(fabs(norm));
    MULOOP { VCon[mu] /= norm; }
  }

  KOKKOS_INLINE_FUNCTION
  void ProjectOut_(double VCona[NDFULL], double VConb[NDFULL],
                   const double GCov[NDFULL][NDFULL]) {
    double VConb_sq = 0.;
    MULOOP NULOOP { VConb_sq += VConb[mu] * VConb[nu] * GCov[mu][nu]; }

    double adotb = 0.;
    MULOOP NULOOP { adotb += VCona[mu] * VConb[nu] * GCov[mu][nu]; }

    MULOOP { VCona[mu] -= VConb[mu] * adotb / VConb_sq; }
  }

  KOKKOS_INLINE_FUNCTION
  void Lower_(const double VCon[NDFULL], const double GCov[NDFULL][NDFULL],
              double VCov[NDFULL]) {
    VCov[0] = GCov[0][0] * VCon[0] + GCov[0][1] * VCon[1] + GCov[0][2] * VCon[2] +
              GCov[0][3] * VCon[3];
    VCov[1] = GCov[1][0] * VCon[0] + GCov[1][1] * VCon[1] + GCov[1][2] * VCon[2] +
              GCov[1][3] * VCon[3];
    VCov[2] = GCov[2][0] * VCon[0] + GCov[2][1] * VCon[1] + GCov[2][2] * VCon[2] +
              GCov[2][3] * VCon[3];
    VCov[3] = GCov[3][0] * VCon[0] + GCov[3][1] * VCon[1] + GCov[3][2] * VCon[2] +
              GCov[3][3] * VCon[3];
  }

  KOKKOS_INLINE_FUNCTION
  void Raise_(const double VCov[NDFULL], const double GCon[NDFULL][NDFULL],
              double VCon[NDFULL]) {
    VCon[0] = GCon[0][0] * VCov[0] + GCon[0][1] * VCov[1] + GCon[0][2] * VCov[2] +
              GCon[0][3] * VCov[3];
    VCon[1] = GCon[1][0] * VCov[0] + GCon[1][1] * VCov[1] + GCon[1][2] * VCov[2] +
              GCon[1][3] * VCov[3];
    VCon[2] = GCon[2][0] * VCov[0] + GCon[2][1] * VCov[1] + GCon[2][2] * VCov[2] +
              GCon[2][3] * VCov[3];
    VCon[3] = GCon[3][0] * VCov[0] + GCon[3][1] * VCov[1] + GCon[3][2] * VCov[2] +
              GCon[3][3] * VCov[3];
  }

  KOKKOS_INLINE_FUNCTION
  int Delta_(const int a, const int b) {
    if (a == b) {
      return 1;
    } else {
      return 0;
    }
  }

  Real ECon_[NDFULL][NDFULL];
  Real ECov_[NDFULL][NDFULL];
};

#undef MULOOP
#undef NULOOP

} // namespace Geometry

#endif // GEOMETRY_TETRADS_HPP_
