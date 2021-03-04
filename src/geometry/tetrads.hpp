#include <parthenon/package.hpp>

#include "geometry/coordinate_systems.hpp"

#ifndef GEOMETRY_TETRADS_HPP_
#define GEOMETRY_TETRADS_HPP_

namespace Geometry {

//#define MULOOP for (int mu = 0; mu < NDFULL; mu++)
//#define SPACETIMELOOP(nu) for (int nu = 0; nu < NDFULL; nu++)

#define SPACETIMELOOP(mu) for (int mu = 0; mu < NDFULL; mu++)

class Tetrads {
public:
  KOKKOS_FUNCTION
  Tetrads(const double Ucon[NDFULL], const double Trial[NDFULL],
          const double Gcov[NDFULL][NDFULL]) {

    Real X1ness = 0.;
    Real X2ness = 0.;
    Real X3ness = 0.;
    SPACETIMELOOP(mu) {
      X1ness += Gcov[1][mu] * Trial[mu] / sqrt(fabs(Gcov[1][1]));
      X2ness += Gcov[2][mu] * Trial[mu] / sqrt(fabs(Gcov[2][2]));
      X3ness += Gcov[3][mu] * Trial[mu] / sqrt(fabs(Gcov[3][3]));
    }
    X1ness = fabs(X1ness);
    X2ness = fabs(X2ness);
    X3ness = fabs(X3ness);

    // Normalize Trial vector
    double norm = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { norm += Trial[mu] * Trial[nu] * Gcov[mu][nu]; }

    // Time component parallel to u^{\mu}
    SPACETIMELOOP(mu) { Econ_[0][mu] = Ucon[mu]; }

    // Use X1, X2, X3. Then, whichever is closest to Trial, overwrite.
    SPACETIMELOOP(mu) {
      Econ_[1][mu] = Delta_(mu, 1);
      Econ_[2][mu] = Delta_(mu, 2);
      Econ_[3][mu] = Delta_(mu, 3);
    }

    if (norm > SMALL) {
      // We can use the Trial vector
      if (X1ness > X2ness && X1ness > X3ness) {
        // Trial vector is closest to X1. Overwrite
        SPACETIMELOOP(mu) { Econ_[1][mu] = Trial[mu]; }
      } else if (X2ness >= X1ness && X2ness > X3ness) {
        // Trial vector is closest to X2. Overwrite
        SPACETIMELOOP(mu) { Econ_[2][mu] = Trial[mu]; }
      } else { // Trial vector is closest X3. Overwrite
        SPACETIMELOOP(mu) { Econ_[3][mu] = Trial[mu]; }
      }
    }

    // Gram-Schmidt and normalization
    Normalize_(Econ_[0], Gcov);
    ProjectOut_(Econ_[1], Econ_[0], Gcov);
    Normalize_(Econ_[1], Gcov);
    ProjectOut_(Econ_[2], Econ_[0], Gcov);
    ProjectOut_(Econ_[2], Econ_[1], Gcov);
    Normalize_(Econ_[2], Gcov);
    ProjectOut_(Econ_[3], Econ_[0], Gcov);
    ProjectOut_(Econ_[3], Econ_[1], Gcov);
    ProjectOut_(Econ_[3], Econ_[2], Gcov);
    Normalize_(Econ_[3], Gcov);

    // Make covariant version
    SPACETIMELOOP(mu) { Lower_(Econ_[mu], Gcov, Ecov_[mu]); }
    SPACETIMELOOP(mu) { Ecov_[0][mu] *= -1.; }
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCon(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    SPACETIMELOOP(mu) {
      VTetrad[mu] = 0.;
      SPACETIMELOOP(nu) { VTetrad[mu] += Ecov_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void CoordToTetradCov(const Real VCoord[NDFULL], Real VTetrad[NDFULL]) {
    SPACETIMELOOP(mu) {
      VTetrad[mu] = 0.;
      SPACETIMELOOP(nu) { VTetrad[mu] += Econ_[mu][nu] * VCoord[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCon(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    SPACETIMELOOP(mu) {
      VCoord[mu] = 0.;
      SPACETIMELOOP(nu) { VCoord[mu] += Econ_[nu][mu] * VTetrad[nu]; }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void TetradToCoordCov(const Real VTetrad[NDFULL], Real VCoord[NDFULL]) {
    SPACETIMELOOP(mu) {
      VCoord[mu] = 0.;
      SPACETIMELOOP(nu) { VCoord[mu] += Ecov_[nu][mu] * VTetrad[nu]; }
    }
  }

private:
  KOKKOS_INLINE_FUNCTION
  void Normalize_(double Vcon[NDFULL], const double Gcov[NDFULL][NDFULL]) {
    double norm = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { norm += Vcon[mu] * Vcon[nu] * Gcov[mu][nu]; }

    norm = sqrt(fabs(norm));
    SPACETIMELOOP(mu) { Vcon[mu] /= norm; }
  }

  KOKKOS_INLINE_FUNCTION
  void ProjectOut_(double Vcona[NDFULL], double Vconb[NDFULL],
                   const double Gcov[NDFULL][NDFULL]) {
    double Vconb_sq = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) {
      Vconb_sq += Vconb[mu] * Vconb[nu] * Gcov[mu][nu];
    }

    double adotb = 0.;
    SPACETIMELOOP(mu) SPACETIMELOOP(nu) { adotb += Vcona[mu] * Vconb[nu] * Gcov[mu][nu]; }

    SPACETIMELOOP(mu) { Vcona[mu] -= Vconb[mu] * adotb / Vconb_sq; }
  }

  KOKKOS_INLINE_FUNCTION
  void Lower_(const double Vcon[NDFULL], const double Gcov[NDFULL][NDFULL],
              double Vcov[NDFULL]) {
    Vcov[0] = Gcov[0][0] * Vcon[0] + Gcov[0][1] * Vcon[1] + Gcov[0][2] * Vcon[2] +
              Gcov[0][3] * Vcon[3];
    Vcov[1] = Gcov[1][0] * Vcon[0] + Gcov[1][1] * Vcon[1] + Gcov[1][2] * Vcon[2] +
              Gcov[1][3] * Vcon[3];
    Vcov[2] = Gcov[2][0] * Vcon[0] + Gcov[2][1] * Vcon[1] + Gcov[2][2] * Vcon[2] +
              Gcov[2][3] * Vcon[3];
    Vcov[3] = Gcov[3][0] * Vcon[0] + Gcov[3][1] * Vcon[1] + Gcov[3][2] * Vcon[2] +
              Gcov[3][3] * Vcon[3];
  }

  KOKKOS_INLINE_FUNCTION
  void Raise_(const double Vcov[NDFULL], const double Gcon[NDFULL][NDFULL],
              double Vcon[NDFULL]) {
    Vcon[0] = Gcon[0][0] * Vcov[0] + Gcon[0][1] * Vcov[1] + Gcon[0][2] * Vcov[2] +
              Gcon[0][3] * Vcov[3];
    Vcon[1] = Gcon[1][0] * Vcov[0] + Gcon[1][1] * Vcov[1] + Gcon[1][2] * Vcov[2] +
              Gcon[1][3] * Vcov[3];
    Vcon[2] = Gcon[2][0] * Vcov[0] + Gcon[2][1] * Vcov[1] + Gcon[2][2] * Vcov[2] +
              Gcon[2][3] * Vcov[3];
    Vcon[3] = Gcon[3][0] * Vcov[0] + Gcon[3][1] * Vcov[1] + Gcon[3][2] * Vcov[2] +
              Gcon[3][3] * Vcov[3];
  }

  KOKKOS_INLINE_FUNCTION
  int Delta_(const int a, const int b) {
    if (a == b) {
      return 1;
    } else {
      return 0;
    }
  }

  Real Econ_[NDFULL][NDFULL];
  Real Ecov_[NDFULL][NDFULL];
};

#undef SPACETIMELOOP
//#undef MULOOP
//#undef SPACETIMELOOP(nu)

} // namespace Geometry

#endif // GEOMETRY_TETRADS_HPP_
