// TODO(BRR) use singularity-eos

#ifndef RADIATION_OPACITY_HPP_
#define RADIATION_OPACITY_HPP_

namespace radiation {

#define C (1.)
#define numax (1.e17)
#define numin (1.e7)

KOKKOS_INLINE_FUNCTION
Real Getyf(Real Ye, NeutrinoSpecies s) {
  if (s == NeutrinoSpecies::Electron) {
    return 2. * Ye;
  } else if (s == NeutrinoSpecies::ElectronAnti) {
    return 1. - 2. * Ye;
  } else {
    return 0.;
  }
}

KOKKOS_INLINE_FUNCTION
Real Getjnu(const Real Ye, const NeutrinoSpecies s, const Real nu) {
  if (nu > numin && nu < numax) {
    return C*Getyf(Ye, s);
  } else {
    return 0.;
  }
}

KOKKOS_INLINE_FUNCTION
Real GetJnu(const Real Ye, const NeutrinoSpecies s, const Real nu) {
  if (nu > numin && nu < numax) {
    return 4.*M_PI*C*Getyf(Ye, s);
  } else {
    return 0.;
  }
}

KOKKOS_INLINE_FUNCTION
Real GetJ(const Real rho_cgs, const Real Ye, const NeutrinoSpecies s) {
  Real Bc = C * (numax - numin);
  Real J = Bc * Getyf(Ye, s);
  return J;
}

KOKKOS_INLINE_FUNCTION
Real GetJye(const Real rho_cgs, const Real Ye, const NeutrinoSpecies s) {
  Real Ac = pc.mp / (pc.h * rho_cgs) * C * log(numax / numin);
  return rho_cgs * Ac * Getyf(Ye, s);
}

#undef C
#undef numax
#undef numin

} // namespace radiation

#endif // RADIATION_OPACITY_HPP_
