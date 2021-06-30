// TODO(BRR) use singularity-eos

#ifndef RADIATION_OPACITY_HPP_
#define RADIATION_OPACITY_HPP_

namespace radiation {

#define SCONST (3)

class Opacity {
public:
  KOKKOS_INLINE_FUNCTION
  Real Getalphanu(const Real rho, const Real T, const Real Ye, const Real nu,
                  const NeutrinoSpecies s) {
    Real Bnu = GetBnu(T, nu);
    Real jnu = Getjnu(rho, T, Ye, s, nu);
    return jnu / Bnu;
  }

  KOKKOS_FUNCTION
  virtual Real Getjnu(const Real rho, const Real T, const Real Ye,
                      const NeutrinoSpecies s, const Real nu) = 0;

  KOKKOS_FUNCTION
  virtual Real GetJnu(const Real rho, const Real T, const Real Ye,
                      const NeutrinoSpecies s, const Real nu) = 0;

  KOKKOS_FUNCTION
  virtual Real GetJ(const Real rho, const Real T, const Real Ye,
                    const NeutrinoSpecies s) = 0;

  KOKKOS_FUNCTION
  virtual Real GetJye(const Real rho, const Real T, const Real Ye,
                      const NeutrinoSpecies s) = 0;

protected:
  KOKKOS_INLINE_FUNCTION
  Real GetBnu(const Real T, const Real nu) {
    Real x = pc.h * nu / (pc.kb * T);
    Real Bnu = SCONST * (2. * pc.h * nu * nu * nu / (pc.c * pc.c)) * 1. / (exp(x) + 1.);
    return Bnu;
  }

  KOKKOS_INLINE_FUNCTION
  Real GetB(const Real T) {
    return 8. * pow(M_PI, 5) * pow(pc.kb, 4) * SCONST * pow(T, 4) /
           (15. * pow(pc.c, 2) * pow(pc.h, 3));
  }
};

class GrayOpacity : public Opacity {
public:
  GrayOpacity(const Real kappa) : kappa_(kappa) {}

  KOKKOS_INLINE_FUNCTION
  Real Getjnu(const Real rho, const Real T, const Real Ye, const NeutrinoSpecies s,
              const Real nu) override {
    Real Bnu = GetBnu(T, nu);
    return kappa_ * Bnu;
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJnu(const Real rho, const Real T, const Real Ye, const NeutrinoSpecies s,
              const Real nu) override {
    return 4. * M_PI * Getjnu(rho, T, Ye, s, nu);
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJ(const Real rho, const Real T, const Real Ye,
            const NeutrinoSpecies s) override {
    return kappa_ * GetB(T);
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJye(const Real rho, const Real T, const Real Ye,
              const NeutrinoSpecies s) override {
    const Real zeta3 = 1.20206;
    return 12. * pow(pc.kb, 3) * pc.mp * M_PI * SCONST * pow(T, 3) * kappa_ * zeta3 /
           (pow(pc.c, 2) * pow(pc.h, 3));
  }

private:
  Real kappa_; // cgs value of absorption coefficient (cm^-1)
};

class TophatOpacity : public Opacity {
public:
  TophatOpacity(const Real C, const Real numin, const Real numax)
      : C_(C), numin_(numin), numax_(numax) {}

  KOKKOS_INLINE_FUNCTION
  Real Getjnu(const Real rho, const Real T, const Real Ye, const NeutrinoSpecies s,
              const Real nu) override {
    if (nu > numin_ && nu < numax_) {
      return C_ * Getyf(Ye, s) / (4. * M_PI);
    } else {
      return 0.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJnu(const Real rho, const Real T, const Real Ye, const NeutrinoSpecies s,
              const Real nu) override {
    if (nu > numin_ && nu < numax_) {
      return C_ * Getyf(Ye, s);
    } else {
      return 0.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJ(const Real rho, const Real T, const Real Ye,
            const NeutrinoSpecies s) override {
    Real Bc = C_ * (numax_ - numin_);
    Real J = Bc * Getyf(Ye, s);
    return J;
  }

  KOKKOS_INLINE_FUNCTION
  Real GetJye(const Real rho, const Real T, const Real Ye,
              const NeutrinoSpecies s) override {
    Real Ac = pc.mp / (pc.h * rho) * C_ * log(numax_ / numin_);
    return rho * Ac * Getyf(Ye, s);
  }

private:
  Real C_;
  Real numin_;
  Real numax_;

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
};

} // namespace radiation

#endif // RADIATION_OPACITY_HPP_
