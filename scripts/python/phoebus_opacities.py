# © 2022. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.  Department
# of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the
# U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.

import numpy as np
from phoebus_constants import cgs, scalefree

# ---------------------------------------------------------------------------- #
# -- Thermal distribution function for neutrinos
class ThermalDistribution:
  def __init__(self, constants=cgs, nspecies=3):
    self.constants = constants
    self.h = self.constants['HPL']
    self.kb = self.constants['KBOL']
    self.c = self.constants['CL']
    self.nspecies = nspecies

  def Bnu(self, T, nu):
    x = self.h*nu/(self.kb*T)
    return self.nspecies * (2. * self.h * nu**3)/self.c**2/(np.exp(x) + 1.)

  def T_from_J(self, J):
    return pow(15.*self.c**3*self.h**3*J/(7.*self.kb**4*np.pi**5*self.nspecies),1./4.)

  def J_from_T(self, T):
    return 7.*self.kb**4*np.pi**5*self.nspecies*T**4/(15.*self.c**3*self.h**3)

# ---------------------------------------------------------------------------- #
# -- Base class containing required data and functions
class BaseOpacity:
  def __init__(self, constants='cgs'):
    self.model = None
    constants_dict = {'cgs' : cgs, 'scalefree' : scalefree}
    self.dist = ThermalDistribution(constants_dict[constants])

  def jnu(self, rho, T, Ye, nu):
    raise NotImplementedError()
  def alphanu(self, rho, T, Ye, nu):
    raise NotImplementedError()

# -- Gray opacity
class GrayOpacity(BaseOpacity):
  def __init__(self, params, constants='cgs'):
    super().__init__(constants)
    self.model = "gray"
    self.kappa = params['opacity/gray_kappa']

  def alphanu(self, rho, T, Ye, nu):
    return rho*self.kappa

  def jnu(self, rho, T, Ye, nu):
    return self.alphanu(rho, T, Ye, nu) * self.dist.Bnu(T, nu)


# ---------------------------------------------------------------------------- #
# -- Dictionary of opacity types
opacity_type_dict = {'gray' : GrayOpacity}
