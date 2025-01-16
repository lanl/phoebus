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
from phoebus_constants import cgs

# ---------------------------------------------------------------------------- #
# -- Base class containing required data and functions
class BaseEOS:
    def __init__(self):
        self.model = None

    def T_from_rho_u_Ye(self, rho, u, Ye):
        raise NotImplementedError()

    def u_from_rho_T_Ye(self, rho, T, Ye):
        raise NotImplementedError()

    def P_from_rho_u_Ye(self, rho, u, Ye):
        raise NotImplementedError()


# -- Ideal gas EOS
class IdealEOS(BaseEOS):
    def __init__(self, params):
        super().__init__()
        self.type = "IdealGas"
        self.Cv = params["eos/Cv"]
        self.gm1 = params["eos/gm1"]

    def T_from_rho_u_Ye(self, rho, u, Ye):
        return u / (rho * self.Cv)

    def u_from_rho_T_Ye(self, rho, T, Ye):
        return rho * self.Cv * T

    def P_from_rho_u_Ye(self, rho, u, Ye):
        return self.gm1 * u


# ---------------------------------------------------------------------------- #
# -- Dictionary of EOS types
eos_type_dict = {"IdealGas": IdealEOS}
