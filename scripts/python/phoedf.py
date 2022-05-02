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

from parthenon_tools import phdf

from phoebus_constants import *
from phoebus_eos import *
from phoebus_opacities import *

# ---------------------------------------------------------------------------- #
# Phoebus-specific derived class from Parthenon's phdf datafile reader
class phoedf(phdf.phdf):

  def __init__(self, filename):
    super().__init__(filename)

    print(self.Params['eos/type'])
    self.eos_type = self.Params['eos/type'].decode('ascii')

    if 'opacity/type' in self.Params:
      self.opacity_model = self.Params['opacity/type'].decode('ascii')

    self.LengthCodeToCGS = self.Params['phoebus/LengthCodeToCGS']
    self.TimeCodeToCGS = self.Params['phoebus/TimeCodeToCGS']
    self.MassCodeToCGS = self.Params['phoebus/MassCodeToCGS']
    self.TemperatureCodeToCGS = self.Params['phoebus/TemperatureCodeToCGS']
    self.EnergyCodeToCGS = self.MassCodeToCGS * cgs['CL']**2
    self.NumberDensityCodeToCGS = self.LengthCodeToCGS**(-3)
    self.EnergyDensityCodeToCGS = self.EnergyCodeToCGS * self.NumberDensityCodeToCGS
    self.MassDensityCodeToCGS = self.MassCodeToCGS * self.NumberDensityCodeToCGS

  def GetEOS(self):
    return eos_type_dict[self.eos_type](self.Params)

  def GetOpacity(self):
    return opacity_type_dict[self.opacity_model](self.Params)
