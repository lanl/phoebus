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

  def __init__(self, filename, geomfile=None):
    super().__init__(filename)

    try:
      self.eos_type = self.Params['eos/type'].decode()
    except (UnicodeDecodeError, AttributeError):
      self.eos_type = self.Params['eos/type']

    if 'opacity/type' in self.Params:
      self.opacity_model = self.Params['opacity/type']

    self.LengthCodeToCGS = self.Params['phoebus/LengthCodeToCGS']
    self.TimeCodeToCGS = self.Params['phoebus/TimeCodeToCGS']
    self.MassCodeToCGS = self.Params['phoebus/MassCodeToCGS']
    self.TemperatureCodeToCGS = self.Params['phoebus/TemperatureCodeToCGS']
    self.EnergyCodeToCGS = self.MassCodeToCGS * cgs['CL']**2
    self.NumberDensityCodeToCGS = self.LengthCodeToCGS**(-3)
    self.EnergyDensityCodeToCGS = self.EnergyCodeToCGS * self.NumberDensityCodeToCGS
    self.MassDensityCodeToCGS = self.MassCodeToCGS * self.NumberDensityCodeToCGS

    self.Nx1 = self.MeshBlockSize[0]
    self.Nx2 = self.MeshBlockSize[1]
    self.Nx3 = self.MeshBlockSize[2]

    self.ScalarField = [self.NumBlocks, self.Nx3, self.Nx2, self.Nx1]

# TODO(BRR) store output data only once with get/load functions

    if geomfile is None:
      self.flatgcov = self.Get("g.c.gcov", flatten=False)
    def flatten_indices(mu, nu):
      ind = [[0,1,3,6],[1,2,4,7],[3,4,5,8],[6,7,8,9]]
      return ind[mu][nu]
    self.gcov = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1, 4, 4])
    print(self.flatgcov.shape)
    for mu in range(4):
      for nu in range(4):
        self.gcov[:,:,:,:,mu,nu] = self.flatgcov[:,:,:,:,flatten_indices(mu,nu)]
    del(self.flatgcov)
    self.gcon = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1, 4, 4])
    for b in range(self.NumBlocks):
      for k in range(self.Nx3):
        for j in range(self.Nx2):
          for i in range(self.Nx1):
            self.gcon[b,k,j,i,:,:] = np.linalg.inv(self.gcov[b,k,j,i,:,:])
    self.alpha = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1])
    self.alpha[:,:,:,:] = np.sqrt(-1./self.gcon[:,:,:,:,0,0])
    self.betacon = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1, 3])
    self.betacon[:,:,:,:,:] = self.gcon[:,:,:,:,1:,0]*(self.alpha[:,:,:,:,np.newaxis]**2)
    self.gammacon = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1, 3, 3])
    for ii in range(3):
      for jj in range(3):
        self.gammacon[:,:,:,:,ii,jj] = self.gcon[:,:,:,:,ii+1,jj+1] + self.betacon[:,:,:,:,ii]*self.betacon[:,:,:,:,jj]/(self.alpha[:,:,:,:]**2)

    # Output quantities loaded upon request
    self.rho = None

    # Derived quantities evaluated upon request
    self.Gamma = None
    self.xi = None
    self.Pg = None
    self.Pm = None
    self.Pr = None
    self.tau = None
    self.Tg = None

  def GetRho(self):
    if self.rho is not None:
      return self.rho

    self.rho = self.Get("p.density", flatten=False)

    return self.rho

  def GetEOS(self):
    return eos_type_dict[self.eos_type](self.Params)

  def GetOpacity(self, constants='cgs'):
    return opacity_type_dict[self.opacity_model](self.Params, constants)

  # Optical depth per zone
  def GetTau(self):
    if self.tau is not None:
      return self.tau

    self.tau = np.zeros(self.ScalarField)

    kappaH = self.Get("r.i.kappaH", flatten=False)

    for b in range(self.NumBlocks):
      dX1 = (self.BlockBounds[b][1] - self.BlockBounds[b][0])/self.Nx1
      dX2 = (self.BlockBounds[b][3] - self.BlockBounds[b][2])/self.Nx1
      self.tau[b,:,:,:] = kappaH[b,:,:,:,]*np.sqrt((dX1*self.gcov[b,:,:,:,1,1])**2 + (dX2*self.gcov[b,:,:,:,2,2])**2)

    self.tau = np.clip(self.tau, 1.e-20, 1.e20)

    return self.tau

  def GetPg(self):
    if self.Pg is not None:
      return self.Pg

    self.Pg = np.zeros(self.ScalarField)

    eos = self.GetEOS()
    rho = self.GetRho()
    u = self.Get("p.energy", flatten=False)
    Ye = np.zeros(self.ScalarField)
    self.Pg[:,:,:,:] = eos.P_from_rho_u_Ye(rho[:,:,:,:], u[:,:,:,:], Ye[:,:,:,:])

    self.Pg = np.clip(self.Pg, 1.e-20, 1.e20)

    return self.Pg

  def GetTg(self):
    if self.Tg is not None:
      return self.Tg

    self.Tg = np.zeros(self.ScalarField)

    eos = self.GetEOS()
    rho = self.GetRho()
    u = self.Get("p.energy", flatten=False)
    Ye = np.zeros(self.ScalarField)
    self.Tg[:,:,:,:] = eos.T_from_rho_u_Ye(rho[:,:,:,:], u[:,:,:,:], Ye[:,:,:,:])
    self.Tg *= self.TemperatureCodeToCGS

    self.Tg = np.clip(self.Tg, 1., 1.e20)

    return self.Tg


  def GetPm(self):
    if self.Pm is not None:
      return self.Pm

    self.Pm = np.zeros(self.ScalarField)

    Gamma = self.GetGamma()

    bcon0 = np.zeros(self.ScalarField)
    Bsq = np.zeros(self.ScalarField)
    bsq = np.zeros(self.ScalarField)
    Bcon = self.Get("p.bfield", flatten=False)
    vpcon = self.Get("p.velocity", flatten=False)
    for ii in range(3):
      for jj in range(3):
        bcon0[:,:,:,:] += self.gcov[:,:,:,:,ii+1,jj+1]*Bcon[:,:,:,:,ii]*vpcon[:,:,:,:,jj]
        Bsq[:,:,:,:] += self.gcov[:,:,:,:,ii+1,jj+1]*Bcon[:,:,:,:,ii]*Bcon[:,:,:,:,jj]
    bsq[:,:,:,:] = (Bsq[:,:,:,:] + (self.alpha[:,:,:,:]*bcon0[:,:,:,:])**2)/(Gamma[:,:,:,:]**2)

    self.Pm = bsq / 2.
    self.Pm = np.clip(self.Pm, 1.e-20, 1.e20)

    return self.Pm

  def GetPr(self):
    if self.Pr is not None:
      return self.Pr

    self.Pr = np.zeros(self.ScalarField)

    J = self.Get("r.p.J", flatten=False)
    self.Pr[:,:,:,:] = 1./3.*J[:,:,:,:]

    self.Pr = np.clip(self.Pr, 1.e-20, 1.e20)

    return self.Pr

  def GetGamma(self):
    if self.Gamma is not None:
      return self.Gamma

    self.Gamma = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1])

    vpcon = self.Get("p.velocity", flatten=False)
    for ii in range(3):
      for jj in range(3):
        self.Gamma[:,:,:,:] += self.gcov[:,:,:,:,ii+1,jj+1] * vpcon[:,:,:,:,ii] * vpcon[:,:,:,:,jj]
    self.Gamma = np.sqrt(1. + self.Gamma)

    return self.Gamma

  def GetXi(self):
    if self.xi is not None:
      return self.xi

    self.xi = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1])
    Hcov = self.Get("r.p.H", flatten=False)
    vcon = self.Get("p.velocity", flatten=False)
    Gamma = self.GetGamma()
    vcon[:,:,:,:,:] /= Gamma[:,:,:,:,np.newaxis]
    vdH = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1])
    for ii in range(3):
      vdH += vcon[:,:,:,:,ii]*Hcov[:,:,:,:,ii]
      for jj in range(3):
        self.xi[:,:,:,:] += self.gammacon[:,:,:,:,ii,jj]*Hcov[:,:,:,:,ii]*Hcov[:,:,:,:,jj] 
    self.xi[:,:,:,:] -= vdH*vdH
    self.xi = np.sqrt(self.xi)
    
    print(self.xi.max())

    self.xi = np.clip(self.xi, 1.e-10, 1.)

    print(self.xi.max())

    return self.xi



