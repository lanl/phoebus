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

import sys
sys.path.insert(0, '../../external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/')
import phdf
#from parthenon_tools import phdf

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
    self.ThreeVectorField = [self.NumBlocks, 3, self.Nx3, self.Nx2, self.Nx1]
    self.FourVectorField = [self.NumBlocks, 4, self.Nx3, self.Nx2, self.Nx1]
    self.ThreeTensorField = [self.NumBlocks, 3, 3, self.Nx3, self.Nx2, self.Nx1]
    self.FourTensorField = [self.NumBlocks, 4, 4, self.Nx3, self.Nx2, self.Nx1]

    self.RadiationActive = self.Params['radiation/active']
    if (self.RadiationActive):
      self.NumSpecies = self.Params['radiation/num_species']

# TODO(BRR) store output data only once with get/load functions

    if geomfile is None:
      self.flatgcov = self.Get("g.c.gcov", flatten=False)
    def flatten_indices(mu, nu):
      ind = [[0,1,3,6],[1,2,4,7],[3,4,5,8],[6,7,8,9]]
      return ind[mu][nu]
    if self.flatgcov is None:
      if self.Params['geometry/geometry_name'] == "Minkowski":
        self.flatgcov = np.zeros([self.NumBlocks, self.Nx3, self.Nx2, self.Nx1, 10])
        self.flatgcov[:,flatten_indices(0, 0),:,:,:] = -1.
        self.flatgcov[:,flatten_indices(1, 1),:,:,:] = 1.
        self.flatgcov[:,flatten_indices(2, 2),:,:,:] = 1.
        self.flatgcov[:,flatten_indices(3, 3),:,:,:] = 1.
    self.gcov = np.zeros([self.NumBlocks, 4, 4, self.Nx3, self.Nx2, self.Nx1])
    for mu in range(4):
      for nu in range(4):
        self.gcov[:,mu,nu,:,:,:] = self.flatgcov[:,flatten_indices(mu,nu),:,4:-4,4:-4]
    del(self.flatgcov)
    self.gcon = np.zeros([self.NumBlocks, 4, 4, self.Nx3, self.Nx2, self.Nx1])
    for b in range(self.NumBlocks):
      for k in range(self.Nx3):
        for j in range(self.Nx2):
          for i in range(self.Nx1):
            self.gcon[b,:,:,k,j,i] = np.linalg.inv(self.gcov[b,:,:,k,j,i])
    self.alpha = np.zeros(self.ScalarField)
    self.alpha[:,:,:,:] = np.sqrt(-1./self.gcon[:,0,0,:,:,:])
    self.betacon = np.zeros(self.ThreeVectorField)
    self.betacon[:,:,:,:,:] = self.gcon[:,1:,0,:,:,:]*(self.alpha[:,np.newaxis,:,:,:]**2)
    self.gammacon = np.zeros(self.ThreeTensorField)
    for ii in range(3):
      for jj in range(3):
        self.gammacon[:,ii,jj,:,:,:] = self.gcon[:,ii+1,jj+1,:,:,:] + \
                                       self.betacon[:,ii,:,:,:] * self.betacon[:,jj,:,:,:] / \
                                       (self.alpha[:,:,:,:]**2)
    self.gdet = np.zeros(self.ScalarField)
    for b in range(self.NumBlocks):
      for k in range(self.Nx3):
        for j in range(self.Nx2):
          for i in range(self.Nx1):
            self.gdet[b,k,j,i] = np.sqrt(-np.linalg.det(self.gcov[b,:,:,k,j,i]))

    # Output quantities loaded upon request
    self.rho = None
    self.ug = None
    self.vpcon = None
    self.Bcon = None

    # Derived quantities evaluated upon request
    self.Gamma = None
    self.xi = None
    self.Pg = None
    self.Pm = None
    self.Pr = None
    self.tau = None
    self.Tg = None
    self.ucon = None
    self.bcon = None
    self.J = None
    self.Hcov = None
    self.E = None
    self.F = None
    self.P = None

  def GetRho(self):
    if self.rho is None:
      self.rho = self.Get("p.density", flatten=False)
      assert self.rho is not None

    return self.rho

  def GetUg(self):
    if self.ug is None:
      self.ug = self.Get("p.energy", flatten=False)
      assert self.ug is not None

    return self.ug

  def GetBcon(self):
    if self.Bcon is None:
      if self.Params['fluid/mhd']:
        self.Bcon = self.Get("p.bfield", flatten=False)
        assert self.Bcon is not None
      else:
        self.Bcon = np.zeros(self.ThreeVectorField)

    return self.Bcon

  def GetJ(self):
    assert self.RadiationActive
    if self.J is None:
      self.J = self.Get("r.p.J", flatten=False)
      assert self.J is not None

    return self.J

  def GetHcov(self):
    if self.Hcov is None:
      self.Hcov = self.Get("r.p.H", flatten=False) * self.GetJ()[:,np.newaxis,:,:,:,:]
      assert self.Hcov is not None

    return self.Hcov

  def Getbcon(self):
    if self.bcon is None:
      self.bcon = np.zeros(self.FourVectorField)

      Bcon = self.GetBcon()
      Gamma = self.GetGamma()
      vcon = self.GetVpCon() / Gamma[:,np.newaxis,:,:,:]
      ucon = self.Getucon()
      gcov = self.gcov
      alpha = self.alpha

      for ii in range(3):
        for jj in range(3):
          self.bcon[:,0,:,:,:] += Gamma * (Bcon[:,ii,:,:,:] * gcov[:,ii+1,jj+1,:,:,:] * \
                                           vcon[:,jj,:,:,:]) / alpha

      for ii in range(3):
        self.bcon[:,ii+1,:,:,:] = (Bcon[:,ii,:,:,:] + alpha * self.bcon[:,0,:,:,:] * \
                                   ucon[:,ii+1,:,:,:]) / Gamma

    return self.bcon

  def GetEOS(self):
    return eos_type_dict[self.eos_type](self.Params)

  def GetOpacity(self, constants='cgs'):
    return opacity_type_dict[self.opacity_model](self.Params, constants)

  # Optical depth per zone
  def GetTau(self):
    if self.tau is None:
      self.tau = np.zeros(self.ScalarField)

      kappaH = self.Get("r.i.kappaH", flatten=False)
      assert kappaH is not None

      for b in range(self.NumBlocks):
        dX1 = (self.BlockBounds[b][1] - self.BlockBounds[b][0])/self.Nx1
        dX2 = (self.BlockBounds[b][3] - self.BlockBounds[b][2])/self.Nx2
        dX3 = (self.BlockBounds[b][5] - self.BlockBounds[b][4])/self.Nx3
        if self.Nx3 > 1:
          dX = np.sqrt((dX1*np.sqrt(self.gcov[b,1,1,:,:,:]))**2 +
                       (dX2*np.sqrt(self.gcov[b,2,2,:,:,:]))**2 +
                       (dX3*np.sqrt(self.gcov[b,3,3,:,:,:]))**2)
        elif self.Nx2 > 1:
          dX = np.sqrt((dX1*np.sqrt(self.gcov[b,1,1,:,:,:]))**2 +
                       (dX2*np.sqrt(self.gcov[b,2,2,:,:,:]))**2)
        else:
          dX = dX1*np.sqrt(self.gcov[b,1,1,:,:,:])

        self.tau[b,:,:,:] = kappaH[b,:,:,:,]*dX

      self.tau = np.clip(self.tau, 1.e-100, 1.e100)

    return self.tau

  def GetPg(self):
    if self.Pg is None:
      self.Pg = np.zeros(self.ScalarField)

      eos = self.GetEOS()
      rho = self.GetRho()
      ug = self.GetUg()
      Ye = np.zeros(self.ScalarField)
      self.Pg[:,:,:,:] = eos.P_from_rho_u_Ye(rho[:,:,:,:]*self.MassDensityCodeToCGS,
        ug[:,:,:,:]*self.EnergyDensityCodeToCGS, Ye[:,:,:,:]) / self.EnergyDensityCodeToCGS

      self.Pg = np.clip(self.Pg, 1.e-100, 1.e100)

    return self.Pg

  def GetTg(self):
    if self.Tg is None:
      self.Tg = np.zeros(self.ScalarField)

      eos = self.GetEOS()
      rho = self.GetRho()
      u = self.GetUg()
      # TODO(BRR) actually get Ye
      Ye = np.zeros(self.ScalarField)
      self.Tg[:,:,:,:] = eos.T_from_rho_u_Ye(rho[:,:,:,:]*self.MassDensityCodeToCGS,
        u[:,:,:,:]*self.EnergyDensityCodeToCGS, Ye[:,:,:,:]) / self.TemperatureCodeToCGS

      self.Tg = np.clip(self.Tg, 1.e-100, 1.e100)

    return self.Tg

  def GetPm(self):
    if self.Pm is None:
      self.Pm = np.zeros(self.ScalarField)

      Gamma = self.GetGamma()
      bcon = self.Getbcon()
      gcov = self.gcov

      bsq = np.zeros(self.ScalarField)
      for mu in range(4):
        for nu in range(4):
          bsq[:,:,:,:] += gcov[:,mu,nu,:,:,:]*bcon[:,mu,:,:,:]*bcon[:,nu,:,:,:]

      self.Pm = bsq / 2.
      self.Pm = np.clip(self.Pm, 1.e-100, 1.e100)

    return self.Pm

  def GetPr(self):
    if self.Pr is None:
      self.Pr = np.zeros(self.ScalarField)

      J = self.GetJ()
      for ispec in range(self.NumSpecies):
        self.Pr[:,:,:,:] += 1. / 3. * J[:,ispec,:,:,:]

      self.Pr = np.clip(self.Pr, 1.e-100, 1.e100)

    return self.Pr

  def GetGamma(self):
    if self.Gamma is None:
      self.Gamma = np.zeros(self.ScalarField)

      vpcon = self.GetVpCon()
      for ii in range(3):
        for jj in range(3):
          self.Gamma[:,:,:,:] += self.gcov[:,ii+1,jj+1,:,:,:] * vpcon[:,ii,:,:,:] * vpcon[:,jj,:,:,:]
      self.Gamma = np.sqrt(1. + self.Gamma)

    return self.Gamma

  def GetVpCon(self):
    if self.vpcon is None:
      self.vpcon = np.clip(self.Get("p.velocity", flatten=False), -1.e100, 1.e100)
      assert self.vpcon is not None

    return self.vpcon

  def Getucon(self):
    if self.ucon is None:
      self.ucon = np.zeros(self.FourVectorField)

      vpcon = self.GetVpCon()
      Gamma = self.GetGamma()

      self.ucon[:,0,:,:,:] = Gamma[:,:,:,:] / self.alpha[:,:,:,:]
      for ii in range(3):
        self.ucon[:,ii + 1,:,:,:] = vpcon[:,ii,:,:,:] - Gamma[:,:,:,:] * \
                                    self.betacon[:,ii,:,:,:] / self.alpha[:,:,:,:]

    return self.ucon

  def GetXi(self):
    if self.xi is None:
      self.xi = np.zeros([self.NumBlocks, self.NumSpecies, self.Nx3, self.Nx2, self.Nx1])

      Hcov = self.GetHcov() / self.GetJ()[:,np.newaxis,:,:,:]
      Gamma = self.GetGamma()
      vcon = self.GetVpCon() / Gamma[:,np.newaxis,:,:,:]
      for ispec in range(self.NumSpecies):
        vdH = np.zeros(self.ScalarField)
        for ii in range(3):
          vdH += vcon[:,ii,:,:,:] * Hcov[:,ii,ispec,:,:,:]
          for jj in range(3):
            self.xi[:,ispec,:,:,:] += self.gammacon[:,ii,jj,:,:,:] * Hcov[:,ii,ispec,:,:,:] \
                                      * Hcov[:,jj,ispec,:,:,:]
        self.xi[:,ispec,:,:,:] -= vdH*vdH
      self.xi = np.sqrt(self.xi)

      self.xi = np.clip(self.xi, 1.e-100, 1.)

    return self.xi

  def GetE(self):
    if self.E is None:
      self.E = np.zeros([self.NumBlocks, self.NumSpecies, self.Nx3, self.Nx2, self.Nx1])

      Gamma = self.GetGamma()
      vcon = self.GetVpCon() / Gamma[:,np.newaxis,:,:,:]
      J = self.GetJ()
      Hcov = self.GetHcov()

      self.E[:,:,:,:,:] = (4. * Gamma[:,np.newaxis,:,:,:]**2 / 3. - 1. / 3.)*J[:,:,:,:,:]

      for ii in range(3):
        self.E[:,:,:,:,:] += 2. * Gamma[:,np.newaxis,:,:,:] * vcon[:,ii,np.newaxis,:,:,:] * \
                             Hcov[:,ii,:,:,:,:]

      # TODO(BRR) tilPi component

    return self.E

  def GetF(self):
    if self.F is None:
      self.F = np.zeros([self.NumBlocks, 3, self.NumSpecies, self.Nx3, self.Nx2, self.Nx1])

      Gamma = self.GetGamma()
      vcon = self.GetVpCon() / Gamma[:,np.newaxis,:,:,:]
      J = self.GetJ()
      Hcov = self.GetHcov()
      gammacon = self.gammacon

      for idir in range(3):
        for ispec in range(self.NumSpecies):
          self.F[:,idir,ispec,:,:,:] = 4. * Gamma[:,:,:,:]**2 / 3. * \
                                vcon[:,idir,:,:,:] * J[:,ispec,:,:,:]
        for ii in range(3):
          self.F[:,idir,ispec,:,:,:] += Gamma[:,:,:,:] * vcon[:,idir,:,:,:] * \
                                 vcon[:,ii,:,:,:]*Hcov[:,ii,ispec,:,:,:]

        for ii in range(3):
          self.F[:,idir,ispec,:,:,:] += Gamma[:,:,:,:] * \
                                 gammacon[:,idir,ii,:,:,:] * Hcov[:,ii,ispec,:,:,:]

      # TODO(BRR) tilPi component

    return self.F

  def GetP(self):
    if self.P is None:
      self.P = np.zeros([self.NumBlocks, 3, 3, self.NumSpecies, self.Nx3, self.Nx2, self.Nx1])

      Gamma = self.GetGamma()
      vcon = self.GetVpCon() / Gamma[:,np.newaxis,:,:,:]
      J = self.GetJ()
      Hcov = self.GetHcov()
      gammacon = self.gammacon

      for ii in range(3):
        for jj in range(3):
          for ispec in range(self.NumSpecies):
            self.P[:,ii,jj,ispec,:,:,:] = (4. / 3. * Gamma[:,:,:,:]**2 * \
                                           vcon[:,ii,:,:,:]*vcon[:,jj,:,:,:] + \
                                           1. / 3. * gammacon[:,ii,jj,:,:,:]) * \
                                           J[:,ispec,:,:,:]

          for kk in range(3):
            self.P[:,ii,jj,ispec,:,:,:] += Gamma[:,:,:,:] * vcon[:,ii,:,:,:] * \
                                           gammacon[:,jj,kk,:,:,:] * Hcov[:,kk,ispec,:,:,:]

          for kk in range(3):
            self.P[:,ii,jj,ispec,:,:,:] += Gamma[:,:,:,:] * vcon[:,jj,:,:,:] * \
                                           gammacon[:,ii,kk,:,:,:] * Hcov[:,kk,ispec,:,:,:]

      # TODO(BRR) tilPi component

    return self.P

  def GetTmunu_concon(self, b, k, j, i):
    Tmunu = np.zeros([4, 4])

    rho = self.GetRho()[b,k,j,i]
    ug = self.GetUg()[b,k,j,i]
    Pg = self.GetPg()[b,k,j,i]
    ucon = self.Getucon()[b,:,k,j,i]
    bcon = self.Getbcon()[b,:,k,j,i]

    gcov = self.gcov[b,:,:,k,j,i]
    gcon = self.gcon[b,:,:,k,j,i]

    bsq = 0.
    for mu in range(4):
      for nu in range(4):
        bsq += gcov[mu,nu] * bcon[mu] * bcon[nu]

    Ptot = ug + bsq / 2.

    Tmunu[:,:] = (rho + ug + Ptot) * ucon[:,np.newaxis] * ucon[np.newaxis,:]
    Tmunu[:,:] += Ptot * gcon[:,:]
    Tmunu[:,:] -= bcon[:,np.newaxis] * bcon[np.newaxis,:]

    return Tmunu

  def GetTmunu_concov(self, b, k, j, i):
    Tmunu_concon = self.GetTmunu_concon(b, k, j, i)

    gcov = self.gcov[b,:,:,k,j,i]

    Tmunu_concov = np.zeros([4, 4])
    for mu in range(4):
      for nu in range(4):
        for lam in range(4):
          Tmunu_concov[mu, nu] += Tmunu_concon[mu, lam] * gcov[lam, nu]

    return Tmunu_concov

  def GetRmunu_concon(self, b, k, j, i):
    Rmunu = np.zeros([4, 4, self.NumSpecies])

    ncon = np.zeros(4)
    ncon[0] = 1. / self.alpha[b,k,j,i]
    ncon[1:] = -self.betacon[b,:,k,j,i] / self.alpha[b,k,j,i]

    ncov = np.zeros(4)
    ncov[0] = -self.alpha[b,k,j,i]

    E = self.GetE()
    F = self.GetF()
    P = self.GetP()

    Rmunu[:,:,:] = E[b,np.newaxis,np.newaxis,:,k,j,i] * ncon[:,np.newaxis,np.newaxis] * \
                   ncon[np.newaxis,:,np.newaxis]
    Rmunu[:,1:,:] += ncon[:,np.newaxis,np.newaxis] * F[b,np.newaxis,:,:,k,j,i]
    Rmunu[1:,:,:] += ncon[np.newaxis,:,np.newaxis] * F[b,:,np.newaxis,:,k,j,i]
    Rmunu[1:,1:,:] += P[b,:,:,:,k,j,i]

    return Rmunu

  def GetRmunu_concov(self, b, k, j, i):
    Rmunu_concon = self.GetRmunu_concon(b, k, j, i)

    gcov = self.gcov[b,:,:,k,j,i]

    Rmunu_concov = np.zeros([4, 4, self.NumSpecies])
    for mu in range(4):
      for nu in range(4):
        for lam in range(4):
          Rmunu_concov[mu, nu, :] += Rmunu_concon[mu, lam, :] * gcov[lam, nu, np.newaxis]

    return Rmunu_concov

  def GetMdotEddington(self, eff = 0.1):
    Mbh = self.LengthCodeToCGS * cgs['CL']**2 / cgs['GNEWT']
    return 1.4e18 * Mbh / cgs['MSOLAR'] # Nominal eff = 0.1

