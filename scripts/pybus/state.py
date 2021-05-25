from numpy import *
from geometry import *

class State:
  def __init__(self, nx1, nx2, geom):
    self.prim_rho = zeros([nx1, nx2])
    self.prim_ug = zeros([nx1, nx2])
    self.prim_v1 = zeros([nx1, nx2])
    self.prim_v2 = zeros([nx1, nx2])

    self.cons_rho = zeros([nx1, nx2])
    self.cons_ug = zeros([nx1, nx2])
    self.cons_v1 = zeros([nx1, nx2])
    self.cons_v2 = zeros([nx1, nx2])

    self.geom = geom

  def ucon(self, loc, i, j):
    ucon = zeros(4)
    vcon = array([0, self.prim_v1[i,j], self.prim_v2[i,j], 0])
    vcov = zeros(4)
    gcov = self.geom.gcov(loc, i, j)
    alpha = self.geom.lapse(loc, i, j)
    beta = self.geom.shift(loc, i,j)
    for mu in range(1,4):
      for nu in range(1,4):
        vcov[mu] += gcov[mu,nu]*vcon[nu]
    vsq = 0
    for mu in range(1,4):
      vsq += vcon[mu]*vcov[mu]
    Gamma = 1./sqrt(1. - vsq)
    ucon[0] = alpha*Gamma
    for mu in range(1,4):
      ucon[mu] = Gamma*(vcon[mu] - beta[mu]/alpha)

    return ucon

  def Tmunu(self, loc, i, j):
    Tmunu = zeros([4, 4])
    ucon = self.ucon(loc, i, j)
