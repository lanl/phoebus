from numpy import *
from params import *
from geometry import *
from util import *

from enum import IntEnum

class Var(IntEnum):
  RHO = 0
  UG = 1
  V1 = 2
  V2 = 3
  SIZE = 4

class State:
  prim = zeros([N1TOT, N2TOT, Var.SIZE])
  cons = zeros([N1TOT, N2TOT, Var.SIZE])
  ql = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])
  qr = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])
  Ul = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])
  Ur = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])
  Fl = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])
  Fr = zeros([N1TOT, N2TOT, Var.SIZE, Dir.SIZE])

  def get_pressure(self, i, j):
    return (gam - 1.)*self.prim[i,j,Var.UG]

  def get_h(self, i, j):
    rho = self.prim[i,j,Var.RHO]
    ug = self.prim[i,j,Var.UG]
    P = self.get_pressure(i,j)
    return 1. + ug/rho + P/rho

  def PrimToCons(self):
    for i in range(N1TOT):
      for j in range(N2TOT):
        vcov = self.get_vcov()
        Gamma = self.get_Gamma()
        rho = self.prim[i,j,Var.RHO]
        ug = self.prim[i,j,Var.UG]
        P = self.get_pressure(i,j)
        h = self.get_h(i,j)
        D = rho*Gamma

        self.cons[i,j,Var.RHO] = Gamma*rho
        self.cons[i,j,Var.UG] = rho*h*Gamma**2 - P - D
        self.cons[i,j,Var.V1] = rho*h*Gamma**2*vcov[1]
        self.cons[i,j,Var.V2] = rho*h*Gamma**2*vcov[2]

  def get_vcon(self, i, j):
    vcon = zeros(4)
    vcon[1] = self.prim[i,j,Var.V1]
    vcon[2] = self.prim[i,j,Var.V2]
    return vcon

  def get_vcov(self, i, j):
    vcov = zeros(4)
    vcon = self.get_vcon(i,j)
    gcov = geom.gcov(self.loc, self.i, self.j)
    for mu in range(1,4):
      for nu in range(1,4):
        vcov[mu] += gcov[mu,nu]*vcon[nu]
    return vcov

  def get_Gamma(self):
    vcon = self.get_vcon()
    vcov = self.get_vcov()
    vsq = 0
    for mu in range(1,4):
      vsq += vcon[mu]*vcov[mu]
    Gamma = 1./sqrt(1. - vsq)
    return Gamma

class Point:
  def __init__(self, loc, i, j, rho, ug, v1, v2):
    self.loc = loc
    self.i = i
    self.j = j

    self.prim = zeros(4)
    self.prim[Var.RHO] = rho
    self.prim[Var.UG] = ug
    self.prim[Var.V1] = v1
    self.prim[Var.V2] = v2

  def get_vcon(self):
    vcon = zeros(4)
    vcon[1] = self.prim[Var.V1]
    vcon[2] = self.prim[Var.V2]
    return vcon

  def get_vcov(self):
    vcov = zeros(4)
    vcon = self.get_vcon()
    gcov = geom.gcov(self.loc, self.i, self.j)
    for mu in range(1,4):
      for nu in range(1,4):
        vcov[mu] += gcov[mu,nu]*vcon[nu]
    return vcov

  def get_Gamma(self):
    vcon = self.get_vcon()
    vcov = self.get_vcov()
    vsq = 0
    for mu in range(1,4):
      vsq += vcon[mu]*vcov[mu]
    Gamma = 1./sqrt(1. - vsq)
    return Gamma

  def get_ucon(self):
    ucon = zeros(4)
    alpha = geom.lapse(loc, i, j)
    beta = geom.shift(loc, i,j)
    vcon = self.get_vcon()
    Gamma = self.get_Gamma()

    ucon[0] = alpha*Gamma
    for mu in range(1,4):
      ucon[mu] = Gamma*(vcon[mu] - beta[mu]/alpha)
    return ucon

  def get_ucov(self):
    ucov = zeros(4)
    ucon = self.get_ucon()
    gcov = geom.gcov(loc, i, j)
    for mu in range(4):
      for nu in range(4):
        ucov[mu] += gcov[mu,nu]*ucon[nu]
    return ucov

  def get_pressure(self):
    return (gam - 1.)*self.prim[Var.UG]

  def get_cons(self):
    cons = zeros(4)
    vcov = self.get_vcov()
    Gamma = self.get_Gamma()
    rho = self.prim[Var.RHO]
    ug = self.prim[Var.UG]
    P = self.get_pressure()
    h = self.get_h()
    D = rho*Gamma

    cons[Var.RHO] = Gamma*rho
    cons[Var.UG] = rho*h*Gamma**2 - P - D
    cons[Var.V1] = rho*h*Gamma**2*vcov[1]
    cons[Var.V2] = rho*h*Gamma**2*vcov[2]

    return cons

  def get_h(self):
    rho = self.prim[Var.RHO]
    ug = self.prim[Var.UG]
    return 1. + ug/rho + self.get_pressure()/rho

  def get_S_D(self):
    S = zeros(4)
    vcov = self.get_vcov()
    h = self.get_h()
    Gamma = self.get_Gamma()
    for mu in range(1,3):
      S[mu] = self.prim[Var.RHO]*h*Gamma**2*vcov[mu]
    return S

  def get_S_U(self):
    S = zeros(4)
    vcon = self.get_vcon()
    h = self.get_h()
    Gamma = self.get_Gamma()
    for mu in range(1,3):
      S[mu] = self.prim[Var.RHO]*h*Gamma**2*vcon[mu]
    return S

  def get_W_UU(self):
    W = zeros([4,4])
    S = self.get_S_U()
    vcon = self.get_vcon()
    P = self.get_pressure()
    gcon = geom.gcon(self.loc, self.i, self.j)

    for mu in range(1,4):
      for nu in range(1,4):
        W[mu,nu] = S[mu]*vcon[nu] + P*gcon[mu,nu]

    return W

  def get_W_UD(self):
    W = zeros([4,4])
    S = self.get_S_U()
    vcov = self.get_vcov()
    P = self.get_pressure()

    for mu in range(1,4):
      for nu in range(1,4):
        W[mu,nu] = S[mu]*vcov[nu] + P*delta(mu, nu)

    return W

  def get_F(self, d):
    flux = zeros(4)

    alpha = geom.lapse(self.loc, self.i, self.j)
    beta = geom.shift(self.loc, self.i, self.j)
    vcon = self.get_vcon()
    Gamma = self.get_Gamma()
    S_U = self.get_S_U()
    S_D = self.get_S_D()
    W_UD = self.get_W_UD()
    D = Gamma*self.prim[Var.RHO]
    h = self.get_h()
    P = self.get_pressure()
    tau = self.prim[Var.RHO]*h*Gamma**2 - P - D

    flux[Var.RHO] = (alpha*vcon[d] - beta[d])*D
    flux[Var.UG] = alpha*(S_U[d] - vcon[d]*D) - beta[d]*tau
    flux[Var.V1] = alpha*W_UD[d,1] - beta[d]*S_D[1]
    flux[Var.V2] = alpha*W_UD[d,2] - beta[d]*S_D[2]

    return flux
