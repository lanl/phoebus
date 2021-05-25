from numpy import *
from geometry import *
from problem import *
from util import *

class Var(Enum):
  RHO = 0
  UG = 1
  V1 = 2
  V2 = 3

class Point:
  def __init__(self, loc, i, j, rho, ug, v1, v2):
    self.loc = loc
    self.i = i
    self.j = j
    self.rho = rho
    self.ug = ug
    self.v1 = v1
    self.v2 = v2

  def get_vcon(self):
    vcon = zeros(4)
    vcon[1] = self.v1
    vcon[2] = self.v2
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
    return (gam - 1.)*self.ug

  def get_cons(self):
    cons = zeros(4)
    vcov = self.get_vcov()
    Gamma = self.get_Gamma()
    rho = self.rho
    ug = self.ug
    P = self.get_pressure()
    h = 1. + ug/rho + P/rho
    D = rho*Gamma

    print("vcov: ", vcov)

    cons[0] = Gamma*rho
    cons[1] = rho*h*Gamma**2 - P - D
    cons[2] = rho*h*Gamma**2*vcov[1]
    cons[3] = rho*h*Gamma**2*vcov[2]

    return cons

  def get_h(self):
    return 1. + self.ug/self.rho + self.get_pressure()/self.rho

  def get_S_D(self):
    S = zeros(4)
    vcov = self.get_vcov()
    h = self.get_h()
    Gamma = self.get_Gamma()
    for mu in range(1,3):
      S[mu] = self.rho*h*Gamma**2*vcov[mu]
    return S

  def get_S_U(self):
    S = zeros(4)
    vcon = self.get_vcon()
    h = self.get_h()
    Gamma = self.get_Gamma()
    for mu in range(1,3):
      S[mu] = self.rho*h*Gamma**2*vcon[mu]
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
    D = Gamma*self.rho
    h = self.get_h()
    P = self.get_pressure()
    tau = self.rho*h*Gamma**2 - P - D

    flux[0] = (alpha*vcon[d] - beta[d])*D
    flux[1] = alpha*(S_U[d] - vcon[d]*D) - beta[d]*tau
    flux[2] = alpha*W_UD[d,1] - beta[d]*S_D[1]
    flux[3] = alpha*W_UD[d,2] - beta[d]*S_D[2]

    return flux


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

  def cons(self, loc, i, j):
    vcon = array([0, self.prim_v1[i,j], self.prim_v2[i,j], 0])
    vcov = zeros(4)
    gcov = self.geom.gcov(loc, i, j)
    for mu in range(1,4):
      for nu in range(1,4):
        vcov[mu] += gcov[mu,nu]*vcon[nu]
    vsq = 0
    for mu in range(1,4):
      vsq += vcon[mu]*vcov[mu]
    Gamma = 1./sqrt(1. - vsq)

    rho = self.prim_rho[i,j]
    ug = self.prim_ug[i,j]
    P = self.pressure(loc, i, j)
    h = 1. + ug/rho + P/rho

    self.cons_rho[i,j] = Gamma*rho
    self.cons_ug[i,j] = rho*h*Gamma**2 - P - self.cons_rho[i,j]
    self.cons_v1[i,j] = rho*h*Gamma**2*vcov[1]
    self.cons_v2[i,j] = rho*h*Gamma**2*vcov[2]


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

  def print(self, loc, i, j):
    X = getX(loc, i, j)
    self.cons(loc, i, j)
    print("STATE FOR [%i" % i + ",%i]" % j)
    print("  X")
    for mu in range(4):
      print("  X[%i]" % mu + " = %e" % X[mu])
    print("")
    print("  PRIM")
    print("  rho = %e" % self.prim_rho[i,j])
    print("  ug  = %e" % self.prim_ug[i,j])
    print("  P   = %e" % self.pressure(loc,i,j))
    print("  v1  = %e" % self.prim_v1[i,j])
    print("  v2  = %e" % self.prim_v2[i,j])
    print("")
    print("  CONS")
    print("  rho = %e" % self.cons_rho[i,j])
    print("  ug  = %e" % self.cons_ug[i,j])
    print("  v1  = %e" % self.cons_v1[i,j])
    print("  v2  = %e" % self.cons_v2[i,j])
    print("")
    Tmunu = self.Tmunu(loc, i, j)
    print("  Tmunu")
    for mu in range(4):
      print("  %e " % Tmunu[mu,0] + "%e " % Tmunu[mu,1] + "%e " % Tmunu[mu,2] + "%e " % Tmunu[mu,3])

  def pressure(self, loc, i, j):
    return (gam - 1.)*self.prim_ug[i,j]

  def Tmunu(self, loc, i, j):
    Tmunu = zeros([4, 4])
    ucon = self.ucon(loc, i, j)
    rho = self.prim_rho[i,j]
    ug = self.prim_ug[i,j]
    P = self.pressure(loc, i, j)
    h = 1. + ug/rho + P/rho
    gcon = self.geom.gcon(loc, i, j)

    for mu in range(4):
      for nu in range(4):
        Tmunu[mu,nu] = rho*h*ucon[mu]*ucon[nu] + P*gcon[mu,nu]

    return Tmunu
