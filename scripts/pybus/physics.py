from params import *
from state import *
from geometry import *

from scipy.optimize import newton

class Position:
  def __init__(self, i, j, loc):
    self.i = i
    self.j = j
    self.loc = loc

def get_pressure(pvec):
  return (gam - 1.)*pvec[Var.UG]

def get_enthalpy(pvec):
  rho = pvec[Var.RHO]
  ug = pvec[Var.UG]
  P = get_pressure(pvec)
  return 1. + ug/rho + P/rho

def get_vcon(pvec):
  vcon = zeros(4)
  vcon[1] = pvec[Var.V1]
  vcon[2] = pvec[Var.V2]
  return vcon

def get_vcov(pvec, pos):
  vcon = get_vcon(pvec)
  vcov = zeros(4)
  for mu in range(1,4):
    for nu in range(1,4):
      vcov[mu] += geom.gcov[pos.i,pos.j,pos.loc,mu,nu]*vcon[nu]
  return vcov

def get_Gamma(pvec, pos):
  vcon = get_vcon(pvec)
  vcov = get_vcov(pvec, pos)
  vsq = 0
  for mu in range(1,4):
    vsq += vcon[mu]*vcov[mu]
  Gamma = 1./sqrt(1. - vsq)
  return Gamma

def prim_to_flux(prim, flux, loc, d):
  for i in range(N1TOT):
    for j in range(N2TOT):
      pvec = prim[i,j,:]
      #fvec = flux[i,j,:,d]
      #pos = Position(i, j, loc)
      #vcon = get_vcon(pvec)
      #alpha = geom.lapse[pos.i,pos.j,pos.loc]
      #beta = geom.shift[pos.i,pos.j,pos.loc,:]
      #V = zeros(4)
      #for mu in range(1,4):
      #  V[mu] = alpha*vcon[mu] - beta[mu]

      #rho = pvec[Var.RHO]
      #ug = pvec[Var.UG]
      #P = get_pressure(pvec)
      #h = get_enthalpy(pvec)
      #D = rho*Gamma


      point = Point(loc, i, j, pvec[Var.RHO], pvec[Var.UG], pvec[Var.V1], pvec[Var.V2])
      flux[i,j,:,d] = point.get_F(d)
      #flux = point.get_F(d)
      #if i == NG and j == NG:
      #  print("prim and flux")
      #  print(pvec)
      #  print(flux)
      #  print("new flux")
      #  print(point.get_F(d))
      #  sys.exit()
      #Scon = point.get_S_U()
      #Wud = point.get_W_UD()

      #flux[Var.RHO] = V[d]*D
      #flux[Var.UG] = alpha*(Scon[d] - vcon[d]*D) - beta[d]*tau


  #FAIL("not implemented")


def prim_to_cons(prim, cons, loc, d=None):
  for i in range(N1TOT):
    for j in range(N2TOT):
      pvec = prim[i,j,:]
      # TODO(BRR) use Point class
      pos = Position(i, j, loc)
      vcov = get_vcov(pvec, pos)
      Gamma = get_Gamma(pvec, pos)

      rho = pvec[Var.RHO]
      ug = pvec[Var.UG]
      P = get_pressure(pvec)
      h = get_enthalpy(pvec)
      D = rho*Gamma

      if d == None:
        cons[i,j,Var.RHO] = Gamma*rho
        cons[i,j,Var.UG] = rho*h*Gamma**2 - P - D
        cons[i,j,Var.V1] = rho*h*Gamma**2*vcov[1]
        cons[i,j,Var.V2] = rho*h*Gamma**2*vcov[2]
      else:
        cons[i,j,Var.RHO,d] = Gamma*rho
        cons[i,j,Var.UG,d] = rho*h*Gamma**2 - P - D
        cons[i,j,Var.V1,d] = rho*h*Gamma**2*vcov[1]
        cons[i,j,Var.V2,d] = rho*h*Gamma**2*vcov[2]

def cons_to_prim(cons, prim):
  loc = Location.CENT
  for i in range(N1TOT):
    for j in range(N2TOT):
      pos = Position(i, j, loc)
      # Guess from previous primitives
      h = get_enthalpy(prim[i,j,:])
      w = prim[i,j,Var.RHO]*h
      Gamma = get_Gamma(prim[i,j,:], pos)
      xi_guess = Gamma**2*w

      cvec = cons[i,j,:]
      D = cvec[Var.RHO]
      Scov = array([0, cvec[Var.V1], cvec[Var.V2], 0])
      Ssq = 0
      tau = cvec[Var.UG]
      for mu in range(1,4):
        for nu in range(1,4):
          Ssq += geom.gcon[i,j,loc,mu,nu]*Scov[mu]*Scov[nu]

      def resid(xi):
        Gamma = sqrt(1./(1. - Ssq/xi**2))
        return xi - (gam - 1.)/gam*(xi/Gamma**2 - D/Gamma) - tau - D

      xi = newton(resid, xi_guess, tol=1.e-12)
      Gamma = sqrt(1./(1. - Ssq/xi**2))
      w = xi/Gamma**2
      rho = D/Gamma
      P = (gam - 1)/gam*(w - rho)
      ug = P/(gam -1.)

      vcov = array([0, Scov[1]/xi, Scov[2]/xi, 0])
      vcon = zeros(4)
      for mu in range(1,4):
        for nu in range(1,4):
          vcon[mu] += geom.gcon[i,j,loc,mu,nu]*vcov[nu]

      prim[i,j,Var.RHO] = rho
      prim[i,j,Var.UG] = ug
      prim[i,j,Var.V1] = vcon[1]
      prim[i,j,Var.V2] = vcon[2]

def get_vchar(point, d):
  i = point.i
  j = point.j
  loc = point.loc

  Acov = zeros(4)
  Acov[d] = 1.
  Acon = zeros(4)
  for mu in range(4):
    for nu in range(4):
      Acon[mu] += geom.gcon[i,j,loc,mu,nu]*Acov[nu]
  Bcov = zeros(4)
  Bcov[0] = 1
  Bcon = zeros(4)
  for mu in range(4):
    for nu in range(4):
      Bcon[mu] += geom.gcon[i,j,loc,mu,nu]*Bcov[nu]

  ucon = point.get_ucon()
  rho = point.prim[Var.RHO]
  u = point.prim[Var.UG]
  ef = rho + gam*u
  cs2 = gam*(gam - 1.)*u/ef

  Asq = 0
  Bsq = 0
  Au = 0
  Bu = 0
  AB = 0
  for mu in range(4):
    Asq += Acon[mu]*Acov[mu]
    Bsq += Bcon[mu]*Bcov[mu]
    Au += Acov[mu]*ucon[mu]
    Bu += Bcov[mu]*ucon[mu]
    AB += Acov[mu]*Bcon[mu]
  Au2 = Au*Au
  Bu2 = Bu*Bu
  AuBu = Au*Bu

  A = Bu2 - (Bsq + Bu2) * cs2
  B = 2. * (AuBu - (AB + AuBu) * cs2)
  C = Au2 - (Asq + Au2) * cs2
  discr = sqrt(B * B - 4. * A * C)
  vp = -(-B + discr)/(2.*A)
  vm = -(-B - discr)/(2.*A)
  if vp > vm:
    vmax = vp
    vmin = vm
  else:
    vmax = vm
    vmin = vp

  return vmax, vmin

def calc_sources(state):
  dU = zeros([N1TOT, N2TOT, Var.SIZE])
  for i in range(NG,N1+NG):
    for j in range(NG,N2+NG):
      point = Point(Location.CENT, i, j, state.prim[i,j,Var.RHO], state.prim[i,j,Var.UG],
        state.prim[i,j,Var.V1], state.prim[i,j,Var.V2])
      source = point.get_source()

      for v in range(Var.SIZE):
        dU[v] = source[v]

  return dU
