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

def prim_to_cons(prim, cons):
  loc = Location.CENT
  for i in range(N1TOT):
    for j in range(N2TOT):
      pvec = prim[i,j,:]
      pos = Position(i, j, loc)
      vcov = get_vcov(pvec, pos)
      Gamma = get_Gamma(pvec, pos)

      rho = pvec[Var.RHO]
      ug = pvec[Var.UG]
      P = get_pressure(pvec)
      h = get_enthalpy(pvec)
      D = rho*Gamma

      cons[i,j,Var.RHO] = Gamma*rho
      cons[i,j,Var.UG] = rho*h*Gamma**2 - P - D
      cons[i,j,Var.V1] = rho*h*Gamma**2*vcov[1]
      cons[i,j,Var.V2] = rho*h*Gamma**2*vcov[2]

def cons_to_prim(cons, prim):
  loc = Location.CENT
  for i in range(N1TOT):
    for j in range(N2TOT):
      pos = Position(i, j, loc)

      if i == 32 and j == 32:
        #rho = prim[i,j,Var.RHO]
        h = get_enthalpy(prim[i,j,:])
        w = prim[i,j,Var.RHO]*h
        Gamma = get_Gamma(prim[i,j,:], pos)
        #vcon = get_vcon(prim[i,j,:])
        #vcov = get_vcov(prim[i,j,:], pos)
        #vsq = 0
        #for mu in range(1,4):
        #  vsq += vcon[mu]*vcov[mu]
        #print("Gamma: ", Gamma)
        xi_guess = Gamma**2*w
        #print("xi: ", Gamma**2*w)
        #print("True Ssq: ", (xi**2*vsq))

        cvec = cons[i,j,:]
        D = cvec[Var.RHO]
        Scov = array([0, cvec[Var.V1], cvec[Var.V2], 0])
        Ssq = 0
        tau = cvec[Var.UG]
        #print("D: ", D)
        #print("tau: ", tau)
        for mu in range(1,4):
          for nu in range(1,4):
            Ssq += geom.gcon[i,j,loc,mu,nu]*Scov[mu]*Scov[nu]
            #Ssq += (rho*h*Gamma**2)**2*vcon[mu]*vcov[nu]
        #print("Scov: ", Scov)
        #print("Ssq: ", Ssq)
        #xi = Gamma**2*w
        #Gamma = sqrt(1./(1. - Ssq/xi**2))
        #print("Now Gamma: ", Gamma)

        def resid(xi):
          Gamma = sqrt(1./(1. - Ssq/xi**2))
          return xi - (gam - 1.)/gam*(xi/Gamma**2 - D/Gamma) - tau - D

        #print(resid(Gamma**2*w))
        #print(resid(2.466361842))
        #print(resid(xi))
        #sys.exit()

        xi = newton(resid, xi_guess, tol=1.e-12)
        print(xi)
        print(resid(xi))
        sys.exit()

