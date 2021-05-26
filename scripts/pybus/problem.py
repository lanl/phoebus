from numpy import *

from params import *
from state import *
from coordinates import *

def initialize_zone(prim, geom, i, j):
  a = geom_params['a']
  k = geom_params['k']

  loc = Location.CENT
  X = getX(loc, i, j)
  x = X[1]
  y = X[2] - a*sin(k*x) # Analytic solution is in Minkowski space

  mode = amp*cos(k1*x + k2*y)

  prim[i,j,Var.RHO] = rho0 + (drho*mode).real
  prim[i,j,Var.UG] = ug0 + (dug*mode).real

  v1 = v10 + (dv1*mode).real
  v2 = v20 + (dv2*mode).real
  vcon_mink = zeros(4)
  vcon_mink[1] = v1
  vcon_mink[2] = v2
  gcov = geom.get_gcov(loc, i, j)
  vsq_mink = 0
  for mu in range(1,4):
    for nu in range(1,4):
      vsq_mink += gcov[mu,nu]*vcon_mink[mu]*vcon_mink[nu]
  Gamma_mink = 1./sqrt(1. - vsq_mink)
  ucon_mink = zeros(4)
  ucon_mink[0] = Gamma_mink
  ucon_mink[1] = Gamma_mink*vcon_mink[1]
  ucon_mink[2] = Gamma_mink*vcon_mink[2]
  ucon_mink[3] = Gamma_mink*vcon_mink[3]
  Jinv = geom.Jinv(loc, i, j)
  ucon_snake = zeros(4)
  for mu in range(4):
    for nu in range(4):
      ucon_snake[mu] += Jinv[mu,nu]*ucon_mink[nu]
  Gamma_snake = ucon_snake[0]

  prim[i,j,Var.V1] = ucon_snake[1]/Gamma_snake
  prim[i,j,Var.V2] = ucon_snake[2]/Gamma_snake
