from numpy import *
import os, sys
sys.dont_write_bytecode = True

# Pybus imports
from geometry import *
from state import *
from util import *
from coordinates import *
from problem import *
from reconstruction import *
from physics import *

def calc_fluxes(state):
  prim_to_flux(state.ql[:,:,:,Dir.X1], state.Fl[:,:,:,Dir.X1], Location.FACE1, Dir.X1)
  prim_to_flux(state.qr[:,:,:,Dir.X1], state.Fr[:,:,:,Dir.X1], Location.FACE1, Dir.X1)
  prim_to_flux(state.ql[:,:,:,Dir.X2], state.Fl[:,:,:,Dir.X2], Location.FACE2, Dir.X2)
  prim_to_flux(state.qr[:,:,:,Dir.X2], state.Fr[:,:,:,Dir.X2], Location.FACE2, Dir.X2)

  prim_to_cons(state.ql[:,:,:,Dir.X1], state.Ul[:,:,:,Dir.X1], Location.FACE1)
  prim_to_cons(state.qr[:,:,:,Dir.X1], state.Ur[:,:,:,Dir.X1], Location.FACE1)
  prim_to_cons(state.ql[:,:,:,Dir.X2], state.Ul[:,:,:,Dir.X2], Location.FACE2)
  prim_to_cons(state.qr[:,:,:,Dir.X2], state.Ur[:,:,:,Dir.X2], Location.FACE2)

  # X1
  d = Dir.X1
  vmax1 = 0
  for i in range(NG,N1+NG+1):
    for j in range(NG,N2+NG+1):

      point_l = Point(Location.FACE1, i, j, state.ql[i,j,Var.RHO,d], state.ql[i,j,Var.UG,d],
        state.ql[i,j,Var.V1,d], state.ql[i,j,Var.V2,d])
      point_r = Point(Location.FACE1, i - 1, j, state.qr[i,j,Var.RHO,d], state.qr[i,j,Var.UG,d],
        state.qr[i,j,Var.V1,d], state.qr[i,j,Var.V2,d])

      vmax_l, vmin_l = get_vchar(point_l, d)
      vmax_r, vmin_r = get_vchar(point_r, d)

      vmax = fabs(max(max(0, vmax_l), vmax_r))
      vmin = fabs(max(max(0., -vmin_l), -vmin_r))
      vtop = max(vmax, vmin)
      if vtop > vmax1:
        vmax1 = vtop

      for v in range(Var.SIZE):
        Fl = state.Fl[i,j,v,d]
        Fr = state.Fr[i-1,j,v,d]
        Ul = state.Ul[i,j,v,d]
        Ur = state.Ur[i-1,j,v,d]
        state.Flux[i,j,v,d] = 0.5*(Fl + Fr - vtop*(Ur - Ul))

  # X2
  d = Dir.X2
  vmax2 = 0
  for i in range(NG,N1+NG+1):
    for j in range(NG,N2+NG+1):

      point_l = Point(Location.FACE2, i, j, state.ql[i,j,Var.RHO,d], state.ql[i,j,Var.UG,d],
        state.ql[i,j,Var.V1,d], state.ql[i,j,Var.V2,d])
      point_r = Point(Location.FACE2, i, j - 1, state.qr[i,j,Var.RHO,d], state.qr[i,j,Var.UG,d],
        state.qr[i,j,Var.V1,d], state.qr[i,j,Var.V2,d])

      vmax_l, vmin_l = get_vchar(point_l, d)
      vmax_r, vmin_r = get_vchar(point_r, d)

      vmax = fabs(max(max(0, vmax_l), vmax_r))
      vmin = fabs(max(max(0., -vmin_l), -vmin_r))
      vtop = max(vmax, vmin)
      if vtop > vmax2:
        vmax2 = vtop

      for v in range(Var.SIZE):
        Fl = state.Fl[i,j,v,d]
        Fr = state.Fr[i-1,j,v,d]
        Ul = state.Ul[i,j,v,d]
        Ur = state.Ur[i-1,j,v,d]
        state.Flux[i,j,v,d] = 0.5*(Fl + Fr - vtop*(Ur - Ul))

  return vmax1, vmax2

def advance(state_i, state_b, state_f, dt):
  state_f.prim = copy(state_i.prim)

  reconstruct_prims(state_i)

  vmax1, vmax2 = calc_fluxes(state_b)
  print("vmax1: ", vmax1)
  print("vmax2: ", vmax2)
