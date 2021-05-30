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
  print("calc_fluxes")
  print(state.ql[:,:,:,Dir.X1].max())
  prim_to_flux(state.ql[:,:,:,Dir.X1], state.Fl, Location.FACE1, Dir.X1)
  #prim_to_flux(state.ql[:,:,:,Dir.X1], state.Fl[:,:,:,Dir.X1], Location.FACE1, Dir.X1)
  print(state.Fl[NG,NG,:,Dir.X1])
  print("DFOIJDSF")
  sys.exit()
  #prim_to_flux(state.qr[:,:,:,Dir.X1], state.Fr[:,:,:,Dir.X1], Location.FACE1, Dir.X1)
  #prim_to_flux(state.ql[:,:,:,Dir.X2], state.Fl[:,:,:,Dir.X2], Location.FACE2, Dir.X2)
  #prim_to_flux(state.qr[:,:,:,Dir.X2], state.Fr[:,:,:,Dir.X2], Location.FACE2, Dir.X2)

  prim_to_cons(state.ql[:,:,:,Dir.X1], state.Ul[:,:,:,Dir.X1], Location.FACE1)
  prim_to_cons(state.qr[:,:,:,Dir.X1], state.Ur[:,:,:,Dir.X1], Location.FACE1)
  prim_to_cons(state.ql[:,:,:,Dir.X2], state.Ul[:,:,:,Dir.X2], Location.FACE2)
  prim_to_cons(state.qr[:,:,:,Dir.X2], state.Ur[:,:,:,Dir.X2], Location.FACE2)

  print(state.Fl.max())
  print(state.Fl[:,:,:,Dir.X1].max())
  print(state.Fr[:,:,:,Dir.X1].max())

  # X1
  d = Dir.X1
  vmax1 = 0
  for i in range(NG,N1+NG+1):
    for j in range(NG,N2+NG+1):

      #point_l = Point(Location.FACE1, i, j, state.ql[i,j,Var.RHO,d], state.ql[i,j,Var.UG,d],
      #  state.ql[i,j,Var.V1,d], state.ql[i,j,Var.V2,d])
      #point_r = Point(Location.FACE1, i, j, state.qr[i-1,j,Var.RHO,d], state.qr[i-1,j,Var.UG,d],
      #  state.qr[i-1,j,Var.V1,d], state.qr[i-1,j,Var.V2,d])
      point_r = Point(Location.FACE1, i, j, state.ql[i,j,Var.RHO,d], state.ql[i,j,Var.UG,d],
        state.ql[i,j,Var.V1,d], state.ql[i,j,Var.V2,d])
      point_l = Point(Location.FACE1, i, j, state.qr[i-1,j,Var.RHO,d], state.qr[i-1,j,Var.UG,d],
        state.qr[i-1,j,Var.V1,d], state.qr[i-1,j,Var.V2,d])

      vmax_l, vmin_l = get_vchar(point_l, d)
      vmax_r, vmin_r = get_vchar(point_r, d)

      vmax = fabs(max(max(0, vmax_l), vmax_r))
      vmin = fabs(max(max(0., -vmin_l), -vmin_r))
      vtop = max(vmax, vmin)
      if vtop > vmax1:
        vmax1 = vtop

      for v in range(Var.SIZE):
        #Fl = state.Fl[i,j,v,d]
        #Fr = state.Fr[i-1,j,v,d]
        #Ul = state.Ul[i,j,v,d]
        #Ur = state.Ur[i-1,j,v,d]
        Fr = state.Fl[i,j,v,d]
        Fl = state.Fr[i-1,j,v,d]
        Ur = state.Ul[i,j,v,d]
        Ul = state.Ur[i-1,j,v,d]
        state.Flux[i,j,v,d] = 0.5*(Fl + Fr - vtop*(Ur - Ul))
        if v == Var.RHO:
          print(Fl, Fr, state.Flux[i,j,v,d])
          sys.exit()

  # X2
#  d = Dir.X2
#  vmax2 = 0
#  for i in range(NG,N1+NG+1):
#    for j in range(NG,N2+NG+1):
#
#      point_l = Point(Location.FACE2, i, j, state.ql[i,j,Var.RHO,d], state.ql[i,j,Var.UG,d],
#        state.ql[i,j,Var.V1,d], state.ql[i,j,Var.V2,d])
#      point_r = Point(Location.FACE2, i, j, state.qr[i,j-1,Var.RHO,d], state.qr[i,j-1,Var.UG,d],
#        state.qr[i,j-1,Var.V1,d], state.qr[i,j-1,Var.V2,d])
#
#      vmax_l, vmin_l = get_vchar(point_l, d)
#      vmax_r, vmin_r = get_vchar(point_r, d)
#
#      vmax = fabs(max(max(0, vmax_l), vmax_r))
#      vmin = fabs(max(max(0., -vmin_l), -vmin_r))
#      vtop = max(vmax, vmin)
#      if vtop > vmax2:
#        vmax2 = vtop
#
#      for v in range(Var.SIZE):
#        Fl = state.Fl[i,j,v,d]
#        Fr = state.Fr[i,j-1,v,d]
#        Ul = state.Ul[i,j,v,d]
#        Ur = state.Ur[i,j-1,v,d]
#        state.Flux[i,j,v,d] = 0.5*(Fl + Fr - vtop*(Ur - Ul))
  vmax2 = vmax1

  return vmax1, vmax2

def advance(state_i, state_b, state_f, dt):
  state_f.prim = copy(state_i.prim)

  reconstruct_prims(state_b)

  print(state_b.ql.max())
  print(state_b.ql[:,:,:,Dir.X1].max())

  vmax1, vmax2 = calc_fluxes(state_b)

  # Get sources

  prim_to_cons(state_f.prim, state_f.cons, Location.CENT)
  for i in range(NG, NG+N1):
    for j in range(NG, NG+N2):
      for v in range(Var.SIZE):
        state_f.cons[i,j,v] += dt*(
          (state_b.Flux[i,j,v,Dir.X1] - state_b.Flux[i+1,j,v,Dir.X1])/dX1
        + (state_b.Flux[i,j,v,Dir.X2] - state_b.Flux[i,j+1,v,Dir.X2])/dX2)

  cons_to_prim(state_f.cons, state_f.prim)

  ndt1 = cfl*dX1/vmax1
  ndt2 = cfl*dX2/vmax2
  return 1./(1./ndt1 + 1./ndt2)
