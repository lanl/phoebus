from numpy import *
import os, sys
sys.dont_write_bytecode = True

# Pybus imports
from geometry import *
from state import *
from util import *
from coordinates import *
from driver import *
from physics import *
from problem import *
from reconstruction import *

# Coordinates
#X1min = -1
#X1max = 1
#X2min = -1
#X2max = 1
#nX1 = 128
#nX2 = 128
#ng = 4
#dX1 = (X1max - X1min)/nX1
#dX2 = (X2max - X2min)/nX2
#nX1e = nX1 + 2*ng
#nX2e = nX2 + 2*ng
#NX1E = nX1e
#NX2E = nX2e
#X1 = zeros(nX1e)
#X2 = zeros(nX2e)
#X1p = zeros(nX1e + 1)
#X2p = zeros(nX2e + 1)

#def getX(loc, i, j):
#  X = zeros(4)
#  if loc == Location.CENT:
#    X[1] = X1min + dX1*(i + 0.5 - ng)
#    X[2] = X2min + dX2*(j + 0.5 - ng)
#  elif loc == Location.FACE1:
#    X[1] = X1min + dX1*(i - ng)
#    X[2] = X2min + dX2*(j + 0.5 - ng)
#  elif loc == Location.FACE2:
#    X[1] = X1min + dX1*(i + 0.5 - ng)
#    X[2] = X2min + dX2*(j - ng)
#  elif loc == Location.CORN:
#    X[1] = X1min + dX1*(i - ng)
#    X[2] = X2min + dX2*(j - ng)
#  else:
#    FAIL("loc not supported!")
#
#  return X

#for i in range(nX1e):
#  X1[i] = getX(Location.CENT, i, 0)[1]
#for j in range(nX2e):
#  X2[j] = getX(Location.CENT, 0, j)[2]

#for i in range(nX1e + 1):
#  X1p[i] = getX(Location.CORN, i, 0)[1]
#for j in range(nX2e + 1):
#  X2p[j] = getX(Location.CORN, 0, j)[2]


# Initial conditions
#omega = 0.0 + 3.8780661653218766j
#drho = 0.5804294924639213
#dug = 0.7739059899518947
#dv1 = 0.1791244302079596
#dv2 = 0.1791244302079596
#rho0 = 1.0
#ug0 = 1.0
#v10 = 0.0
#v20 = 0.0
#k1 = 2.*pi
#k2 = 2.*pi
#amp = 1.e-3

#state = zeros([N1TOT, N2TOT, Var.SIZE])

state = State()
state_tmp = State()

#for i in range(N1TOT):
#  for j in range(N2TOT):
#    initialize_zone(state.prim, geom, i, j)

#prim_to_cons(state.prim, state.cons)
#cons_to_prim(state.cons, state.prim)

#reconstruct_prims(state)

t = 0
dt = dt_init
ncyc = 0
while t < tf:
  advance(state, state, state_tmp, dt)
  print("n = %6d" % ncyc + " t = %e" % t + " dt = %e" % dt)

  t += dt
  ncyc += 1

  if ncyc > 10:
    sys.exit()


import matplotlib.pyplot as plt
plt.figure()
#plt.pcolormesh(state.prim[:,:,Var.RHO], cmap='jet')
plt.pcolormesh(state.ql[:,:,Var.RHO,Dir.X1], cmap='jet', vmin=1. - amp, vmax=1.+amp)
ax = plt.gca()
ax.set_aspect('equal')
#ax.set_xlim([X1min, X1max])
#ax.set_ylim([X2min, X2max])
plt.show()

sys.exit()

