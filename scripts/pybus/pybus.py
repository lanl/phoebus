from numpy import *
import os, sys
sys.dont_write_bytecode = True

# Pybus imports
from geometry import *
from state import *
from util import *
from coordinates import *
from problem import *

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

for i in range(nX1e):
  X1[i] = getX(Location.CENT, i, 0)[1]
for j in range(nX2e):
  X2[j] = getX(Location.CENT, 0, j)[2]

for i in range(nX1e + 1):
  X1p[i] = getX(Location.CORN, i, 0)[1]
for j in range(nX2e + 1):
  X2p[j] = getX(Location.CORN, 0, j)[2]


# Variables
prim = {}
prim['rho'] = zeros([nX1e, nX2e])
prim['ug'] = zeros([nX1e, nX2e])
prim['v1'] = zeros([nX1e, nX2e])
prim['v2'] = zeros([nX1e, nX2e])

# Initial conditions
omega = 0.0 + 3.8780661653218766j
drho = 0.5804294924639213
dug = 0.7739059899518947
dv1 = 0.1791244302079596
dv2 = 0.1791244302079596
rho0 = 1.0
ug0 = 1.0
v10 = 0.0
v20 = 0.0
k1 = 2.*pi
k2 = 2.*pi
amp = 1.e-3

for i in range(nX1e):
  for j in range(nX2e):
    loc = Location.CENT
    X = getX(Location.CENT, i, j)
    x = X[1]
    y = X[2] - a*sin(k*x)
    #mode = amp*cos(k1*X1[i] + k2*X2[j])
    mode = amp*cos(k1*x + k2*y)
    prim['rho'][i,j] = rho0 + (drho*mode).real
    prim['ug'][i,j] = ug0 + (dug*mode).real
    prim['v1'][i,j] = v10 + (dv1*mode).real
    prim['v2'][i,j] = v20 + (dv2*mode).real

    if i == 64 and j == 64:
      print("x = %e" % x)
      print("y = %e" % y)
      print(prim['rho'][i,j])

    # Convert Minkowski velocity to snake coordinates
    vcon_mink = zeros(4)
    vcon_mink[1] = prim['v1'][i,j]
    vcon_mink[2] = prim['v2'][i,j]

    gcov = geom.gcov(Location.CENT, i, j)#X)
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
    prim['v1'][i,j] = ucon_snake[1]/Gamma_snake
    prim['v2'][i,j] = ucon_snake[2]/Gamma_snake

#i = 32
#j = 32
## X1 dir
#print("X1 dir")
#d = 1
#ql = State(Location.FACE1, i, j, 1.000511e+00, 1.000682e+00, 1.577645e-04, 1.386013e-04)
#qr = State(Location.FACE1, i, j, 1.000511e+00, 1.000681e+00, 1.576970e-04, 1.384516e-04)
#Ul = ql.get_cons()
#Ur = qr.get_cons()
#Fl = ql.get_F(d)
#Fr = qr.get_F(d)
#print("Ul: ", Ul)
#print("Ur: ", Ur)
#print("Fl: ", Fl)
#print("Fr: ", Fr)
#print("")
#
## X2 dir
#print("X2 dir")
#d = 2
#ql = State(Location.FACE2, i, j, 1.000510e+00, 1.000679e+00, 1.572381e-04, 1.403478e-04)
#qr = State(Location.FACE2, i, j, 1.000509e+00, 1.000679e+00, 1.572177e-04, 1.403296e-04)
#Ul = ql.get_cons()
#Ur = qr.get_cons()
#Fl = ql.get_F(d)
#Fr = qr.get_F(d)
#print("Ul: ", Ul)
#print("Ur: ", Ur)
#print("Fl: ", Fl)
#print("Fr: ", Fr)
#print("")

if do_plot:
  import matplotlib.pyplot as plt
  plt.figure()
  plt.pcolormesh(X1p, X2p, prim['v2'], cmap='jet')
  ax = plt.gca()
  ax.set_aspect('equal')
  ax.set_xlim([X1min, X1max])
  ax.set_ylim([X2min, X2max])
  plt.show()
