from numpy import *
from enum import Enum
import os, sys
sys.dont_write_bytecode = True

# Pybus imports
from geometry import *

def FAIL(message):
  print(message)
  sys.exit()

class Location(Enum):
  CENT = 0
  FACE1 = 1
  FACE2 = 2
  CORN = 3

# Coordinates
X1min = -1
X1max = 1
X2min = -1
X2max = 1
nX1 = 128
nX2 = 128
ng = 4
dX1 = (X1max - X1min)/nX1
dX2 = (X2max - X2min)/nX2
nX1e = nX1 + 2*ng
nX2e = nX2 + 2*ng
X1 = zeros(nX1e)
X2 = zeros(nX2e)

def getX(loc, i, j):
  X = zeros(4)
  if loc == Location.CENT:
    X[1] = X1min + dX1*(i + 0.5 - ng)
    X[2] = X2min + dX2*(j + 0.5 - ng)
  elif loc == Location.FACE1:
    X[1] = X1min + dX1*(i - ng)
    X[2] = X2min + dX2*(j + 0.5 - ng)
  elif loc == Location.FACE2:
    X[1] = X1min + dX1*(i + 0.5 - ng)
    X[2] = X2min + dX2*(j - ng)
  elif loc == Location.CORN:
    X[1] = X1min + dX1*(i - ng)
    X[2] = X2min + dX2*(j - ng)
  else:
    FAIL("loc not supported!")

  return X

for i in range(nX2e):
  X1[i] = getX(Location.CENT, i, 0)[1]#X1min + dX1*(i + 0.5 - ng)
for j in range(nX2e):
  X2[j] = getX(Location.CENT, 0, j)[2]#X2min + dX2*(j + 0.5 - ng)

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
amp = 1.e-5

# Choose geometry
a = 0.2
k = pi
geom = Snake(a, k)

for i in range(nX1e):
  for j in range(nX2e):
    X = getX(Location.CENT, i, j)
    mode = amp*cos(k1*X1[i] + k2*X2[j])
    prim['rho'][i,j] = rho0 + (drho*mode).real
    prim['ug'][i,j] = ug0 + (dug*mode).real
    prim['v1'][i,j] = v10 + (dv1*mode).real
    prim['v2'][i,j] = v20 + (dv2*mode).real

    # Convert Minkowski velocity to snake coordinates
    vcon_mink = zeros(4)
    vcon_mink[1] = prim['v1'][i,j]
    vcon_mink[2] = prim['v2'][i,j]

    gcov = geom.gcov(X)
    vsq_mink = 0
    for mu in range(1,4):
      for nu in range(1,4):
        print("nu: %i" % nu)
        vsq_mink += gcov[mu,nu]*vcon_mink[mu]*vcon_mink[nu]

## Geometry
#class Geometry:
#  def gcon(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.gcon(X)
#
#  def gcon(self, X):
#    gcov = self.gcov(X)
#    return linalg.inv(gcov)
#
#  def dgcov(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.dgcov(X)
#
#  def dgcov(self, X):
#    dgcov = zeros([4,4,4])
#    eps = 1.e-5
#    X0 = zeros(4)
#    X1 = zeros(4)
#    for lam in range(4):
#      for mu in range(4):
#        X0[mu] = X[mu]
#        X1[mu] = X[mu]
#      X0[lam] -= eps
#      X1[lam] += eps
#      gcov0 = self.gcov(X0)
#      gcov1 = self.gcov(X1)
#      for mu in range(4):
#        for nu in range(4):
#          dgcov[mu,nu,lam] = (gcov1[mu,nu] - gcov0[mu,nu])/(X1[lam] - X0[lam])
#    return dgcov
#
#  def dgcon(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.dgcon(X)
#
#  def dgcon(self, X):
#    dgcon = zeros([4,4,4])
#    eps = 1.e-5
#    X0 = zeros(4)
#    X1 = zeros(4)
#    for lam in range(4):
#      for mu in range(4):
#        X0[mu] = X[mu]
#        X1[mu] = X[mu]
#      X0[lam] -= eps
#      X1[lam] += eps
#      gcon0 = self.gcon(X0)
#      gcon1 = self.gcon(X1)
#      for mu in range(4):
#        for nu in range(4):
#          dgcon[mu,nu,lam] = (gcon1[mu,nu] - gcon0[mu,nu])/(X1[lam] - X0[lam])
#    return dgcon
#
#  def Jinv(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.Jinv(X)
#
#  def Jinv(self, X):
#    J = self.J(X)
#    return J.inv()
#
#class Snake(Geometry):
#  def __init__(self, a, k):
#    self.a = a
#    self.k = k
#
#  def gcov(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.gcov(X)
#
#  def gcov(self, X):
#    gcov = zeros([4, 4])
#    delta = self.a*self.k*cos(self.k*X[1])
#    gcov[0,0] = -1
#    gcov[1,1] = 1
#    gcov[2,1] = -delta
#    gcov[1,2] = -delta
#    gcov[2,2] = delta**2 + 1
#    gcov[3,3] = 1
#    return gcov
#
#  def J(self, loc, i, j):
#    X = getX(loc, i, j)
#    return self.J(X)
#
#  def J(self, X):
#    J = zeros([4, 4])
#    for mu in range(4):
#      J[mu,mu] = 1.
#    J[2,1] = self.a*self.k*cos(self.k*X[1])
#
## Parameters
#a = 0.2
#k = pi
#
#
#geom = Snake(a, k)

print(geom.gcov([0, 0, 0, 0]))
print(geom.gcon([0, 0, 0, 0]))
print(geom.dgcov([0, 0, 0, 0]))
print(geom.dgcon([0, 0, 0, 0]))

import matplotlib.pyplot as plt
plt.figure()
plt.pcolormesh(prim['rho'], cmap='jet')
plt.show()
