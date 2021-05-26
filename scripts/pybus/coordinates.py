from enum import Enum

from numpy import *
from problem import *

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
NX1E = nX1e
NX2E = nX2e
X1 = zeros(nX1e)
X2 = zeros(nX2e)
X1p = zeros(nX1e + 1)
X2p = zeros(nX2e + 1)

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
