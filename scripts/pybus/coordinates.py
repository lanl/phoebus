from enum import IntEnum

from numpy import *
from params import *

class Dir(IntEnum):
  X0 = 0
  X1 = 1
  X2 = 2
  SIZE = 3

class Location(IntEnum):
  CENT = 0
  FACE1 = 1
  FACE2 = 2
  CORN = 3
  SIZE = 4

# Coordinates
dX1 = (X1max - X1min)/N1
dX2 = (X2max - X2min)/N2
N1TOT = N1 + 2*NG
N2TOT = N2 + 2*NG

def getX(loc, i, j):
  X = zeros(4)
  if loc == Location.CENT:
    X[1] = X1min + dX1*(i + 0.5 - NG)
    X[2] = X2min + dX2*(j + 0.5 - NG)
  elif loc == Location.FACE1:
    X[1] = X1min + dX1*(i - NG)
    X[2] = X2min + dX2*(j + 0.5 - NG)
  elif loc == Location.FACE2:
    X[1] = X1min + dX1*(i + 0.5 - NG)
    X[2] = X2min + dX2*(j - NG)
  elif loc == Location.CORN:
    X[1] = X1min + dX1*(i - NG)
    X[2] = X2min + dX2*(j - NG)
  else:
    FAIL("loc not supported!")

  return X
