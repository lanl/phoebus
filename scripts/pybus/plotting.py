from coordinates import *
from params import *
from state import *
from util import *

import matplotlib.pyplot as plt

x = zeros(N1+1)
y = zeros(N2+1)
for i in range(N1+1):
  x[i] = getX(Location.FACE1, i+NG, 0)[1]
for j in range(N2+1):
  y[j] = getX(Location.FACE2, 0, j+NG)[2]

def plot(state, varname, vmin=None, vmax=None):
  plt.figure()
  if varname == 'rho':
    var = state.prim[NG:N1+NG,NG:N2+NG,Var.RHO]
  else:
    FAIL("varname \"" + str(varname) + "\" not recognized!")
  plt.pcolormesh(x, y, var, cmap='jet',vmin=vmin,vmax=vmax)
  ax = plt.gca()
  ax.set_aspect('equal')
  ax.set_xlim([X1min, X1max])
  ax.set_ylim([X2min, X2max])
  plt.colorbar()
  plt.show()
