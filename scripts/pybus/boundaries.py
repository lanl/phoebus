from numpy import *

from state import *
from params import *
from coordinates import *

def apply_boundaries(state):
  for j in range(NG,N2+NG):
    for i in range(0, NG):
      iactive = i + N1
      for v in range(Var.SIZE):
        state.prim[i,j,v] = state.prim[iactive,j,v]
    for i in range(N1+NG,N1+2*NG):
      iactive = i - N1
      for v in range(Var.SIZE):
        state.prim[i,j,v] = state.prim[iactive,j,v]

  for i in range(NG,N1+NG):
    for j in range(0, NG):
      jactive = j + N2
      for v in range(Var.SIZE):
        state.prim[i,j,v] = state.prim[i,jactive,v]
    for j in range(N2+NG,N2+2*NG):
      jactive = j - N2
      for v in range(Var.SIZE):
        state.prim[i,j,v] = state.prim[i,jactive,v]
