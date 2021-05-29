from numpy import *

from params import *
from coordinates import *
from state import *

def reconstruct_linear(y0, y1, y2, y3, y4):
  # Monotonized Central
  Dqm = 2. * (y2 - y1)
  Dqp = 2. * (y3 - y2)
  Dqc = 0.5 * (y3 - y1)
  s = Dqm * Dqp
  if s <= 0.:
    slope = 0.
  else:
    if fabs(Dqm) < fabs(Dqp) and fabs(Dqm) < fabs(Dqc):
      slope = Dqm
    elif fabs(Dqp) < fabs(Dqc):
      slope = Dqp
    else:
      slope = Dqc

  y_l = y2 - 0.5*slope
  y_r = y2 + 0.5*slope

  return y_l, y_r

def reconstruct_prims(state):
  for i in range(NG-1, N1+NG+1):
    for j in range(NG-1, N2+NG+1):
      for p in range(Var.SIZE):
        # X1
       state.ql[i,j,p,Dir.X1], state.qr[i,j,p,Dir.X1] = reconstruct_linear(state.prim[i-2,j,p],
                          state.prim[i-1,j,p],
                          state.prim[i,j,p],
                          state.prim[i+1,j,p],
                          state.prim[i+2,j,p])

        # X2
       state.ql[i,j,p,Dir.X2], state.qr[i,j,p,Dir.X2] = reconstruct_linear(state.prim[i,j-2,p],
                          state.prim[i,j-1,p],
                          state.prim[i,j,p],
                          state.prim[i,j+1,p],
                          state.prim[i,j+2,p])
