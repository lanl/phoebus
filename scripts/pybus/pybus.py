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
from boundaries import *
from plotting import *
from output import *

state = State()
state_tmp = State()

for i in range(NG,N1+NG):
  for j in range(NG,N2+NG):
    initialize_zone(state.prim, geom, i, j)
apply_boundaries(state)

write_dump(state, 0)
nd = 1
tdump = DTd

#plot(state, 'rho')

t = 0
dt = dt_init
ncyc = 0
safe_dt = 1.3
while t < tf:
  print("n = %6d" % ncyc + " t = %e" % t + " dt = %e" % dt)
  advance(state, state, state_tmp, dt/2)
  apply_boundaries(state_tmp)
  ndt = advance(state, state_tmp, state, dt)
  apply_boundaries(state)

  if ndt > safe_dt*dt:
    ndt = safe_dt*dt
  dt = ndt
  if t + dt > tf:
    dt = tf - t

  t += dt
  ncyc += 1

  if t > tdump:
    write_dump(state, nd)
    tdump += DTd
    nd += 1

  #if ncyc % 10 == 0:
  #  plot(state, 'rho')

write_dump(state, nd)
