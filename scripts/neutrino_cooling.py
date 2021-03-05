PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'

import numpy as np
import sys
import matplotlib.pyplot as plt
import shutil
import os
from subprocess import call, DEVNULL
import glob
sys.path.append(PHDF_PATH)
import phdf
import time
from enum import Enum

NeutrinoSpecies = Enum('electron', 'electron anti')

s = NeutrinoSpecies.electron
mp = 1.672621777e-24
h = 6.62606957e-27
numax = 1.e17
numin = 1.e7
C = 1.
rho = 1.e6
u0 = 1.e20


Ac = mp/(h*rho)*C*np.log(numax/numin)
Bc = C*(numax - numin)

def get_yf(Ye):
  if s == NeutrinoSpecies.electron:
    return 2.*Ye
  elif s == NeutrinoSpecies.electron:
    return 1. - 2.*Ye
  else:
    return 0.
def get_Ye0():
  if s == NeutrinoSpecies.electron:
    return 0.5
  elif s == NeutrinoSpecies.electron:
    return 0.
  else:
    return None
def get_Ye(t):
  Ye0 = get_Ye0()
  if s == NeutrinoSpecies.electron:
    return np.exp(-2.*Ac*t)*Ye0
  elif s == NeutrinoSpecies.electron:
    return 0.5 + np.exp(-2.*Ac*t)*(Ye0 - 0.5)
  else:
    return None
def get_u(t):
  return u0 + Bc/(2.*Ac)*(np.exp(-2.*Ac*t) - 1.)


print("Ac = %e" % Ac)
print("BC = %e" % Bc)

t = np.logspace(0, 2, 128)
Ye = get_Ye(t)
u = get_u(t)

fig, axes = plt.subplots(2, 1)
ax = axes[0]
#plt.xlabel('N')
#plt.ylabel('L1')
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticklabels([])
#plt.title(physics + ' ' + mode_name)
ax.plot(t, Ye)
ax.set_ylabel('Ye')

ax = axes[1]
ax.plot(t, u)
ax.set_xlabel('t')
ax.set_ylabel('u')

plt.show()


