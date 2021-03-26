PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'
DUMP_NAMES = '/home/brryan/builds/phoebus/cooling.out1.*.phdf'

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

NeutrinoSpecies = Enum('NeutrinoSpecies', 'electron electronanti')

s = NeutrinoSpecies.electron
mp = 1.672621777e-24
h = 6.62606957e-27
numax = 1.e22
numin = 1.e20
C = 1.
rho = 1.e6
u0 = 1.e20

Ac = mp/(h*rho)*C*np.log(numax/numin)
Bc = C*(numax - numin)

def get_yf(Ye):
  if s == NeutrinoSpecies.electron:
    return 2.*Ye
  elif s == NeutrinoSpecies.electronanti:
    return 1. - 2.*Ye
  else:
    return 0.
def get_Ye0():
  if s == NeutrinoSpecies.electron:
    return 0.5
  elif s == NeutrinoSpecies.electronanti:
    return 0.
  else:
    return None
def get_Ye(t):
  Ye0 = get_Ye0()
  if s == NeutrinoSpecies.electron:
    return np.exp(-2.*Ac*t)*Ye0
  elif s == NeutrinoSpecies.electronanti:
    return 0.5 + np.exp(-2.*Ac*t)*(Ye0 - 0.5)
  else:
    return None
def get_u(t):
  return u0 + Bc/(2.*Ac)*(np.exp(-2.*Ac*t) - 1.)

t = np.logspace(0, 2, 128)
Ye = get_Ye(t)
u = get_u(t)

# TODO(BRR) get these from dump files
T_unit = 1./2.997925e-04
U_unit = 8.987552e-22

dfnams = np.sort(glob.glob(DUMP_NAMES))
t_code = np.zeros(dfnams.size)
Ye_code = np.zeros(dfnams.size)
u_code = np.zeros(dfnams.size)
for n, dfnam in enumerate(dfnams):
  dfile = phdf.phdf(dfnam)
  t_code[n] = dfile.Time*T_unit
  Ye_code[n] = dfile.Get("p.ye").mean()
  u_code[n] = dfile.Get("p.energy").mean()*U_unit

fig, axes = plt.subplots(2, 1, figsize=(8,6))
ax = axes[0]
ax.set_yscale('log')
ax.set_xticklabels([])
ax.plot(t_code, Ye_code, color='r', label='phoebus')
ax.plot(t, Ye, color='k', linestyle='--', label='Analytic')
ax.set_xlim([0, t[-1]])
ax.set_ylabel('Ye')
ax.legend(loc=1)

ax = axes[1]
ax.plot(t_code, u_code, color='r')
ax.plot(t, u, color='k', linestyle='--')
ax.set_xlim([0, t[-1]])
ax.set_xlabel('t')
ax.set_ylabel('u')

plt.show()
