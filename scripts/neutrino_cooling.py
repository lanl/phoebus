# © 2021. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.  Department
# of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the
# U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.

DUMP_NAMES = '/home/brryan/builds/phoebus/cooling.out1.*.phdf'

import numpy as np
import sys
import matplotlib.pyplot as plt
import shutil
import os
from subprocess import call, DEVNULL
import glob
from parthenon_tools import phdf
import time
from enum import Enum

NeutrinoSpecies = Enum('NeutrinoSpecies', 'electron electronanti')

dfnams = np.sort(glob.glob(DUMP_NAMES))
dfile0 = phdf.phdf(dfnams[0])

cl = 2.99792458e10
mp = 1.672621777e-24
h = 6.62606957e-27

T_unit = dfile0.Params['eos/time_unit']
L_unit = dfile0.Params['eos/length_unit']
M_unit = dfile0.Params['eos/mass_unit']
U_unit = M_unit / L_unit**3 * cl**2
numin = dfile0.Params['radiation/nu_min']
numax = dfile0.Params['radiation/nu_max']
do_nu_electron = dfile0.Params['radiation/do_nu_electron']
do_nu_electron_anti = dfile0.Params['radiation/do_nu_electron_anti']
do_nu_heavy = dfile0.Params['radiation/do_nu_heavy']

s = NeutrinoSpecies.electron
C = 4. * np.pi # This definition of j_nu differs by a factor 4pi from the nubhlght paper
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

t = np.logspace(0, 3, 128)
Ye = get_Ye(t)
u = get_u(t)

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
