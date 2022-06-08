#!/usr/bin/env python

# © 2022. Triad National Security, LLC. All rights reserved.  This
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

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import shutil
import os
from subprocess import call, DEVNULL
import glob
from parthenon_tools import phdf
import time
from enum import Enum
from phoebus_constants import cgs, scalefree
import phoebus_utils
from phoedf import phoedf

parser = argparse.ArgumentParser(description='Plot neutrino thermalization')
parser.add_argument('-f', '--files', dest='files', nargs='*', default=['rad_eql*.phdf'], help='List of input Parthenon hdf files to plot')
parser.add_argument('--savefig', type=bool, default=False, help='Whether to save figure')
parser.add_argument('--numin', type=float, default=1.e-4, help='Minimum frequency (Hz)')
parser.add_argument('--numax', type=float, default=1.e2, help='Maximum frequency (Hz)')
parser.add_argument('--nnu', type=int, default=100, help='Number of frequency support points')
args = parser.parse_args()

if args.savefig:
  matplotlib.use('Agg')

# Initial state
rho0 = 1.
Ye0 = 0.5
# TODO(BRR) Add velocity?
gamma = 5./3. # Ideal gas EOS
NSPECIES=3

nnu = args.nnu
lnumin = np.log10(args.numin)
lnumax = np.log10(args.numax)
dlnu = (lnumax - lnumin)/(nnu - 1)
nus = np.zeros(nnu)
for n in range(nnu):
  nus[n] = 10**(lnumin + n*dlnu)

# Read in the files
files = []
for file in args.files:
  files += glob.glob(file)
files = sorted(files)
nfiles = len(files)

# Set up unit conversions
dfile0 = phoedf(files[0])
params = dfile0.Params
eos = dfile0.GetEOS()
opac = dfile0.GetOpacity('scalefree')
const = scalefree

t = np.zeros(nfiles)
J = np.zeros(nfiles)
Jstd = np.zeros(nfiles)
ug = np.zeros(nfiles)
Tr = np.zeros(nfiles)
Tg = np.zeros(nfiles)
for n, filename in enumerate(files[0::1]):
  dfile = phoedf(filename)
  t[n] = dfile.Time
  Jfile = dfile.Get("r.p.J", flatten=False)[:,:,:,:,0]
  J[n] = np.mean(Jfile)
  Jstd[n] = np.std(Jfile)
  ugfile = dfile.Get("p.energy", flatten=False)
  ug[n] = np.mean(ugfile)
  Tr[n] = opac.dist.T_from_J(J[n])
  Tg[n] = eos.T_from_rho_u_Ye(1., ug[n], Ye0)

# ---------------------------------------------------------------------------- #
# -- Calculate analytic solution

tmax = t[-1]
n_soln = 1000
dt_soln = tmax/n_soln

t_soln = np.zeros(n_soln+1)
ug_soln = np.zeros(n_soln+1)
J_soln = np.zeros(n_soln+1)
Tg_soln = np.zeros(n_soln+1)
Tr_soln = np.zeros(n_soln+1)

t_soln[0] = 0.
Tg_soln[0] = Tg[0]
Tr_soln[0] = Tr[0]
ug_soln[0] = eos.u_from_rho_T_Ye(rho0, Tg[0], Ye0)
J_soln[0] = opac.dist.J_from_T(Tr[0])

# Initialize neutrino spectrum
Inu = np.zeros(nnu)
for n in range(nnu):
  Inu[n] = max(1.e-100, opac.dist.Bnu(Tr[0], nus[n]))

for cycle in range(n_soln):
  phoebus_utils.progress_bar((cycle+1)/n_soln)
  dJ = 0
  dInu = np.zeros(nnu)
  for n in range(nnu):
    nu = nus[n]
    jnu = opac.jnu(rho0, Tg_soln[cycle], Ye0, nu)
    alphanu = opac.alphanu(rho0, Tg_soln[cycle], Ye0, nu)

    dInu[n] = (Inu[n] + const['CL']*dt_soln*jnu)/(1. + const['CL']*dt_soln*alphanu) - Inu[n]
    Inu[n] += dInu[n]
    Inu[n] = max(Inu[n], 1.e-100)

  for n in range(nnu - 1):
    dJ += 4.*np.pi/const['CL']*(dInu[n+1] + dInu[n])/2*(nus[n+1]-nus[n])
  dJ -= 0.5 * 4.*np.pi/const['CL']*(dInu[0] + dInu[1])/2*(nus[1] - nus[0])
  dJ -= 0.5 * 4.*np.pi/const['CL']*(dInu[-2] + dInu[-1])/2*(nus[-1] - nus[-2])

  J_soln[cycle+1] = J_soln[cycle] + dJ
  ug_soln[cycle+1] = ug_soln[cycle] - dJ
  Tr_soln[cycle+1] = opac.dist.T_from_J(J_soln[cycle+1])
  Tg_soln[cycle+1] = eos.T_from_rho_u_Ye(rho0, ug_soln[cycle+1], Ye0)

  t_soln[cycle+1] = t_soln[cycle] + dt_soln

utot = ug[0] + J[0]
def resid(T):
  return utot - eos.u_from_rho_T_Ye(rho0, T, Ye0) - opac.dist.J_from_T(T)

# ---------------------------------------------------------------------------- #
# -- Plot solution

fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].plot(t, J)
ax[0].plot(t, ug)
ax[0].plot(t_soln, J_soln, color='k', linestyle='--')
ax[0].plot(t_soln, ug_soln, color='k', linestyle='--')
ax[0].set_xlim([0, None])
ax[0].set_ylim([0, None])
ax[1].plot(t, Tr)
ax[1].plot(t, Tg)
ax[1].plot(t_soln, Tr_soln, color='k', linestyle='--')
ax[1].plot(t_soln, Tg_soln, color='k', linestyle='--')
ax[1].set_xlim([0, None])
ax[1].set_ylim([0, None])
plt.show()
