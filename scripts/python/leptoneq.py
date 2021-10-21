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

PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'
DUMP_NAMES = '/home/brryan/builds/phoebus/leptoneq.out1.*.phdf'

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
numax = 1.e17
numin = 1.e15
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

t = np.logspace(0, 3, 128)
Ye = get_Ye(t)
u = get_u(t)

# TODO(BRR) get these from dump files
T_unit = 1./2.997925e-04
U_unit = 8.987552e-22
#T_unit = 1./3.335641e-01
#U_unit = 8.987552e-10

dfnams = np.sort(glob.glob(DUMP_NAMES))
dfile = phdf.phdf(dfnams[-1])

nblocks = dfile.NumBlocks

meshblocksize = dfile.MeshBlockSize
nx = meshblocksize[0]
ny = meshblocksize[1]

blockbounds = dfile.BlockBounds
dx = (blockbounds[0][1] - blockbounds[0][0])/nx
dy = (blockbounds[0][3] - blockbounds[0][2])/ny

# Get pcolormesh grid for each block
xblock = np.zeros([nblocks, nx+1, ny+1])
yblock = np.zeros([nblocks, nx+1, ny+1])
for n in range(nblocks):
  for i in range(nx+1):
    for j in range(ny+1):
      dx = (blockbounds[n][1] - blockbounds[n][0])/nx
      dy = (blockbounds[n][3] - blockbounds[n][2])/ny
      xblock[n,i,j] = blockbounds[n][0] + i*dx
      yblock[n,i,j] = blockbounds[n][2] + j*dy

# Numblocks, nz, ny, nx
Ye_code = dfile.Get("p.ye", flatten=False)
print(Ye_code.max())
print(Ye_code.min())

vmin = 0.1
vmax = 0.35
fig, axes = plt.subplots(1, 1, figsize=(8,6))
ax = axes
for n in range(nblocks):
  im = ax.pcolormesh(xblock[n,:,:], yblock[n,:,:], Ye_code[n,0],
    vmin=vmin, vmax=vmax)
for n in range(nblocks):
  rect = patches.Rectangle((blockbounds[n][0], blockbounds[n][2]),
    blockbounds[n][1]-blockbounds[n][0], blockbounds[n][3]-blockbounds[n][2],
    linewidth=1, edgecolor='k', facecolor='none')
  ax.add_patch(rect)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, label='Ye')
plt.show()

sys.exit()

#t_code = np.zeros(dfnams.size)
#Ye_code = np.zeros(dfnams.size)
#u_code = np.zeros(dfnams.size)
#for n, dfnam in enumerate(dfnams):
#  dfile = phdf.phdf(dfnam)
#  t_code[n] = dfile.Time*T_unit
#  Ye_code[n] = dfile.Get("p.ye").mean()
#  u_code[n] = dfile.Get("p.energy").mean()*U_unit

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
