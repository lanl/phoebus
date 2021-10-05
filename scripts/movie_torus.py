#!/usr/bin/env python3

PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'
DUMP_NAMES = '/home/brryan/builds/phoebus/torus.out1.*.phdf'

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

#plot = "mks"
plot = "cartesian"

# Outer radius to plot or None
rmax = 40
#rmax = None
mblw = 0.5

#xmin=0
#xmax=40
#ymin=-40
#ymax=40
xmin = 0
xmax = 12
ymin = -10
ymax = 10
xmin = 6
xmax = 10
ymin = -5
ymax = 5

#xmin = 0
#xmax = 5
#ymin = -5
#ymax = 5

# Whether to plot meshblock boundaries
plot_meshblocks = True
h_ = 0.3

# Plotting macro
def myplot(myvar, n, vmin=-5, vmax=0, uselog=True, cmap='jet', half=False):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  ax = axes[n]
  #ax = axes
  for n in range(nblocks): # shadowing previous n
    if plot == "mks":
      im = ax.pcolormesh(xblock[n,:,:], yblock[n,:,:], np.log10(myvar[n,0].transpose()),
        vmin=vmin, vmax=vmax, cmap=cmap)
    elif plot == "cartesian":
      if uselog:
        if half:
          im = ax.pcolormesh(x[n,:,:128], y[n,:,:128], np.log10(myvar[n,0].transpose()[:,:128]),
            vmin=vmin, vmax=vmax, cmap=cmap)
        else:
          im = ax.pcolormesh(x[n,:,:], y[n,:,:], np.log10(myvar[n,0].transpose()),
            vmin=vmin, vmax=vmax, cmap=cmap)
      else:
        im = ax.pcolormesh(x[n,:,:], y[n,:,:], myvar[n,0].transpose(),
          vmin=vmin, vmax=vmax, cmap=cmap)

      if plot_meshblocks:
        ax.plot(x[n,0,:], y[n,0,:], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[n,-1,:], y[n,-1,:], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[n,:,0], y[n,:,0], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[n,:,-1], y[n,:,-1], color='k', linewidth=mblw, linestyle='--')
      if rmax is not None:
        ax.set_xlim([0,rmax])
        ax.set_ylim([-rmax,rmax])
      #ax.set_xlim([6,12])
      #ax.set_ylim([-5,5])
      #ax.set_xlim([0,40])
      #ax.set_ylim([-40,40])
      ax.set_xlim([xmin,xmax])
      ax.set_ylim([ymin,ymax])
    else:
      print("Plotting coordinates \"" + plot + "\" unknown")
      sys.exit()
  if plot == "cartesian":
    ax.set_aspect('equal')
  ax.set_xlabel('x')
  #if n == 0:
  #  ax.set_ylabel('y')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  #plt.colorbar(im, label='density')

dfnams = np.sort(glob.glob(DUMP_NAMES))

# Get geometry
dfile = phdf.phdf(dfnams[0])
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

# Convert from FMKS to xy
r = np.exp(xblock)
th = np.pi*yblock + ((1. - h_)/2.)*np.sin(2.*np.pi*yblock)
x = r*np.sin(th)
y = r*np.cos(th)

def calc_mass(dfile):
  dx1 = dx
  dx2 = dy
  dx3 = 2.*np.pi
  dV = dx1*dx2*dx3
  crho = dfile.Get("c.density", flatten=True)
  rhosum = np.sum(crho)
  if np.isnan(rhosum):
    print("NAN in c.density!")
    crho = dfile.Get("c.density", flatten=False)
    print(list(map(tuple, np.where(np.isnan(crho)))))
    sys.exit()
  detgam = dfile.Get("g.c.detgam", flatten=True)
  mass = sum(crho*detgam*dV)
  return mass

mass = []
for n, dfnam in enumerate(dfnams):
  print(f"Frame {n+1} out of {len(dfnams)}")
  dfile = phdf.phdf(dfnam)
  mass.append(calc_mass(dfile))
fig, axes = plt.subplots(1, 1, figsize=(10,10))
axes.plot(mass/mass[0], color='k')
axes.set_xlim([0,200])
plt.show()
sys.exit()

dfile0 = phdf.phdf(dfnams[0])
dfile0 = phdf.phdf(dfnams[50])
print(calc_mass(dfile0))
sys.exit()
density0 = dfile0.Get("p.density", flatten=False)
ug0 = dfile0.Get("p.energy", flatten=False)
vel0 = dfile0.Get("p.velocity", flatten=False)
v10 = np.clip(vel0[:,:,:,:,0], 1.e-100, 1.e100)
v20 = np.clip(vel0[:,:,:,:,1], 1.e-100, 1.e100)
v30 = np.clip(vel0[:,:,:,:,2], 1.e-100, 1.e100)
for n, dfnam in enumerate(dfnams):
  print(f"Frame {n+1} out of {len(dfnams)}")
  dfile = phdf.phdf(dfnam)
  vel = dfile.Get("p.velocity", flatten=False)
  density = dfile.Get("p.density", flatten=False)
  ug = dfile.Get("p.energy", flatten=False)
  v1 = vel[:,:,:,:,0]
  v2 = vel[:,:,:,:,1]
  v3 = vel[:,:,:,:,2]

  fig, axes = plt.subplots(1, 5, figsize=(14,8))
  myplot(density, 0)
  myplot(ug/density, 1)
  myplot(np.fabs(v1), 2)
  myplot(np.fabs(v2), 3)
  myplot(np.fabs(v3), 4)

  myplot(density0, 0, half=True)
  myplot(ug0/density0, 1, half=True)
  myplot(np.fabs(v10), 2, half=True)
  myplot(np.fabs(v20), 3, half=True)
  myplot(np.fabs(v30), 4, half=True)
  plt.savefig("frame_%08d.png" % n, bbox_inches='tight', dpi=300)
  plt.close()
