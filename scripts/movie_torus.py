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

dpi = 150

xmin=0
xmax=40
ymin=-40
ymax=40
#xmin = 0
#xmax = 12
#ymin = -10
#ymax = 10
#xmin = 6
#xmax = 10
#ymin = -5
#ymax = 5
xmin = 4
xmax = 8
ymin = -4
ymax = 4

xmin = 6.8
xmax = 8.0
ymin = -3.2
ymax = -1.6

xmin = 5.9
xmax = 6.3
ymin = -.65
ymax = -.3

#xmin = 0
#xmax = 5
#ymin = -5
#ymax = 5

# Whether to plot meshblock boundaries
plot_meshblocks = True
h_ = 0.3

ndim = 256

# Plotting macro
#def myplot(myvar, n, vmin=-12, vmax=0, uselog=True, cmap='jet', half=False, cbar=True):
def myplot(myvar, n, vmin=-8, vmax=0, uselog=True, cmap='jet', half=False, cbar=True):
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
          im = ax.pcolormesh(x[n,:,:ndim//2], y[n,:,:ndim//2], np.log10(myvar[n,0].transpose()[:,:ndim//2]),
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
      #ax.plot([7.38419], [-2.51564], marker='.', color='r')
      ax.plot([6.1], [-0.42], marker='.', color='r', markeredgecolor='k')
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
  if cbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
  #plt.colorbar(im, label='density')
  ax.set_yticklabels([])

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

#print(blockbounds)
#sys.exit()

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

xblockc = np.zeros([nblocks, nx, ny])
yblockc = np.zeros([nblocks, nx, ny])
for n in range(nblocks):
  for i in range(nx):
    for j in range(ny):
      dx = (blockbounds[n][1] - blockbounds[n][0])/nx
      dy = (blockbounds[n][3] - blockbounds[n][2])/ny
      xblockc[n,i,j] = blockbounds[n][0] + (i + 0.5)*dx
      yblockc[n,i,j] = blockbounds[n][2] + (j + 0.5)*dy
rc = np.exp(xblockc)
thc = np.pi*yblockc + ((1. - h_)/2.)*np.sin(2.*np.pi*yblockc)
xc = rc*np.sin(thc)
yc = rc*np.cos(thc)

#def calc_mass(dfile):
#  dx1 = dx
#  dx2 = dy
#  dx3 = 2.*np.pi
#  dV = dx1*dx2*dx3
#  crho = dfile.Get("c.density", flatten=True)
#  rhosum = np.sum(crho)
#  if np.isnan(rhosum):
#    print("NAN in c.density!")
#    crho = dfile.Get("c.density", flatten=False)
#    print(list(map(tuple, np.where(np.isnan(crho)))))
#    sys.exit()
#  detgam = dfile.Get("g.c.detgam", flatten=True)
#  mass = sum(crho*detgam*dV)
#  return mass
#
#mass = []
#for n, dfnam in enumerate(dfnams):
#  print(f"Frame {n+1} out of {len(dfnams)}")
#  dfile = phdf.phdf(dfnam)
#  mass.append(calc_mass(dfile))
#fig, axes = plt.subplots(1, 1, figsize=(10,10))
#axes.plot(mass/mass[0], color='k')
#axes.set_xlim([0,200])
#plt.show()
#sys.exit()

dfile0 = phdf.phdf(dfnams[0])
#dfile0 = phdf.phdf(dfnams[50])
#print(calc_mass(dfile0))
#sys.exit()
density0 = dfile0.Get("p.density", flatten=False)
ug0 = dfile0.Get("p.energy", flatten=False)
vel0 = dfile0.Get("p.velocity", flatten=False)
v10 = np.clip(vel0[:,:,:,:,0], 1.e-100, 1.e100)
v20 = np.clip(vel0[:,:,:,:,1], 1.e-100, 1.e100)
v30 = np.clip(vel0[:,:,:,:,2], 1.e-100, 1.e100)
for n, dfnam in enumerate(dfnams):
  print(f"Frame {n+1} out of {len(dfnams)}")
  if n < 35:
    continue
  dfile = phdf.phdf(dfnam)
  vel = dfile.Get("p.velocity", flatten=False)
  density = dfile.Get("p.density", flatten=False)
  ug = dfile.Get("p.energy", flatten=False)
  v1 = vel[:,:,:,:,0]
  v2 = vel[:,:,:,:,1]
  v3 = vel[:,:,:,:,2]
  mom = dfile.Get("c.momentum", flatten=False)
  mass = dfile.Get("c.density", flatten=False)
  ener = dfile.Get("c.energy", flatten=False)
  cons = [mass, mom[:,:,:,:,0], mom[:,:,:,:,1], mom[:,:,:,:,2], ener]
  fd = dfile.Get("flux_divergence", flatten=False)
  st = dfile.Get("src_terms", flatten=False)

  fig, axes = plt.subplots(1, 5, figsize=(14,8))
  
  #for idx in range(5):
  #  myplot(np.fabs(fd[:,:,:,:,idx] + st[:,:,:,:,idx]), idx)
  #  myplot(np.fabs(cons[idx] / (fd[:,:,:,:,idx] + st[:,:,:,:,idx])), idx, vmin=-3, vmax=3)
  #myplot(density, 0, half=True, cbar=False)
  #myplot(np.fabs(v1), 1, half=True, cbar=False)
  #myplot(np.fabs(v2), 2, half=True, cbar=False)
  #myplot(np.fabs(v3), 3, half=True, cbar=False)
  #myplot(ug, 4, half=True, cbar=False)

  myplot(density, 0)
  #myplot(ug/density, 1)
  myplot(np.fabs(v1), 1)
  myplot(np.fabs(v2), 2)
  myplot(np.fabs(v3), 3)
  myplot(ug, 4)

  myplot(density0, 0, half=True)
  myplot(ug0/density0, 1, half=True)
  myplot(np.fabs(v10), 2, half=True)
  myplot(np.fabs(v20), 3, half=True)
  myplot(np.fabs(v30), 4, half=True)

  for idx in range(5):
    axes[idx].contour(xc[0,:,:], yc[0,:,:], density0[0,0,:,:].transpose(), [1.e-3], colors='k',
      linestyles='--')
  
  plt.savefig("frame_%08d.png" % n, bbox_inches='tight', dpi=dpi)
  plt.close()
