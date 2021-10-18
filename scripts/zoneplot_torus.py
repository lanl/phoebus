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

iplt = 138-4
jplt = 186-4

# Whether to plot meshblock boundaries
plot_meshblocks = True
h_ = 0.3

#xmin = 0
#xmax = 12
#ymin = -10
#ymax = 10

xmin = 0
xmax = 40
ymin = -40
ymax = 40

nfinal = -1
#nfinal = 1
#nfinal = 15
mblw = 0.5

def myplot(myvar, n, vmin=-4, vmax=0, uselog=True, cmap='jet'):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  ax = axes[n]
  #ax = axes
  for n in range(nblocks):
    if plot == "mks":
      im = ax.pcolormesh(xblock[n,:,:], yblock[n,:,:], np.log10(myvar[n,0].transpose()),
        vmin=vmin, vmax=vmax, cmap=cmap)
    elif plot == "cartesian":
      if uselog:
        im = ax.pcolormesh(x[n,:,:], y[n,:,:], np.log10(myvar[n,0].transpose()),
          vmin=vmin, vmax=vmax, cmap=cmap)
        #im = ax.pcolormesh(x[n,:,:], y[n,:,:], np.log10(1-fail[n,0]).transpose())
        # Plot active point
        ax.plot([x[n,iplt,jplt]], [y[n,iplt,jplt]], marker='.', markersize=14,
        color='k')
        ax.plot([x[n,iplt,jplt]], [y[n,iplt,jplt]], marker='.', markersize=12,
        color='r')
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
      ax.set_xlim([xmin,xmax])
      ax.set_ylim([ymin,ymax])
    else:
      print("Plotting coordinates \"" + plot + "\" unknown")
      sys.exit()
  if plot == "cartesian":
    ax.set_aspect('equal')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')

dfnams = np.sort(glob.glob(DUMP_NAMES))
#dfnam = dfnams[nfinal]
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
# Plot initial torus and show active zone
density = dfile.Get("p.density", flatten=False)
fig, axes = plt.subplots(1, 1, figsize=(6,8))
axes = [axes]
myplot(density[:,:,:,:], 0)

data = {}
data['prho'] = np.zeros(len(dfnams))
data['v1'] = np.zeros(len(dfnams))
data['v2'] = np.zeros(len(dfnams))
data['v3'] = np.zeros(len(dfnams))
data['pener'] = np.zeros(len(dfnams))
data['crho'] = np.zeros(len(dfnams))
data['mom1'] = np.zeros(len(dfnams))
data['mom2'] = np.zeros(len(dfnams))
data['mom3'] = np.zeros(len(dfnams))
data['cener'] = np.zeros(len(dfnams))

t = []

for n, dfnam in enumerate(dfnams):
  print(f'Loading file {n+1} of {len(dfnams)}')
  dfile = phdf.phdf(dfnam)
  t.append(dfile.Time)
  data['prho'][n] = dfile.Get("p.density", flatten=False)[0,0,jplt,iplt]
  v = dfile.Get("p.velocity", flatten=False)
  data['v1'][n] = v[0,0,jplt,iplt,0]
  data['v2'][n] = v[0,0,jplt,iplt,1]
  data['v3'][n] = v[0,0,jplt,iplt,2]
  data['pener'][n] = dfile.Get("p.energy", flatten=False)[0,0,jplt,iplt]
  data['crho'][n] = dfile.Get("c.density", flatten=False)[0,0,jplt,iplt]
  mom = dfile.Get("c.momentum", flatten=False)
  data['mom1'][n] = mom[0,0,jplt,iplt,0]
  data['mom2'][n] = mom[0,0,jplt,iplt,1]
  data['mom3'][n] = mom[0,0,jplt,iplt,2]
  data['cener'][n] = dfile.Get("c.energy", flatten=False)[0,0,jplt,iplt]
  t[n] = n

print(data)
print(t)

def axisrhs(ax):
  ax.yaxis.tick_right()

fig, axes = plt.subplots(5, 2, figsize=(8,8))
axes[0,0].plot(t, data['prho'], label='prho', marker='.')
axes[1,0].plot(t, data['v1'], label='v1', marker='.')
axes[2,0].plot(t, data['v2'], label='v2', marker='.')
axes[3,0].plot(t, data['v3'], label='v3', marker='.')
axes[4,0].plot(t, data['pener'], label='pener', marker='.')
axes[0,1].plot(t, data['crho'], label='crho', marker='.')
axes[1,1].plot(t, data['mom1'], label='mom1', marker='.')
axes[2,1].plot(t, data['mom2'], label='mom2', marker='.')
axes[3,1].plot(t, data['mom3'], label='mom3', marker='.')
axes[4,1].plot(t, data['cener'], label='cener', marker='.')
axisrhs(axes[0,1])
axisrhs(axes[1,1])
axisrhs(axes[2,1])
axisrhs(axes[3,1])
axisrhs(axes[4,1])
for axis in axes:
  for ax in axis:
    ax.legend(loc=2)
    ax.set_xlim([t[0], t[-1]])
    ax.set_xticklabels([])
    ax.set_yscale('log')

plt.show()
sys.exit()

# Numblocks, nz, ny, nx
Pg = dfile.Get("pressure", flatten=False)
#bfield = dfile.Get("p.bfield", flatten=False)
vel = dfile.Get("p.velocity", flatten=False)
density = dfile.Get("p.density", flatten=False)
ug = dfile.Get("p.energy", flatten=False)
fd = dfile1.Get("flux_divergence", flatten=False)
st = dfile1.Get("src_terms", flatten=False)
v1 = vel[:,:,:,:,0]
v2 = vel[:,:,:,:,1]
v3 = vel[:,:,:,:,2]
fail = dfile.Get("fail")
#b1 = bfield[:,:,:,:,0]
#b2 = bfield[:,:,:,:,1]
#b3 = bfield[:,:,:,:,2]
#b2 = b1*b1 + b2*b2 + b3*b3
#beta = 2*Pg/(b2 + 1.e-100)

#fig = plt.figure()
#ax = plt.gca()
#ax.plot(density[3,0,:,64])
#print(density[3,:,:,64])
#plt.show()
#sys.exit()

var = density
vmin = -8
vmax = 0

var1 = dfile1.Get("p.density", flatten=False)
fail = dfile1.Get("fail", flatten=False)

#var = np.fabs(v1)
#vmin=-4
#vmax=0

#var = beta
#vmin = -2
#vmax = 2

  #plt.colorbar(im, label='density')
#myplot(var,0)


#fig, axes = plt.subplots(1,5,figsize=(10,10))
#for idx in range(5):
#  myplot(np.fabs(fd[:,:,:,:,idx] + st[:,:,:,:,idx]), idx)
#plt.show()
#sys.exit()


myfd = fd[:,:,:,:,1]
vmin = -1.e0
vmax = -vmin
idx = 0
#print(np.fabs(fd[:,:,:,:,idx] + st[:,:,:,:,idx]).max())
fig, axes = plt.subplots(1, 3, figsize=(14,8))
#myplot(fd[:,:,:,:,idx],0,vmin=vmin,vmax=vmax,uselog=False,cmap='RdBu')
#myplot(st[:,:,:,:,idx],1,vmin=vmin,vmax=vmax,uselog=False,cmap='RdBu')
myplot(var[:,:,:,:],0)
myplot(var1[:,:,:,:],1)
#myplot(fd[:,:,:,:,idx]+st[:,:,:,:,idx],1,vmin=vmin,vmax=vmax,uselog=False,cmap='RdBu')
#myplot(var1,1)
#varmax = var.max()
myplot(2*(var1-var)/(var1+var),2,vmin=-1,vmax=1,uselog=False,cmap='RdBu')
#print((2*(var1-var)/(var1+var))[0,0,64,100])
plt.show()
plt.savefig("torus_snapshot.png", bbox_inches='tight')
