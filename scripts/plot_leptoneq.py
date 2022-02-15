DUMP_NAMES = '/home/brryan/builds/phoebus/leptoneq.out1.*.phdf'

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os
from subprocess import call, DEVNULL
import glob
from parthenon_tools import phdf
import time
from enum import Enum

parser = argparse.ArgumentParser(description='Plot torus')
parser.add_argument('--nfinal', type=int, default=-1, help='dump to plot')
parser.add_argument('--savefig', type=bool, default=False, help='Whether to save figure')
args = parser.parse_args()

# Whether to plot meshblock boundaries
plot_meshblocks = True

nfinal = args.nfinal

dfnams = np.sort(glob.glob(DUMP_NAMES))
dfile0 = phdf.phdf(dfnams[0])
dfile1 = phdf.phdf(dfnams[nfinal])

nblocks = dfile0.NumBlocks

meshblocksize = dfile0.MeshBlockSize
nb = nblocks
nx = meshblocksize[0]
ny = meshblocksize[1]
nz = meshblocksize[2]

print("File: ", dfnams[nfinal], end="\n\n")

time = dfile1.Time
T_unit = dfile1.Params['eos/time_unit']
print("Time: ", time)
print("Time: ", (time*T_unit), " s")
print("Time: ", (time*T_unit*1.e3), " ms", end="\n\n")

print("Nblocks: ", nblocks)
print("  nx: %i" % nx + " ny: %i" % ny)
print("")

blockbounds = dfile0.BlockBounds
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

print("Variables:")
for var in dfile0.Variables:
  print("  " + var)
print("")

# Numblocks, nz, ny, nx
ye0 = dfile0.Get("p.ye", flatten=False)
ye = dfile1.Get("p.ye", flatten=False)
rho = dfile1.Get("p.density", flatten=False)
prs = dfile1.Get("p.energy", flatten=False)

mblw = 0.5

def myplot(axes, myvar, n, uselog=True, cmap='jet',label=None,vmin=None,vmax=None):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  ax = axes[n]
  for nb in range(nblocks):
    if uselog:
      im = ax.pcolormesh(xblock[nb,:,:], yblock[nb,:,:], np.log10(myvar[nb,0].transpose()),
        vmin=vmin, vmax=vmax, cmap=cmap)
    else:
      im = ax.pcolormesh(xblock[nb,:,:], yblock[nb,:,:], myvar[nb,0].transpose(),
        vmin=vmin, vmax=vmax, cmap=cmap)
      print(myvar[nb,0].max(), myvar[nb,0].min())
    if plot_meshblocks:
      ax.plot(xblock[nb,0,:], yblock[nb,0,:], color='k', linewidth=mblw, linestyle='--')
      ax.plot(xblock[nb,-1,:], yblock[nb,-1,:], color='k', linewidth=mblw, linestyle='--')
      ax.plot(xblock[nb,:,0], yblock[nb,:,0], color='k', linewidth=mblw, linestyle='--')
      ax.plot(xblock[nb,:,-1], yblock[nb,:,-1], color='k', linewidth=mblw, linestyle='--')
  ax.set_aspect('equal')
  ax.set_xlabel('x')
  ax.set_ylabel('y')

  if label is not None:
    ax.set_title(label)

  if n > 0:
    ax.set_yticklabels([])

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')

fig, axes = plt.subplots(1, 2, figsize=(8,8))
myplot(axes,ye0,0,label='Initial',vmin=0,vmax=1,uselog=False)
myplot(axes,ye,1,label='Final',vmin=0,vmax=1,uselog=False)

print("min: ", ye.min())
print("max: ", ye.max())
print("mean: ", ye.mean())

if args.savefig:
  plt.savefig('frame_%08d.png' % args.nfinal, bbox_inches='tight', dpi=300)
else:
  plt.show()
