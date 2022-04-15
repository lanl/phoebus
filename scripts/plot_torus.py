#PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'
#PHDF_PATH = '/home/brryan/github/phoebus/external/parthenon/scripts/python/'
#DUMP_NAMES = '/home/brryan/builds/phoebus/torus.out1.*.phdf'
DUMP_NAMES = 'torus.out1.*.phdf'

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os
from subprocess import call, DEVNULL
import glob
#sys.path.append(PHDF_PATH)
#import phdf
from parthenon_tools import phdf
import time
from enum import Enum

#plot = "mks"
plot = "cartesian"

# Outer radius to plot or None
rmax = 40
#rmax = None

parser = argparse.ArgumentParser(description='Plot torus')
parser.add_argument('--nfinal', type=int, default=-1, help='dump to plot')
parser.add_argument('--savefig', type=bool, default=False, help='Whether to save figure')
args = parser.parse_args()

# Whether to plot meshblock boundaries
plot_meshblocks = True
h_ = 0.3
a = 0.9375
rh = 1. + np.sqrt(1. - a*a)

nfinal = args.nfinal

dfnams = np.sort(glob.glob(DUMP_NAMES))
#dfnam = dfnams[nfinal]
dfile = phdf.phdf(dfnams[0])
dfile1 = phdf.phdf(dfnams[nfinal])

nblocks = dfile.NumBlocks

meshblocksize = dfile.MeshBlockSize
nb = nblocks
nx = meshblocksize[0]
ny = meshblocksize[1]
nz = meshblocksize[2]

print("File: ", dfnams[nfinal], end="\n\n")

time = dfile1.Time
print("Time: ", time, end="\n\n")

print("Nblocks: ", nblocks)
print("  nx: %i" % nx + " ny: %i" % ny)
print("")

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

print("Variables:")
for var in dfile.Variables:
  print("  " + var)
print("")

# Numblocks, nz, ny, nx
Pg = dfile1.Get("pressure", flatten=False)
#bfield = dfile.Get("p.bfield", flatten=False)
vcon = dfile.Get("p.velocity", flatten=False)
density = dfile1.Get("p.density", flatten=False)
crho = dfile1.Get("c.density", flatten=False)
ug = dfile1.Get("p.energy", flatten=False)
fd = dfile1.Get("flux_divergence", flatten=False)
st = dfile1.Get("src_terms", flatten=False)
v1 = vcon[:,:,:,:,0]
v2 = vcon[:,:,:,:,1]
v3 = vcon[:,:,:,:,2]
Bcon = dfile1.Get("p.bfield", flatten=False)
flatgcov = dfile1.Get("g.c.gcov", flatten=False)
alpha = dfile1.Get("g.c.alpha", flatten=False)
gcov = np.zeros([nb,nz,ny,nx,4,4])

def flatten(m,n):
    ind = [[0,1,3,6],[1,2,4,7],[3,4,5,8],[6,7,8,9]]
    return ind[m][n]

for mu in range(4):
  gcov[:,:,:,:,mu,0] = flatgcov[:,:,:,:,flatten(mu,0)]
  gcov[:,:,:,:,0,mu] = flatgcov[:,:,:,:,flatten(0,mu)]
for mu in range(1,4):
  gcov[:,:,:,:,mu,1] = flatgcov[:,:,:,:,flatten(mu,1)]
  gcov[:,:,:,:,1,mu] = flatgcov[:,:,:,:,flatten(1,mu)]
for mu in range(2,4):
  gcov[:,:,:,:,mu,2] = flatgcov[:,:,:,:,flatten(mu,2)]
  gcov[:,:,:,:,2,mu] = flatgcov[:,:,:,:,flatten(2,mu)]
gcov[:,:,:,:,3,3] = flatgcov[:,:,:,:,flatten(3,3)]

Bcov = np.zeros([nb,nz,ny,nx,3])
vcov = np.zeros([nb,nz,ny,nx,3])
for ii in range(3):
  for jj in range(3):
    Bcov[:,:,:,:,ii] += gcov[:,:,:,:,ii+1,jj+1]*Bcon[:,:,:,:,jj]
    vcov[:,:,:,:,ii] += gcov[:,:,:,:,ii+1,jj+1]*vcon[:,:,:,:,jj]

Bsq = np.zeros([nb,nz,ny,nx])
Bdv = np.zeros([nb,nz,ny,nx])
vsq = np.zeros([nb,nz,ny,nx])
Gamma = np.zeros([nb,nz,ny,nx])
for ii in range(3):
  Bsq[:,:,:,:] += Bcon[:,:,:,:,ii]*Bcov[:,:,:,:,ii]
  Bdv[:,:,:,:] += Bcon[:,:,:,:,ii]*vcov[:,:,:,:,ii]
  vsq[:,:,:,:] += vcon[:,:,:,:,ii]*vcov[:,:,:,:,ii]
Gamma[:,:,:,:] = 1./np.sqrt(1 - vsq[:,:,:,:])
b0 = Gamma*Bdv/alpha
bsq = (Bsq + alpha**2*b0**2)/Gamma**2
beta = 2.*Pg/(bsq + 1.e-20)

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
#var = ug
vmin = -5
vmax = 0

#var1 = dfile1.Get("p.density", flatten=False)
#var1 = dfile1.Get("p.energy", flatten=False)
var1 = density

#var = np.fabs(v1)
#vmin=-4
#vmax=0

#var = beta
#vmin = -2
#vmax = 2

mblw = 0.5

def myplot(myvar, n, vmin=vmin, vmax=vmax, uselog=True, cmap='jet',label=None):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  ax = axes[n]
  #ax = axes
  for nb in range(nblocks):
    if plot == "mks":
      im = ax.pcolormesh(xblock[nb,:,:], yblock[nb,:,:], np.log10(myvar[nb,0].transpose()),
        vmin=vmin, vmax=vmax, cmap=cmap)
    elif plot == "cartesian":
      if uselog:
        im = ax.pcolormesh(x[nb,:,:], y[nb,:,:], np.log10(myvar[nb,0].transpose()),
          vmin=vmin, vmax=vmax, cmap=cmap)
      else:
        im = ax.pcolormesh(x[nb,:,:], y[nb,:,:], myvar[nb,0].transpose(),
          vmin=vmin, vmax=vmax, cmap=cmap)

      if plot_meshblocks:
        ax.plot(x[nb,0,:], y[nb,0,:], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[nb,-1,:], y[nb,-1,:], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[nb,:,0], y[nb,:,0], color='k', linewidth=mblw, linestyle='--')
        ax.plot(x[nb,:,-1], y[nb,:,-1], color='k', linewidth=mblw, linestyle='--')
      if rmax is not None:
        ax.set_xlim([0,rmax])
        ax.set_ylim([-rmax,rmax])
    else:
      print("Plotting coordinates \"" + plot + "\" unknown")
      sys.exit()
  if plot == "cartesian":
    ax.set_aspect('equal')
  ax.set_xlabel('x')
  ax.set_ylabel('y')

  # Draw black hole
  bh = plt.Circle((0, 0), rh, color='k')
  ax.add_patch(bh)

  if label is not None:
    ax.set_title(label)

  if n > 0:
    ax.set_yticklabels([])

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')

fig, axes = plt.subplots(1, 2, figsize=(8,8))
myplot(var1,0,label='density')
myplot(beta,1,vmin=-3,vmax=3,uselog=True,cmap='RdBu',label='plasma beta')

if args.savefig:
  plt.savefig('frame_%08d.png' % args.nfinal, bbox_inches='tight')
else:
  plt.show()
