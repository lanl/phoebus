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

print(prs.max())
print(prs.min())



#Pg = dfile1.Get("pressure", flatten=False)
#vcon = dfile.Get("p.velocity", flatten=False)
#density = dfile1.Get("p.density", flatten=False)
#crho = dfile1.Get("c.density", flatten=False)
#ug = dfile1.Get("p.energy", flatten=False)
#fd = dfile1.Get("flux_divergence", flatten=False)
#st = dfile1.Get("src_terms", flatten=False)
#v1 = vcon[:,:,:,:,0]
#v2 = vcon[:,:,:,:,1]
#v3 = vcon[:,:,:,:,2]
#Bcon = dfile1.Get("p.bfield", flatten=False)
#flatgcov = dfile1.Get("g.c.gcov", flatten=False)
#alpha = dfile1.Get("g.c.alpha", flatten=False)
#gcov = np.zeros([nb,nz,ny,nx,4,4])
#for mu in range(4):
#  gcov[:,:,:,:,mu,0] = flatgcov[:,:,:,:,mu]
#  gcov[:,:,:,:,0,mu] = flatgcov[:,:,:,:,mu]
#for mu in range(1,4):
#  gcov[:,:,:,:,mu,1] = flatgcov[:,:,:,:,3+mu]
#  gcov[:,:,:,:,1,mu] = flatgcov[:,:,:,:,3+mu]
#for mu in range(2,4):
#  gcov[:,:,:,:,mu,2] = flatgcov[:,:,:,:,3+2+mu]
#  gcov[:,:,:,:,2,mu] = flatgcov[:,:,:,:,3+2+mu]
#gcov[:,:,:,:,3,3] = flatgcov[:,:,:,:,9]
#
#Bcov = np.zeros([nb,nz,ny,nx,3])
#vcov = np.zeros([nb,nz,ny,nx,3])
#for ii in range(3):
#  for jj in range(3):
#    Bcov[:,:,:,:,ii] += gcov[:,:,:,:,ii+1,jj+1]*Bcon[:,:,:,:,jj]
#    vcov[:,:,:,:,ii] += gcov[:,:,:,:,ii+1,jj+1]*vcon[:,:,:,:,jj]
#
#Bsq = np.zeros([nb,nz,ny,nx])
#Bdv = np.zeros([nb,nz,ny,nx])
#vsq = np.zeros([nb,nz,ny,nx])
#Gamma = np.zeros([nb,nz,ny,nx])
#for ii in range(3):
#  Bsq[:,:,:,:] += Bcon[:,:,:,:,ii]*Bcov[:,:,:,:,ii]
#  Bdv[:,:,:,:] += Bcon[:,:,:,:,ii]*vcov[:,:,:,:,ii]
#  vsq[:,:,:,:] += vcon[:,:,:,:,ii]*vcov[:,:,:,:,ii]
#Gamma[:,:,:,:] = 1./np.sqrt(1 - vsq[:,:,:,:])
#b0 = Gamma*Bdv/alpha
#bsq = (Bsq + alpha**2*b0**2)/Gamma**2
#beta = 2.*Pg/(bsq + 1.e-20)

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

#var = np.fabs(v1)
#vmin=-4
#vmax=0

#var = beta
#vmin = -2
#vmax = 2

mblw = 0.5

def myplot(axes, myvar, n, uselog=True, cmap='jet',label=None,vmin=None,vmax=None):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  #if isinstance(axes, (str,tuple,list)):
  #  ax = axes[n]
  #else:
  #  ax = axes
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

#fig, axes = plt.subplots(1, 1, figsize=(8,8))
#myplot(axes,ye,0,label='ye',vmin=0,vmax=1,uselog=False)

fig, axes = plt.subplots(1, 2, figsize=(8,8))
#myplot(axes,rho,0,label='rho',vmin=0.5*rho.min(),vmax=2.*rho.max(),uselog=False)
myplot(axes,ye0,0,label='Initial',vmin=0,vmax=1,uselog=False)
myplot(axes,ye,1,label='Final',vmin=0,vmax=1,uselog=False)
#myplot(axes,prs,2,label='prs',vmin=0.5*prs.min(),vmax=2*prs.max(),uselog=False)

print("min: ", ye.min())
print("max: ", ye.max())
print("mean: ", ye.mean())

if args.savefig:
  plt.savefig('frame_%08d.png' % args.nfinal, bbox_inches='tight', dpi=300)
else:
  plt.show()
