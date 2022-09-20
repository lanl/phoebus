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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_frame(ifname, fname, savefig, geomfile=None, rlim=40):
  print(fname)

  cmap_uniform = 'viridis'
  cmap_diverging = 'RdYlBu'

  dfile = phoedf(fname)

  rad_active = dfile.Params['radiation/active']

  a = dfile.Params['geometry/a']
  hslope = dfile.Params['geometry/h']

  if rad_active:
    fig, axes = plt.subplots(2, 4, figsize=(10,8))
  else:
    fig, axes = plt.subplots(2, 2, figsize=(6,8))

  if geomfile is None:
    geomfile = dfile

  nblocks = geomfile.NumBlocks
  nx = geomfile.MeshBlockSize[0]
  ny = geomfile.MeshBlockSize[1]
  nz = geomfile.MeshBlockSize[2]

  time = dfile.Time

  # Get pcolormesh grid for each block
  x1block = np.zeros([nblocks, nx+1, ny+1])
  x2block = np.zeros([nblocks, nx+1, ny+1])
  for b in range(nblocks):
    for i in range(nx+1):
      for j in range(ny+1):
        dx1 = (geomfile.BlockBounds[b][1] - geomfile.BlockBounds[b][0])/nx
        dx2 = (geomfile.BlockBounds[b][3] - geomfile.BlockBounds[b][2])/ny
        x1block[b,i,j] = geomfile.BlockBounds[b][0] + i*dx1
        x2block[b,i,j] = geomfile.BlockBounds[b][2] + j*dx2
  rblock = np.exp(x1block)
  thblock = np.pi*x2block + ((1. - hslope)/2.)*np.sin(2.*np.pi*x2block)
  xblock = rblock*np.sin(thblock)
  yblock = rblock*np.cos(thblock)

  ax = axes[0,0]
  ldensity = np.log10(np.clip(dfile.Get("p.density", flatten=False), 1.e-20, None))
  for b in range(nblocks):
    im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], ldensity[b,0,:,:].transpose(), vmin=-5, vmax=0,
    cmap=cmap_uniform)
  div = make_axes_locatable(ax)
  cax = div.append_axes('right', size="5%", pad = 0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  ax.set_aspect('equal')
  ax.set_title(r'$\log_{10}~\rho$')

  ax = axes[0,1]
  Gamma = dfile.GetGamma()
  for b in range(nblocks):
    im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], Gamma[b,0,:,:].transpose(), vmin=1, vmax=5,
    cmap=cmap_uniform)
  div = make_axes_locatable(ax)
  cax = div.append_axes('right', size="5%", pad = 0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  ax.set_aspect('equal')
  ax.set_title(r'$\Gamma$')

  if rad_active:

    ax = axes[0,2]
    Pg = dfile.GetPg()
    Pm = np.clip(dfile.GetPm(), 1.e-20, 1.e20)
    lbeta = np.log10(Pg/Pm)
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lbeta[b,0,:,:].transpose(), vmin=-2, vmax=2,
        cmap=cmap_diverging)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~\beta_{\rm M}$')

    ax = axes[0,3]
    lTg = np.log10(dfile.GetTg())
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lTg[b,0,:,:].transpose(), vmin=5, vmax=10,
        cmap=cmap_diverging)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~T_{\rm g}~({\rm K})$')

    ax = axes[1,0]
    lJ = np.log10(np.clip(dfile.Get("r.p.J", flatten=False), 1.e-20, None))
    #lJ = np.log10(np.clip(dfile.Get("r.c.E", flatten=False), 1.e-20, None))
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lJ[b,0,:,:].transpose(), vmin=-5, vmax=0,
      cmap=cmap_uniform)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~J$')

    ax = axes[1,1]
    lxi = np.log10(dfile.GetXi())
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lxi[b,0,:,:].transpose(), vmin=-3, vmax=0,
      cmap=cmap_uniform)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~\xi$')

    ax = axes[1,2]
    Pg = dfile.GetPg()
    Pr = dfile.GetPr()
    lbetar = np.log10(Pg/Pr)
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lbetar[b,0,:,:].transpose(), vmin=-2, vmax=2,
      cmap=cmap_diverging)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~\beta_{\rm R}$')

    ax = axes[1,3]
    ltau = np.log10(dfile.GetTau())
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], ltau[b,0,:,:].transpose(), vmin=-3, vmax=3,
      cmap=cmap_diverging)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~\tau_{\rm zone}$')

  else:
    ax = axes[1,0]
    lenergy = np.log10(np.clip(dfile.Get("p.energy", flatten=False), 1.e-20, None))
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lenergy[b,0,:,:].transpose(), vmin=-5, vmax=0,
      cmap=cmap_uniform)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~u$')

    ax = axes[1,1]
    Pg = dfile.GetPg()
    Pm = np.clip(dfile.GetPm(), 1.e-20, 1.e20)
    lbeta = np.log10(Pg/Pm)
    for b in range(nblocks):
      im = ax.pcolormesh(xblock[b,:,:], yblock[b,:,:], lbeta[b,0,:,:].transpose(), vmin=-2, vmax=2,
        cmap=cmap_diverging)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size="5%", pad = 0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    ax.set_title(r'$\log_{10}~\beta_{\rm M}$')

  for tmp_ax in axes:
    for ax in tmp_ax:
      ax.set_xlim([0, rlim])
      ax.set_ylim([-rlim, rlim])

  plt.suptitle("Time = %g M" % dfile.Time)

  fig.tight_layout()

  #if savefig:
  savename = str(ifname).rjust(5,"0") + '.png'
  plt.savefig(savename, dpi=300, bbox_inches='tight')
  plt.close(fig)
 # else:
 #   plt.show()

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Plot neutrino thermalization')
  #parser.add_argument('-f', '--files', dest='files', nargs='*', default=['torus.out1.*.phdf'], help='List of input Parthenon hdf files to plot')
  parser.add_argument('-gf', '--geometry_file', dest='geometry_file', default=None, help='Optional dump file containing the time-independent (no AMR) geometry data')
  parser.add_argument('--savefig', type=bool, default=False, help='Whether to save figure')
  parser.add_argument('--numin', type=float, default=1.e-4, help='Minimum frequency (Hz)')
  parser.add_argument('--numax', type=float, default=1.e2, help='Maximum frequency (Hz)')
  parser.add_argument('--nnu', type=int, default=100, help='Number of frequency support points')
  parser.add_argument('files', type=str, nargs='+',
                        help='Files to take a snapshot of')
  args = parser.parse_args()

  #if args.savefig:
  matplotlib.use('Agg')

  for i, fname in enumerate(args.files):
    plot_frame(i, fname, args.savefig)
