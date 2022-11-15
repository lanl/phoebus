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
import time
from enum import Enum
from phoebus_constants import cgs, scalefree
import phoebus_utils
from phoedf import phoedf

def get_torus_fluxes(dfile):

  fluxes = {}

  rho = dfile.GetRho()
  ucon = dfile.Getucon()
  gdet = dfile.gdet
  rho = dfile.GetRho()

  E = dfile.GetE()
  F = dfile.GetF()
  P = dfile.GetP()

  Nx1 = dfile.MeshBlockSize[0]
  Nx2 = dfile.MeshBlockSize[1]
  Nx3 = dfile.MeshBlockSize[2]

  x1max = 0.
  for b in range(dfile.NumBlocks):
    x1max = max(x1max, dfile.BlockBounds[b][1])
  xh = dfile.Params['geometry/xh']
  print(f'ximax : {x1max}')

  print(dfile.NumBlocks)
  mdot_in = 0.
  mdot_out = 0.
  Eg_in = 0.
  Eg_out = 0.
  Lg_in = 0.
  Lg_out = 0.
  Er_in = 0.
  Er_out = 0.
  Lr_in = 0.
  Lr_out = 0.
  phi = 0.
  for b in range(dfile.NumBlocks):
    print(f'{b}\n\n')
    blockBounds = dfile.BlockBounds[b]
    print(dfile.BlockBounds[b])
    dfile.GetCartesianBlockBounds(b)

    dx2 = (blockBounds[3] - blockBounds[2]) / dfile.MeshBlockSize[1]
    dx3 = (blockBounds[5] - blockBounds[4]) / dfile.MeshBlockSize[2]

    # Block contains event horizon
    if blockBounds[0] < xh and blockBounds[1] > xh:
      i_eh = 0 # TODO(BRR) interpolate ucon, rho to EH
      mdot_in += np.sum(dx2*dx3*gdet[b,:,:,i_eh]*ucon[b,1,:,:,i_eh]*rho[b,:,:,i_eh])

      for j in range(Nx2):
        for k in range(Nx3):
          Tmunu_concov = dfile.GetTmunu_concov(b,k,j,i_eh)
          Eg_in += -dx2*dx3*gdet[b,k,j,i_eh]*Tmunu_concov[1,0]
          Lg_in += dx2*dx3*gdet[b,k,j,i_eh]*Tmunu_concov[1,3]

      for j in range(Nx2):
        for k in range(Nx3):
          Rmunu_concov = dfile.GetRmunu_concov(b,k,j,i_eh)
          for ispec in range(dfile.NumSpecies):
            Er_in += -dx2*dx3*gdet[b,k,j,i_eh]*Rmunu_concov[1,0,ispec]
            Lr_in += dx2*dx3*gdet[b,k,j,i_eh]*Rmunu_concov[1,3,ispec]

    # Block contains outer boundary
    if np.fabs(blockBounds[1] - x1max) / x1max < 1.e-10:
      mdot_out += np.sum(dx2*dx3*gdet[b,:,:,-1]*ucon[b,1,:,:,-1]*rho[b,:,:,-1])

      for j in range(Nx2):
        for k in range(Nx3):
          Tmunu_concov = dfile.GetTmunu_concov(b,k,j,-1)
          Eg_out += -dx2*dx3*gdet[b,k,j,-1]*Tmunu_concov[1,0]
          Lg_out += dx2*dx3*gdet[b,k,j,-1]*Tmunu_concov[1,3]

      for j in range(Nx2):
        for k in range(Nx3):
          Rmunu_concov = dfile.GetRmunu_concov(b,k,j,-1)
          for ispec in range(dfile.NumSpecies):
            Er_out += -dx2*dx3*gdet[b,k,j,-1]*Rmunu_concov[1,0,ispec]
            Lr_out += dx2*dx3*gdet[b,k,j,-1]*Rmunu_concov[1,3,ispec]



  fluxes['mdot_in'] = mdot_in
  fluxes['mdot_edd'] = mdot_in * ( dfile.MassCodeToCGS / dfile.TimeCodeToCGS ) / dfile.GetMdotEddington()

  fluxes['mdot_out'] = mdot_out

  fluxes['Eg_in'] = Eg_in
  fluxes['Er_in'] = Er_in

  fluxes['Eg_out'] = Eg_out
  fluxes['Er_out'] = Er_out

  fluxes['Lg_in'] = Lg_in
  fluxes['Lr_in'] = Lr_in

  fluxes['Lg_out'] = Lg_out
  fluxes['Lr_out'] = Lr_out


  return fluxes

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Get fluxes from torus dump')
  parser.add_argument('files', type=str, nargs='+',
                        help='Files to take a snapshot of')
  args = parser.parse_args()

  logfile = open("fluxes_logfile.txt", "a")

  for n, fname in enumerate(args.files):
    print(f'Opening file {fname}... ', end='')
    dfile = phoedf(fname)
    print('done')


    fluxes = get_torus_fluxes(dfile)
    if n == 0:
      for flux in fluxes.keys():
        logfile.write(f'{flux} ')
      logfile.write('\n')
    print('Fluxes:')
    for flux in fluxes.keys():
      print(f'  {flux}: {fluxes[flux]}')
      logfile.write(f'{fluxes[flux]} ')
    logfile.write('\n')

  logfile.close()

