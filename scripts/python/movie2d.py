#!/usr/bin/env python
# =========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

"""This is a python script for generating  2d movies (but only 2d for now)
that leverages the coords output so it always gets the geometry carrect and plots in
x,y coordinates.
"""

from __future__ import print_function
from argparse import ArgumentParser
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Assumes phdf in global python path
try:
    from parthenon_tools.phdf import phdf
except ModuleNotFoundError:
    from phdf import phdf

import plot_snap2d

from multiprocessing import Pool
import gc, os, shutil

# Set your ffmpeg executable here. If not available, set codec to None
codec = 'ffmpeg'

parser = ArgumentParser(description="Plot a 2d simulation snapshot.")
parser.add_argument('-n', '--nprocs', default=None,
                    help='Number of processor cores to use')
parser.add_argument('--plane', type=str, default='xz',
                    choices=['xy','xz'],
                    help='plane to project onto')
parser.add_argument('--xbounds', type=float, nargs=2, default=None,
                    help='min and max bounds for x')
parser.add_argument('--ybounds', type=float, nargs=2, default=None,
                    help='min and max bounds for y')
parser.add_argument('--cbarbounds', type=float, nargs=2, default=None,
                    help='min and max values of colorbar')
parser.add_argument('--cmap', type=str, default='viridis',
                    help='colormap')
parser.add_argument('--cbarlabel', type=str, default=None,
                    help='Color bar label')
parser.add_argument('-s','--savename', type=str,
                    default="movie.mp4",
                    help='Name to save movie as')
parser.add_argument('-l','--linear',action='store_true',
                    help='Use linear, instead of log scale')
parser.add_argument('--noclean',action='store_true',
                     help="Don't clean up the temporary directory when done")
parser.add_argument('varname', type=str, help='Variable to plot')
parser.add_argument('files', type=str, nargs='+',
                    help='Files to take a snapshot of')
args = parser.parse_args()
log = not args.linear
clean = not args.noclean

tmpdir = 'FRAMES'
try:
    shutil.rmtree(tmpdir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
os.mkdir(tmpdir)

def make_frame(stuff):
    fnam, savename = stuff
    print(savename)
    plot_snap2d.plot_dump(fnam, args.varname, savename,
                          cbar_label = args.cbarlabel,
                          x1bounds = args.xbounds,
                          x2bounds = args.ybounds,
                          cbarbounds = args.cbarbounds,
                          colormap = args.cmap,
                          log = log)
    return

filenames = args.files
savenames = [os.path.join(tmpdir,str(i).rjust(8,"0") + '.png') for i in range(len(filenames))]
p = Pool(processes = args.nprocs)
p.map(make_frame, zip(filenames, savenames))

if codec is not None:
  from subprocess import call
  if os.path.exists(args.savename):
      os.remove(args.savename)
  try:
      call([codec, '-i', os.path.join(tmpdir, '%08d.png'), args.savename])
      if clean:
          try:
              shutil.rmtree(tmpdir)
          except OSError as e:
              print("Error: %s - %s." % (e.filename, e.strerror))
  except:
      print("Failed to generate mp4 from frames. Leaving frames directory,",tmpdir)
