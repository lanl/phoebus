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

"""This is a python script for plotting 1d simulations that leverages the coords output so
it always gets the geometry correct and plots in x coordinates.
"""

from __future__ import print_function
from argparse import ArgumentParser
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt

# Assumes phdf in global python path
try:
    from parthenon_tools.phdf import phdf
except ModuleNotFoundError:
    from phdf import phdf

def plot_dump(filename, varname,
              savename = None,
              cbar_label = None,
              x1bounds=None,
              lidx=0,
              filename0 = None,
              log=True):

    overlay_initial = filename0 is not None

    data = phdf(filename)
    if overlay_initial:
      data0 = phdf(filename0)

    if savename is not None:
      matplotlib.use('Agg')

    q = data.Get(varname, False)
    if q is None:
      print("ERROR: variable not found!")
      sys.exit()
    if overlay_initial:
      q0 = data0.Get(varname, False)
      if q0 is None:
        print("ERROR: variable not found in initial dump!")
        sys.exit()
    NB = q.shape[0]

    if log:
        q = np.log10(np.abs(q))
        if overlay_initial:
          q0 = np.log10(np.abs(q0))

    # We have to play this stupid game because
    # outputs for variables of non-standard shape don't work
    coordname = "g.c.coord"
    coord = data.Get(coordname, False)
    using_phoebus_coords = True
    if coord is None:
      x = data.x
      using_phoebus_coords = False
    else:
      x = coord[:,1,:,:,:]

    # Aligned to cell centers if NONSTANDARD_SHAPE_IO_BROKEN
    # but whatever
    if x1bounds is None:
        x1bounds = [x.min(), x.max()]

    if cbar_label is None:
        cbar_label = varname
        if log:
            cbar_label = r'$\log_{10}$' + cbar_label

    fig = plt.figure()
    p = fig.add_subplot(111)
    for i in range(NB):
        if using_phoebus_coords:
          xplt = x[i,0,0,:]
        else:
          xplt = x[i,:]
        p.plot(xplt, q[i,0,0,:,lidx], color='r')
        if overlay_initial:
          p.plot(xplt, q0[i,0,0,:,lidx], color='k', linestyle='--')

    plt.xlim(x1bounds[0], x1bounds[1])

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    if savename is None:
      plt.show()
    else:
      plt.savefig(savename, dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot a 1d simulation snapshot.")
    parser.add_argument('--xbounds', type=float, nargs=2, default=None,
                        help='min and max bounds for x')
    parser.add_argument('-s','--saveprefix', type=str,
                        default="", help='Prefix for file save names')
    parser.add_argument('--linear',action='store_true',
                        help='Use linear, instead of log scale')
    parser.add_argument('-l', '--lidx', type=int, default=0,
                        help='Variable index')
    parser.add_argument('--pdf', action='store_true',
                        help="Save as pdf instead of png")
    parser.add_argument('varname', type=str, help='Variable to plot')
    parser.add_argument('files', type=str, nargs='+',
                        help='Files to take a snapshot of')
    parser.add_argument('--nplot', type=int, default=-1,
                        help='Which file to plot')
    parser.add_argument('--initial',action='store_true',
                        help='Overlay initial condition')
    args = parser.parse_args()


    log = not args.linear
    postfix = '.pdf' if args.pdf else '.png'
    savename = None
    if args.saveprefix is not "":
      savename = args.saveprefix + str(i).rjust(5,"0") + postfix
    filename0 = None
    if args.initial:
      filename0 = args.files[0]
    plot_dump(args.files[args.nplot], args.varname, savename,
              cbar_label = args.cbarlabel,
              x1bounds = args.xbounds,
              lidx = args.lidx,
              filename0 = filename0,
              log = log)

