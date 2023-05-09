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

"""This is a python script for plotting 2d simulations (but only 2d for now)
that leverages the coords output so it always gets the geometry carrect and plots in
x,y coordinates.
"""

from __future__ import print_function
from argparse import ArgumentParser
import numpy as np
import warnings

import matplotlib

matplotlib.use("Agg")
try:
    import cmocean
except:
    warnings.warn("cmocean package not found. some colormaps unavailable.")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from multiprocessing import Pool

# Assumes phdf in global python path
try:
    from parthenon_tools.phdf import phdf
except ModuleNotFoundError:
    from phdf import phdf


def plot_dump(
    filename,
    varname,
    savename,
    plane="xz",
    cbar_label=None,
    x1bounds=None,
    x2bounds=None,
    cbarbounds=None,
    colormap="viridis",
    log=True,
):
    data = phdf(filename)
    print(data)
    time = data.Time

    q = data.Get(varname, False)
    NB = q.shape[0]

    if log:
        q = np.log10(np.abs(q))

    # We have to play this stupid game because
    # outputs for variables of non-standard shape don't work
    nG = data.NGhost
    coordname = "g.n.coord"
    coord = data.Get(coordname, False)
    z = coord[:, 3, :, nG - 1 : -1 - nG, nG - 1 : -1 - nG]
    y = coord[:, 2, :, nG - 1 : -1 - nG, nG - 1 : -1 - nG]
    x = coord[:, 1, :, nG - 1 : -1 - nG, nG - 1 : -1 - nG]

    if plane == "xz":
        rho = np.sqrt(x ** 2 + y ** 2)
        x = rho
        y = z

    if cbarbounds is not None:
        qmin, qmax = cbarbounds
    else:
        qmin = q.min()
        qmax = q.max()

    # Aligned to cell centers if NONSTANDARD_SHAPE_IO_BROKEN
    # but whatever
    if x1bounds is None:
        x1bounds = [x.min(), x.max()]
    if x2bounds is None:
        x2bounds = [y.min(), y.max()]

    if cbar_label is None:
        cbar_label = varname
        if log:
            cbar_label = r"$\log_{10}$" + cbar_label

    fig = plt.figure()
    p = fig.add_subplot(111, aspect=1)
    for i in range(NB):
        val = 100.0 * (q[i, 0, :, :] - 0.225)
        if len(val.shape) > 2:
            print("WARNING plotting the 0th index of multidimensional variable!")
            val = val[:, :, 0]

        mesh = p.pcolormesh(
            x[i, 0, :, :], y[i, 0, :, :], val[:, :], vmin=qmin, vmax=qmax, cmap=colormap
        )

    plt.xlim(x1bounds[0], x1bounds[1])
    plt.ylim(x2bounds[0], x2bounds[1])

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar(mesh, label=cbar_label)

    plt.title("t = %g" % time)

    plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot a 2d simulation snapshot.")
    parser.add_argument(
        "--plane",
        type=str,
        default="xz",
        choices=["xy", "xz"],
        help="plane to project onto",
    )
    parser.add_argument(
        "--xbounds", type=float, nargs=2, default=None, help="min and max bounds for x"
    )
    parser.add_argument(
        "--ybounds", type=float, nargs=2, default=None, help="min and max bounds for y"
    )
    parser.add_argument(
        "--cbarbounds",
        type=float,
        nargs=2,
        default=None,
        help="min and max values of colorbar",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="colormap")
    parser.add_argument("--cbarlabel", type=str, default=None, help="Color bar label")
    parser.add_argument(
        "-s", "--saveprefix", type=str, default="", help="Prefix for file save names"
    )
    parser.add_argument(
        "-l", "--linear", action="store_true", help="Use linear, instead of log scale"
    )
    parser.add_argument("--pdf", action="store_true", help="Save as pdf instead of png")
    parser.add_argument("--nproc", type=int, default=1, help="Number of CPUs to use")
    parser.add_argument("varname", type=str, help="Variable to plot")
    parser.add_argument(
        "files", type=str, nargs="+", help="Files to take a snapshot of"
    )
    args = parser.parse_args()
    log = not args.linear
    postfix = ".pdf" if args.pdf else ".png"

    def make_frame(pair):
        i, f = pair
        savename = args.saveprefix + str(i).rjust(5, "0") + postfix
        print(savename)
        plot_dump(
            f,
            args.varname,
            savename,
            plane=args.plane,
            cbar_label=args.cbarlabel,
            x1bounds=args.xbounds,
            x2bounds=args.ybounds,
            cbarbounds=args.cbarbounds,
            colormap=args.cmap,
            log=log,
        )

    p = Pool(processes=args.nproc)
    p.map(make_frame, enumerate(args.files))
