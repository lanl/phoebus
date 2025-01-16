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
from phoebus_constants import cgs, scalefree

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot fluxes logfile")
    parser.add_argument("filename", type=str, help="Fluxes logfile to plot")
    args = parser.parse_args()

    # matplotlib.use('Agg')

    figsize = (10, 8)

    datafile = np.loadtxt(args.filename, skiprows=1)

    gcolor = "tab:blue"
    rcolor = "tab:orange"

    data = {}
    data["time"] = datafile[:, 0]
    data["mdot"] = datafile[:, 1]
    data["egdot"] = datafile[:, 4]
    data["erdot"] = datafile[:, 5]
    data["lgdot"] = datafile[:, 8]
    data["lrdot"] = datafile[:, 9]

    data["mdot_out"] = datafile[:, 3]
    data["egdot_out"] = datafile[:, 6]
    data["erdot_out"] = datafile[:, 7]
    data["lgdot_out"] = datafile[:, 10]
    data["lrdot_out"] = datafile[:, 11]

    # Plot inward fluxes
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax = axes[0]
    ax.plot(data["time"], -data["mdot"], color=gcolor, label="MHD")
    ax.plot(data["time"], data["mdot"], color=gcolor, linestyle="--")
    ax.plot([0], [0], color=rcolor, label="Radiation")
    ax.legend(loc=2)
    ax.set_ylabel("$\\dot{M}$")

    ax = axes[1]
    ax.plot(data["time"], -data["egdot"], color=gcolor)
    ax.plot(data["time"], -data["erdot"], color=rcolor)
    ax.plot(data["time"], data["egdot"], color=gcolor, linestyle="--")
    ax.plot(data["time"], data["erdot"], color=rcolor, linestyle="--")
    ax.set_ylabel("$\\dot{E}$")

    ax = axes[2]
    ax.plot(data["time"], -data["lgdot"], color=gcolor)
    ax.plot(data["time"], -data["lrdot"], color=rcolor)
    ax.plot(data["time"], data["lgdot"], color=gcolor, linestyle="--")
    ax.plot(data["time"], data["lrdot"], color=rcolor, linestyle="--")
    ax.set_ylabel("$\\dot{L}$")

    for n, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_xlim([0, None])
        ax.set_ylim([1.0e-4, 1.0e1])
        if n < len(axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$t c^3 / (G M)$")

    plt.suptitle("Inflow")

    plt.savefig("inflow.png", dpi=300, bbox_inches="tight")

    plt.show()

    plt.close()

    # Plot outward fluxes
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax = axes[0]
    ax.plot(data["time"], data["mdot_out"], color=gcolor, label="MHD")
    ax.plot(data["time"], -data["mdot_out"], color=gcolor, linestyle="--")
    ax.set_ylabel("$\\dot{M}$")
    ax.plot([0], [0], color=rcolor, label="Radiation")
    ax.legend(loc=2)

    ax = axes[1]
    ax.plot(data["time"], data["egdot_out"], color=gcolor)
    ax.plot(data["time"], data["erdot_out"], color=rcolor)
    ax.plot(data["time"], -data["egdot_out"], color=gcolor, linestyle="--")
    ax.plot(data["time"], -data["erdot_out"], color=rcolor, linestyle="--")
    ax.set_ylabel("$\\dot{E}$")

    ax = axes[2]
    ax.plot(data["time"], data["lgdot_out"], color=gcolor)
    ax.plot(data["time"], data["lrdot_out"], color=rcolor)
    ax.plot(data["time"], -data["lgdot_out"], color=gcolor, linestyle="--")
    ax.plot(data["time"], -data["lrdot_out"], color=rcolor, linestyle="--")
    ax.set_ylabel("$\\dot{L}$")

    for n, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_xlim([0, None])
        ax.set_ylim([1.0e-5, 1.0e0])
        if n < len(axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$t c^3 / (G M)$")

    plt.suptitle("Outflow")

    plt.savefig("outflow.png", dpi=300, bbox_inches="tight")

    plt.show()
