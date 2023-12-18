#!/usr/bin/env python

# © 2023. Triad National Security, LLC. All rights reserved.  This
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

# This file was generated in part by OpenAI's GPT-4.

import argparse
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-average radial profiles from torus dumps")
    parser.add_argument(
        "filename", type=str, help="Pickle file containing time-averaged radial profiles (from tavg_torus_radial_profiles.py)"
    )
    parser.add_argument(
        "key", type=str, help="Profile to plot"
    )
    parser.add_argument(
        "--rmin", type=float, default=None, help="Minimum radial coordinate to plot"
    )
    parser.add_argument(
        "--rmax", type=float, default=None, help="Maximum radial coordinate to plot"
    )
    parser.add_argument(
        "--ymin", type=float, default=None, help="Minimum value to enforce on data"
    )
    parser.add_argument(
        "--ymax", type=float, default=None, help="Maximum value to enforce on data"
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to use logarithmic y axis"
    )
    # TODO(BRR) add geometric scaling with time averaging window
    args = parser.parse_args()

    with open(args.filename, 'rb') as handle:
        tavgs = pickle.load(handle)

    fig, ax = plt.subplots(1,1)

    for n, avg_key in enumerate(tavgs.keys()):
        if n == len(tavgs.keys()) - 1:
          continue
        col = cm.get_cmap("Spectral")(n / len(tavgs.keys()))
        #ax.plot(tavgs[avg_key]['r'], tavgs[avg_key][args.key], label=str(tavgs[avg_key]['tmin']), color=col)
        if args.log:
            ax.plot(tavgs[avg_key]['r'], -tavgs[avg_key][args.key], linestyle='--', color=col)
    ax.legend(loc=2)

    ax.set_xlabel('r')
    ax.set_ylabel(args.key)
    ax.set_xscale('log')

    ax.set_xlim([args.rmin, args.rmax])
    ax.set_ylim([args.ymin, args.ymax])

    if args.log:
        ax.set_yscale('log')

    plt.savefig("tavg_profile.png")
