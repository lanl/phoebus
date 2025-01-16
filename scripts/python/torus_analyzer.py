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
from multiprocessing import Pool
from phoebus_constants import cgs, scalefree
import phoebus_utils
from phoedf import phoedf

from get_torus_fluxes import get_torus_fluxes
from plot_torus_3d import plot_frame_from_phoedf

# Script to perform complete analysis of an output directory


def process_file(in_filename, fig_folder, log_folder, avg_folder):
    print(f"Processing {in_filename}")

    log_filename = (
        os.path.join(log_folder, in_filename.split(os.sep)[-1]).rstrip(".phdf") + ".log"
    )
    fig_filename = (
        os.path.join(fig_folder, in_filename.split(os.sep)[-1]).rstrip(".phdf") + ".png"
    )

    # Skip if already done and overwrite not requested
    if (
        not args.overwrite
        and os.path.exists(log_filename)
        and os.path.exists(fig_filename)
    ):
        return

    # Load data
    dfile = phoedf(in_filename)

    # Create log file
    if not (args.overwrite == False and os.path.exists(log_filename)):
        fluxes = get_torus_fluxes(dfile)

        with open(log_filename, "w") as log_file:
            log_file.write("# ")
            for flux in fluxes:
                log_file.write(flux + " ")
            log_file.write("\n")
            for flux in fluxes:
                log_file.write(str(fluxes[flux]) + " ")
            log_file.write("\n")

    # Create figure
    if not (args.overwrite == False and os.path.exists(fig_filename)):
        plot_frame_from_phoedf(
            dfile,
            0,
            True,
            None,
            args.rout,
            coords="cartesian",
            draw_bh=True,
            custom_name=fig_filename,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze torus simulation")

    parser.add_argument(
        "--no_figs", type=bool, default=False, help="Whether to disable image creation"
    )
    parser.add_argument(
        "--no_logs", type=bool, default=False, help="Whether to disable log creation"
    )
    parser.add_argument(
        "--no_avgs",
        type=bool,
        default=False,
        help="Whether to disable averages creation",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing outputs"
    )

    parser.add_argument(
        "--rout", type=float, default=40.0, help="Outer radius to plot for images"
    )

    parser.add_argument(
        "--serial",
        dest="serial",
        default=False,
        action="store_true",
        help="Run in serial",
    )
    parser.add_argument(
        "--nproc",
        dest="nproc",
        default=None,
        type=int,
        help="Number of parallel processes to use. Defaults to all available cores.",
    )

    parser.add_argument(
        "directory", type=str, help="Directory containing simulation output"
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default="torus.out1",
        help="Name of problem for output file",
    )

    args = parser.parse_args()

    matplotlib.use("Agg")

    assert os.path.isdir(args.directory)

    base_filename = args.problem_name + "*.phdf"
    filenames = np.sort(glob.glob(os.path.join(args.directory, base_filename)))

    fig_folder = os.path.join(args.directory, "figures")
    log_folder = os.path.join(args.directory, "logs")
    avg_folder = os.path.join(args.directory, "averages")

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(avg_folder):
        os.makedirs(avg_folder)

    if args.serial:
        for in_filename in filenames:
            process_file(in_filename, fig_folder, log_folder, avg_folder)
    else:
        p = Pool(processes=args.nproc)
        from itertools import repeat

        p.starmap(
            process_file,
            zip(filenames, repeat(fig_folder), repeat(log_folder), repeat(avg_folder)),
        )
