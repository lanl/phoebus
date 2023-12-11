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

import argparse
import numpy as np
import sys
import os
import phoebus_utils
from phoedf import phoedf

def get_torus_radial_profiles(dfile):

    profiles = {}

    gdet = dfile.gdet
    dx1 = dfile.dx1

    print(dx1)

    for b in range(dfile.NumBlocks):
        dx2 = dfile.Dx2[b]
        dx3 = dfile.Dx3[b]
        print(f"b: {b} dx2: {dx2} dx3: {dx3})")


    return profiles

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get radial profiles from torus dump")
    parser.add_argument(
        "filenames", type=str, nargs="+", help="Files to derive radial profiles for"
    )
    args = parser.parse_args()

    print("Phoebus GRMHD analysis script for generating radial profiles from dumps")
    print(f"  Number of files: {len(args.filenames)}")

    for filename in args.filenames:
        print(f"Opening file {os.path.basename(filename)}... ", end="", flush=True)
        dfile = phoedf(filename)
        print("done")

        profiles = get_torus_radial_profiles(dfile)

