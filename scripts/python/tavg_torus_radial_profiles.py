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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Time-average radial profiles from torus dumps"
    )
    parser.add_argument(
        "filenames", type=str, nargs="+", help="Files to derive radial profiles for"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing outputs"
    )
    parser.add_argument(
        "--tmin", type=float, default=0, help="Minimum time to begin averaging"
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Time window over which to separate averages",
    )
    parser.add_argument("--dt_sim", type=float, required=True, help="Dump cadence")
    parser.add_argument(
        "--tbins", type=float, nargs="+", help="Time-averaging window bin edges to use"
    )
    args = parser.parse_args()

    print("Phoebus GRMHD analysis script for time-averaging radial profiles from dumps")
    print(f"  Number of files:      {len(args.filenames)}")
    print(f"  Overwrite?            {args.overwrite}")
    print(f"  Time averaging bins:  {args.tbins}")

    using_custom_tavg = args.tbins is not None

    tavgs = {}
    custom_tavgs = {}

    def get_custom_avg_idx(t):
        for nedge in range(len(args.tbins) - 1):
            if t >= args.tbins[nedge] and t < args.tbins[nedge + 1]:
                return nedge
        return -1

    keys_to_tavg = [
        "r",
        "F_M",
        "F_M_in",
        "F_M_out",
        "F_Eg",
        "F_Eg_out",
        "F_Eg_in",
        "F_Pg",
        "F_Pg_out",
        "F_Pg_in",
        "F_Lg",
        "F_Lg_out",
        "F_Lg_in",
        "beta",
        "vconr",
        "vcovr",
        "vconr_out",
        "vconr_in"
    ]

    for n, filename in enumerate(args.filenames):
        base_directory = os.path.dirname(filename)
        base_filename = os.path.basename(filename)
        print(f"Processing file {base_filename}")

        # Overly specific way to get time
        try:
            nfile = int(base_filename[11:19])
        except ValueError:
            print(f"  Can't extract dumpfile index; skipping!")
            continue
        tfile = args.dt_sim * nfile

        avg_idx = int(tfile / args.dt)
        avg_key = str(avg_idx)

        if using_custom_tavg:
            custom_avg_idx = get_custom_avg_idx(tfile)
            custom_avg_key = str(custom_avg_idx)

        if avg_key not in tavgs.keys():
            tavgs[avg_key] = {}
            tavgs[avg_key]["nfiles"] = 0
            tavgs[avg_key]["tmin"] = avg_idx * args.dt
            tavgs[avg_key]["dt"] = args.dt

        if using_custom_tavg and custom_avg_key not in custom_tavgs.keys():
            custom_tavgs[custom_avg_key] = {}
            custom_tavgs[custom_avg_key]["nfiles"] = 0
            custom_tavgs[custom_avg_key]["tmin"] = avg_idx * args.dt
            custom_tavgs[custom_avg_key]["dt"] = args.dt

        data_indices = {}

        with open(filename, "r") as profile:
            lines = profile.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                for key in keys_to_tavg:
                    if line == key:
                        data_indices[key] = idx + 1

            # Initialize data if necessary
            ndata = len(
                np.fromstring(
                    lines[data_indices[keys_to_tavg[0]]].strip(), dtype=float, sep=" "
                )
            )
            if tavgs[avg_key]["nfiles"] == 0:
                for key in keys_to_tavg:
                    tavgs[avg_key][key] = np.zeros(ndata)
            if using_custom_tavg and custom_tavgs[custom_avg_key]["nfiles"] == 0:
                for key in keys_to_tavg:
                    custom_tavgs[custom_avg_key][key] = np.zeros(ndata)

            # Sum data
            for key in keys_to_tavg:
                tavgs[avg_key][key] += np.fromstring(
                    lines[data_indices[key]].strip(), dtype=float, sep=" "
                )
                if using_custom_tavg:
                    custom_tavgs[custom_avg_key][key] += np.fromstring(
                        lines[data_indices[key]].strip(), dtype=float, sep=" "
                    )

            # Record number of files in this bin for averaging
            tavgs[avg_key]["nfiles"] += 1

    # Normalize
    for avg_key in tavgs.keys():
        for key in keys_to_tavg:
            tavgs[avg_key][key] /= tavgs[avg_key]["nfiles"]
    if using_custom_tavg:
        for custom_avg_key in custom_tavgs.keys():
            for key in keys_to_tavg:
                custom_tavgs[custom_avg_key][key] /= custom_tavgs[custom_avg_key][
                    "nfiles"
                ]

    # Save dictionary as pickle
    import pickle

    with open(os.path.join(base_directory, "tavgs.pickle"), "wb") as handle:
        pickle.dump(tavgs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if using_custom_tavg:
        with open(os.path.join(base_directory, "custom_tavgs.pickle"), "wb") as handle:
            pickle.dump(custom_tavgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save dictionary as plaintext (json)
    import json
    import copy

    # Convert NumPy arrays to lists in the copy
    def convert_numpys_to_lists(d):
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, dict):
                convert_numpys_to_lists(value)

    tavgs_jsonable = copy.deepcopy(tavgs)
    convert_numpys_to_lists(tavgs_jsonable)
    with open(os.path.join(base_directory, "tavgs.json"), "w") as handle:
        json.dump(tavgs_jsonable, handle, indent=4)
    if using_custom_tavg:
        custom_tavgs_jsonable = copy.deepcopy(custom_tavgs)
        convert_numpys_to_lists(custom_tavgs_jsonable)
        with open(os.path.join(base_directory, "custom_tavgs.json"), "w") as handle:
            json.dump(custom_tavgs_jsonable, handle, indent=4)
