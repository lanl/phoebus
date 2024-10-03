#!/usr/bin/env python3

"""
This script retrieves raw MESA profile data variables, reverses them to be center to surface, and outputs them in the format needed for GRSOLVER for use in Phoebus.

Usage:
    python read_MESA_for_grsolver.py --file='20m_cc' --ext='.dat'

Arguments:
    input_file: filename and file extension of MESA profile
"""

import numpy as np
import pandas as pd
from argparse import ArgumentParser


def load(file, ext):
    df = pd.read_csv(file + ext, sep=r"\s+", header=4)
    df = df.iloc[::-1].reset_index(drop=True)
    nzones = len(df)
    if nzones > 0:
        print("loaded file:", file + ext, " with ", nzones, " zones.")
    else:
        print("failed to load file", file + ext)
        exit()
    return df


def retrieve_and_save(df, file):
    radius = df["radius_cm"]
    velocity = df["velocity"]
    # if model non-rotating, pass array of zeros
    try:
        angular_velocity = df["omega"]
    except:
        angular_velocity = np.zeros_like((radius))
    density = df["density"]  # g/cm^3
    pressure = df["pressure"]  # g/cm^3
    ye = df["ye"]
    temperature = df["temperature"]  # K
    sie = df["energy"]  #  ! internal energy (ergs/g)

    np.savetxt(
        file + str("_raw_data.txt"),
        np.column_stack(
            [
                radius,
                velocity,
                angular_velocity,
                density,
                pressure,
                ye,
                temperature,
                sie,
            ]
        ),
        header="MESA output from model `" + str(file) + "` | \ "
        "radius (cm) velocity (cm/s) angular_velocity (rad/s)  density (g/cm^3) pressure (dyne/cm^2) ye temperature (K) specific internal energy (erg/g)",
    )
    return


def main():
    parser = ArgumentParser(
        description="Retrieve raw MESA profile data variables and output to format needed for GRSOLVER for use in Phoebus."
    )

    parser.add_argument(
        "--file",
        type=str,
        default="file",
        help="name of the input MESA file without extension",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".dat",
        help="input MESA file extension",
    )

    args = parser.parse_args()
    print(args.file, args.ext)
    df = load(args.file, args.ext)
    retrieve_and_save(df, args.file)

    return


if __name__ == "__main__":
    main()
