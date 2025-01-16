### This script converts adm quantities in Phoebus units.


import numpy as np
from sys import exit
import math
from astropy.io import ascii
from astropy.table import Table
from astropy import constants as const
import matplotlib.pyplot as pl
from argparse import ArgumentParser

G = const.G.cgs.value
c = const.c.cgs.value
msun = const.M_sun.cgs.value


def ConvertData(filename, formattype):
    data = ascii.read(filename, format=formattype)
    rhoc = max(data["mass_density"])
    M0 = 1.0 / (rhoc) ** (0.5) * (c ** 2.0 / G) ** (1.5)
    newdata = Table()
    newdata["r"] = data["r"] * c ** 2.0 / G / M0
    # np.save('r',newdata['r'])
    newdata["mass_density"] = data["mass_density"] * (G / c ** 2.0) ** 3.0 * M0 ** 2.0
    newdata["temp"] = data["temp"]
    newdata["Ye"] = data["Ye"]
    newdata["specific_internal_energy"] = data["specific_internal_energy"] / c ** 2.0
    newdata["velocity"] = data["velocity"] / c
    newdata["pressure"] = data["pressure"] * G ** 3.0 / c ** 8.0 * M0 ** 2.0
    dvdr = (newdata["velocity"][1:] - newdata["velocity"][:-1]) / (
        newdata["r"][1:] - newdata["r"][:-1]
    )
    newdata["adm_density"] = data["adm_density"] * (G / c ** 2.0) ** 3.0 * M0 ** 2.0
    newdata["adm_momentum"] = data["adm_momentum"] * G ** 3.0 / c ** 7.0 * M0 ** 2.0
    newdata["S_adm"] = data["S_adm"] * G ** 3.0 / c ** 8.0 * M0 ** 2.0
    newdata["Srr_adm"] = data["Srr_adm"] * G ** 3.0 / c ** 8.0 * M0 ** 2.0
    ascii.write(newdata, "converted_" + filename, overwrite=True, format=formattype)
    return


parser = ArgumentParser(
    prog="convert", description="convert a progenitor file into Phoebus coordinates"
)
parser.add_argument("filename", type=str, help="File to convert")
parser.add_argument(
    "-f",
    "--formattype",
    type=str,
    default="commented_header",
    help="How to read progenitor file. Default is commented_header",
)
args = parser.parse_args()
ConvertData(args.filename, args.formattype)
