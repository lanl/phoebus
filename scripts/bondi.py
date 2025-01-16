#!/usr/bin/env python

# © 2021. Triad National Security, LLC. All rights reserved.  This
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

PHDF_PATH = "/home/brryan/rpm/phoebus/external/parthenon/scripts/python/"

import numpy as np
import sys
import matplotlib.pyplot as plt
import shutil
import os
from subprocess import call, DEVNULL
import glob
import time
from argparse import ArgumentParser

try:
    # if parthenon tools is installed
    from parthenon_tools import phdf
except:
    sys.path.append(PHDF_PATH)
    import phdf

parser = ArgumentParser(description="Plot Bondi result against initial conditions")
parser.add_argument("-s", "--save", type=str, default=None, help="File to save plot as")
parser.add_argument("executable", metavar="E", type=str, help="Executable to run")
parser.add_argument("input_file", metavar="pin", type=str, help="Input file to use")
args = parser.parse_args()

PROBLEM = "bondi"
EXECUTABLE = args.executable
TMPINPUTFILE = "tmpbondi.pin"
shutil.copyfile(args.input_file, TMPINPUTFILE)


def clean_dump_files():
    # Clean up dump files
    for dump in glob.glob(PROBLEM + "*.phdf*"):
        os.remove(dump)


# Run simulation
start = time.time()
print("Running problem... ", end="", flush=True)
call([EXECUTABLE, "-i", TMPINPUTFILE], stdout=DEVNULL, stderr=DEVNULL)
stop = time.time()
print("done in %g seconds" % (stop - start))

dumps = np.sort(glob.glob(PROBLEM + ".out1*.phdf"))

dump0 = phdf.phdf(dumps[0])
dump1 = phdf.phdf(dumps[-1])
t = dump1.Time
x = dump0.x[0, :]
y = dump0.y[0, :]

initial = {}
final = {}

initial["rho"] = dump0.Get("p.density", flatten=False)[0, 0, 0, :]
final["rho"] = dump1.Get("p.density", flatten=False)[0, 0, 0, :]
initial["ug"] = dump0.Get("p.energy", flatten=False)[0, 0, 0, :]
final["ug"] = dump1.Get("p.energy", flatten=False)[0, 0, 0, :]
initial["u1"] = dump0.Get("p.velocity", flatten=False)[0, 0, 0, :, 0]
final["u1"] = dump1.Get("p.velocity", flatten=False)[0, 0, 0, :, 0]

clean_dump_files()

fig, axes = plt.subplots(3, 1)
ax = axes[0]
ax.plot(x, final["rho"], color="r")
ax.plot(x, initial["rho"], color="k", linestyle="--")
ax.set_ylabel("rho")
ax.set_yscale("log")

ax = axes[1]
ax.plot(x, final["ug"], color="r")
ax.plot(x, initial["ug"], color="k", linestyle="--")
ax.set_ylabel("ug")
ax.set_yscale("log")

ax = axes[2]
ax.plot(x, final["u1"], color="r")
ax.plot(x, initial["u1"], color="k", linestyle="--")
ax.set_ylabel("u^1")
ax.set_yscale("log")
ax.set_xlabel("X1")

if args.save:
    plt.savefig(args.save, bbox_inches="tight")
else:
    plt.show()
