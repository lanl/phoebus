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

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import glob
import argparse
from parthenon_tools import phdf

import numpy as np
import scipy.integrate as integrate

# See Murchikova et al. for analytic solution
def HomSphereRadius(r, Rsphere, B, kappa):
    def gIn(mu):
        return np.sqrt(1 - (r * r / (Rsphere * Rsphere)) * (1 - mu * mu))

    def gOut(mu):
        return np.where(
            mu > np.sqrt(1 - (Rsphere * Rsphere) / (r * r)),
            np.sqrt(1 - (r * r / (Rsphere * Rsphere)) * (1 - mu * mu)),
            0,
        )

    def s(mu):
        return np.where(r < Rsphere, r / Rsphere * mu + gIn(mu), 2 * gOut(mu))

    (J, err) = integrate.quad(lambda x: 1 - np.exp(-kappa * Rsphere * s(x)), -1, 1)
    (H, err) = integrate.quad(
        lambda x: x * (1 - np.exp(-kappa * Rsphere * s(x))), -1, 1
    )

    return (0.5 * B * J, 0.5 * B * H)


def HomSphere(radii, Rsphere, B, kappa):

    J = np.zeros(radii.shape)
    H = np.zeros(radii.shape)

    for i in range(radii.shape[0]):
        (J[i], H[i]) = HomSphereRadius(radii[i], Rsphere, B, kappa)
    return (J, H)


parser = argparse.ArgumentParser(description="Plot homogeneous sphere.")
parser.add_argument(
    "-f",
    "--files",
    dest="files",
    nargs="*",
    default="rad_adv*.phdf",
    help="List of input Parthenon hdf files to plot",
)
parser.add_argument(
    "-o", "--out", dest="out_file", default="rad_adv_J.pdf", help="Plot output file"
)

# Set the parameters defining the initial conditions
# Defaults should be consistent with inputs/radiation_advection.pin
parser.add_argument("-v", dest="v", default=0.3, action="store", type=float)
parser.add_argument(
    "-k",
    "--kappa",
    dest="kappa",
    default=10,
    action="store",
    type=float,
    help="Background opacity inside the sphere",
)
parser.add_argument(
    "-r",
    "--radius",
    dest="radius",
    default=1,
    action="store",
    type=float,
    help="Sphere radius",
)
parser.add_argument(
    "-B",
    dest="B",
    default=1.0,
    action="store",
    type=float,
    help="Blackbody value of sphere",
)


parser.add_argument("--savefig", type=bool, default=True, help="Whether to save figure")
parser.add_argument(
    "--analytic",
    type=bool,
    default=True,
    help="Whether to include analytic boosted diffusion in plot",
)

args = parser.parse_args()


# Read in the files
files = []
for file in args.files:
    files += glob.glob(file)
files = sorted(files)

# Find the minimum and maximum times of the data
minTime = sys.float_info.max
maxTime = -sys.float_info.max
for file in files:
    dfile = phdf.phdf(file)
    minTime = min([dfile.Time, minTime])
    maxTime = max([dfile.Time, maxTime])
maxTime = max([maxTime, minTime + 0.01])

# Set up the axes with a time colorbar
cmap = cm.get_cmap("viridis")
fig = plt.figure(figsize=[20, 8])
plt_ax = fig.add_axes([0.15, 0.15, 0.68, 0.8])
c_map_ax = fig.add_axes([0.86, 0.2, 0.03, 0.7])
mpl.colorbar.ColorbarBase(
    c_map_ax,
    cmap=cmap,
    norm=mpl.colors.Normalize(minTime, maxTime),
    orientation="vertical",
    label="Time",
)

# Plot the data (should work for refinement, but untested)
# Choose the species and y and z locations
# ispec currently has to be fixed to 0 because of tensor issues in Parthenon output
ispec = 0
iz = 0
iy = 0
for file in files[0::1]:
    dfile = phdf.phdf(file)
    J = dfile.Get("r.p.J", flatten=False)
    x = dfile.x
    t = dfile.Time

    if t > maxTime:
        continue

    color = cmap((t - minTime) / (maxTime - minTime))
    for block in range(dfile.NumBlocks):
        plt_ax.plot(x[block, :], J[block, iz, iy, :, ispec], color=color)

    xmin = np.amin(x)
    xmax = np.amax(x)
    xgrid = np.arange(xmin, xmax, (xmax - xmin) / 1000)

xl = 0.0
xh = 5.0
yl = -0.1
yh = 1.05

# Plot the analytic solution
if args.analytic:
    radii = np.arange(xl, xh, (xh - xl) / 1000)
    (J, H) = HomSphere(radii, args.radius, args.B, args.kappa)
    plt_ax.plot(radii, J, linestyle="--", color="k")

plt_ax.set_ylabel("J (arb. units)")
plt_ax.set_xlabel("x (arb. units)")
plt_ax.set_xlim([xl, xh])
plt_ax.set_ylim([yl, yh])

plt_ax.text(
    0.05 * (xh - xl) + xl, 0.95 * (xh - xl) + xl, "$\kappa={}$".format(args.kappa)
)

if args.savefig:
    plt.savefig(args.out_file)
else:
    plt.show()
