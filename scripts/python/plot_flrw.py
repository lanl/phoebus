#!/usr/bin/env python

import h5py
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from glob import glob
from scipy import integrate
from scipy import interpolate

mpl.use("Agg")
rc("font", size=18)


parser = ArgumentParser(description="Plot the FLRW Universe")
parser.add_argument(
    "--a0", type=float, default=1, help="Initial value of scale factor a"
)
parser.add_argument(
    "--dadt", type=float, default=1, help="Time derivative of scale factor a"
)
parser.add_argument(
    "--savename", type=str, default="flrw.png", help="Name to save plot to"
)
parser.add_argument(
    "files", type=str, nargs="+", help="Files to use for time evolution"
)


def get_rho_adm(f):
    with h5py.File(f, "r") as f:
        t = f["Info"].attrs["Time"]
        rho = f["p.density"][()]
        u = f["p.energy"][()]
        P = f["pressure"][()]
    return t, rho.mean(), (rho + u).mean(), P.mean()


if __name__ == "__main__":
    args = parser.parse_args()
    files = list(sorted(args.files))
    times = np.empty(len(files))
    rhos = np.empty(len(files))
    rho_adms = np.empty(len(files))
    pressures = np.empty(len(files))
    for i in range(len(files)):
        times[i], rhos[i], rho_adms[i], pressures[i] = get_rho_adm(files[i])
    P_interp = interpolate.interp1d(times, pressures, fill_value="extrapolate")

    def a(t):
        return args.a0 + args.dadt * t

    def H(t):
        return args.dadt / a(t)

    def rhorhs(t, rho):
        P = P_interp(t)
        return -3 * (rho + P) * H(t)

    rho_solved = np.empty(len(files))
    rho_solved[0] = rho_adms[0]
    integrator = integrate.ode(rhorhs)
    integrator.set_initial_value(rho_adms[0], 0)
    for i, t in enumerate(times[1:]):
        integrator.integrate(t)
        rho_solved[i + 1] = integrator.y

    plt.plot(times, rho_adms, label="code")
    plt.plot(times, rho_solved, ls="--", label="semi-analytic")
    plt.xlabel("time")
    plt.ylabel(r"$\rho_{ADM}$")
    plt.legend()
    plt.savefig(args.savename, bbox_inches="tight")
