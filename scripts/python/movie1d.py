# =========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import newton


def addPath():
    """ add the vis/python directory to the pythonpath variable """
    myPath = os.path.realpath(os.path.dirname(__file__))
    sys.path.insert(0, myPath + "/../../external/parthenon/scripts/python/")
    # sys.path.insert(0,myPath+'/vis/python')


def read(filename, nGhost=0):
    """ Read the parthenon hdf file """
    try:
        from parthenon_tools.phdf import phdf
    except ModuleNotFoundError:
        from phdf import phdf

    f = phdf(filename)
    return f


def sodsolution(pld, prd, rhold, rhord, timed=0.1, x0=0.5):
    # A standard nonrelativisitic shock tube problem is
    # rhol=1.0
    # Pl = 1.0
    # rhor = 0.125
    # Pr = 0.1
    # In a nonrelativisitic code this should be
    # rhol=1.0
    # Pl = 1.e-5
    # rhor = 0.125
    # Pr = 1.e-6
    # For Newtonian speeds, (c_s/c)**2 = gamma*P/rho

    # Find the analytic solution for the Sod shock tube problem
    # L | E | 2 | 1 | R
    # L x1 E x2 2 x3 1 x4 R
    gamma = 1.6666666666666666667
    ald = np.sqrt(gamma * pld / rhold)
    ard = np.sqrt(gamma * prd / rhord)

    # make dimensionless
    pl = pld / (gamma * prd)
    pr = 1.0 / gamma
    rhol = rhold / rhord
    rhor = 1.0
    al = ald / ard
    ar = 1.0
    time = timed * ard

    # First find the Ms, the mach number of the shock
    args = (pl, pr, al, ar, gamma)
    ms = newton(msfunc, 2.0, args=args)

    # Now find the rest of the solutions
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    p1 = pr * (2.0 * gamma / gp1 * ms ** 2 - gm1 / gp1)
    rho1 = rhor / (2.0 / gp1 / ms ** 2 + gm1 / gp1)
    u1 = 2.0 / gp1 * (ms - 1.0 / ms)

    u2 = u1
    p2 = p1
    rho2 = rhol * (p2 / pl) ** (1.0 / gamma)
    a2 = np.sqrt(gamma * p2 / rho2)

    # Positions
    # solutions for expansion fan
    # left of expansion fan
    x1 = x0 - al * time
    # right of expansion fan
    x2 = x0 + (u2 - a2) * time
    x = np.linspace(x1, x2, 20)
    ue = 2.0 / gp1 * (al + (x - x0) / time)
    ae = al - gm1 * ue * 0.5
    pe = pl * (ae / al) ** (2.0 * gamma / gm1)
    rhoe = gamma * pe / ae ** 2

    # Contact discontinuity
    x3 = x0 + u2 * time

    # shock
    x4 = x0 + ms * time

    x = np.append([0.0, x1], x)
    x = np.append(x, [x2, x3, x3, x4, x4, 1.0])
    rho = np.append([rhol, rhol], rhoe)
    rho = np.append(rho, [rho2, rho2, rho1, rho1, rhor, rhor])
    rho = rho * rhord
    return x, rho


def msfunc(ms, pl, pr, al, ar, gam):
    gp1 = gam + 1.0
    gm1 = gam - 1.0
    f = (
        ms
        - 1.0 / ms
        - al
        / ar
        * gp1
        / gm1
        * (
            1.0
            - (pr / pl * (2.0 * gam / gp1 * ms ** 2 - gm1 / gp1)) ** (gm1 / (2.0 * gam))
        )
    )
    print(f)
    return f


def plot_dump(x, q, xana, qana, name, with_mesh=False):
    fig = plt.figure()
    # p = fig.add_subplot(111,aspect=1)
    p = fig.add_subplot(111)
    qmin = q.min()
    qmax = q.max()
    NumBlocks = q.shape[0]
    for i in range(NumBlocks):
        p.plot(x, q)
        p.plot(xana, qana)
        # p.pcolormesh(xf[i,:], yf[i,:], q[i,0,:,:], vmin=qmin, vmax=qmax)
        if with_mesh:
            rect = mpatches.Rectangle(
                (xf[i, 0], yf[i, 0]),
                (xf[i, -1] - xf[i, 0]),
                (yf[i, -1] - yf[i, 0]),
                linewidth=0.225,
                edgecolor="k",
                facecolor="none",
            )
            p.add_patch(rect)
    plt.savefig(name, dpi=300)
    plt.close()


if __name__ == "__main__":
    addPath()
    field = sys.argv[1]
    files = sys.argv[2:]
    dump_id = 0
    for f in files:
        print(f)
        data = read(f)
        print(data)
        x = data.x
        q = data.Get(field, False)
        pres = data.Get("pressure", False)
        dens = data.Get("p.density", False)
        pl = pres[0, 0, 0, 0]
        pr = pres[0, 0, 0, -1]
        rhol = dens[0, 0, 0, 0]
        rhor = dens[0, 0, 0, -1]
        xana, qana = sodsolution(pl, pr, rhol, rhor, timed=data.Time)
        name = str(dump_id).rjust(4, "0") + ".png"
        plot_dump(x[0, :], q[0, 0, 0, :], xana, qana, name, False)
        dump_id += 1
