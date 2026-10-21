#!/usr/bin/env python

# © 2022-2026. Triad National Security, LLC. All rights reserved.  This
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

"""Plot one or more 1D-profile variables from a phoebus/parthenon HDF5 dump, using the
coords output so it always gets the geometry correct and plots in x1 coordinates. Each
requested variable gets its own panel (subplots stacked, sharing the x-axis), and vector
fields can select a component with "name:component" (e.g. "p.velocity:0").
"""

from __future__ import print_function
from argparse import ArgumentParser
import os
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "balsara_exact"))

# Assumes phdf in global python path
try:
    from parthenon_tools.phdf import phdf
except ModuleNotFoundError:
    from phdf import phdf


def parse_varspec(spec):
    """'name' or 'name:component' -> (name, component)."""
    parts = spec.split(":")
    name = parts[0]
    comp = int(parts[1]) if len(parts) > 1 else 0
    return name, comp


def get_profile(data, varname, comp):
    """Return (x, q) profile arrays for varname's comp-th component, one entry per block
    (not concatenated -- blocks are plotted as separate line segments of the same color so
    a possibly-unsorted block order in the file never produces a visually scrambled line).
    """
    # phdf.Get(..., flatten=False) returns shape [NumBlocks, tensor_components, Nz, Ny,
    # Nx] -- component is axis 1, not the last axis.
    coordname = "g.c.coord"
    coord = data.Get(coordname, False)
    using_phoebus_coords = coord is not None
    if using_phoebus_coords:
        # g.c.coord is the 4-vector [t, x1, x2, x3]; x1 is component 1.
        x = coord[:, 1, :, :, :]
    else:
        x = data.x  # shape [NumBlocks, Nx], no component axis

    q = data.Get(varname, False)
    if q is None:
        raise ValueError(f'variable "{varname}" not found in {data.file}')
    # A scalar (1-component) field is stored without a tensor_components axis at all --
    # shape [NumBlocks, Nz, Ny, Nx] (4D) rather than [NumBlocks, ncomp, Nz, Ny, Nx] (5D).
    if q.ndim not in (4, 5):
        raise ValueError(
            f'"{varname}" has unsupported rank (ndim={q.ndim}); only scalar and '
            "vector fields are supported"
        )
    is_scalar = q.ndim == 4
    ncomp = 1 if is_scalar else q.shape[1]
    if comp >= ncomp:
        raise ValueError(
            f'"{varname}" has {ncomp} component(s); requested component {comp}'
        )

    NB = q.shape[0]
    xblocks = []
    qblocks = []
    for b in range(NB):
        xblocks.append(x[b, 0, 0, :] if using_phoebus_coords else x[b, :])
        qblocks.append(q[b, 0, 0, :] if is_scalar else q[b, comp, 0, 0, :])
    return xblocks, qblocks


def plot_dump(
    filename,
    varspecs,
    savename=None,
    x1bounds=None,
    filename0=None,
    log=True,
    balsara_test=None,
    color="#8cc8f3",
):
    if savename is not None:
        matplotlib.use("Agg")

    data = phdf(filename)
    data0 = phdf(filename0) if filename0 is not None else None

    nvar = len(varspecs)
    fig, axes = plt.subplots(
        nvar, 1, sharex=True, figsize=(6, 2.5 * nvar), squeeze=False
    )
    axes = axes[:, 0]

    for ax, spec in zip(axes, varspecs):
        name, comp = parse_varspec(spec)
        xblocks, qblocks = get_profile(data, name, comp)

        ylabel = name if comp == 0 else f"{name}[{comp}]"
        if log:
            ylabel = r"$\log_{10}|$" + ylabel + r"$|$"

        for xb, qb in zip(xblocks, qblocks):
            qplt = np.log10(np.abs(qb)) if log else qb
            ax.plot(xb, qplt, color=color)

        if data0 is not None:
            x0blocks, q0blocks = get_profile(data0, name, comp)
            for xb, qb in zip(x0blocks, q0blocks):
                qplt = np.log10(np.abs(qb)) if log else qb
                ax.plot(xb, qplt, color="k", linestyle="--")

        if balsara_test is not None:
            import balsara_exact

            analytic = balsara_exact.get_profile(balsara_test, name, comp)
            if analytic is not None:
                xa, qa = analytic
                qaplt = np.log10(np.abs(qa)) if log else qa
                ax.plot(xa, qaplt, color="b", linestyle=":", linewidth=1)

        ax.set_ylabel(ylabel)

    if x1bounds is not None:
        axes[0].set_xlim(x1bounds[0], x1bounds[1])
    axes[-1].set_xlabel(r"$x$")

    fig.tight_layout()

    if savename is None:
        plt.show()
    else:
        fig.savefig(savename, dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot 1d profile(s) from a phoebus dump.")
    parser.add_argument(
        "--xbounds", type=float, nargs=2, default=None, help="min and max bounds for x"
    )
    parser.add_argument(
        "-s",
        "--saveprefix",
        type=str,
        default=None,
        help="Basename for the saved figure (omit to show interactively)",
    )
    parser.add_argument(
        "--linear", action="store_true", help="Use linear, instead of log10|.|, scale"
    )
    parser.add_argument("--pdf", action="store_true", help="Save as pdf instead of png")
    parser.add_argument(
        "-c", "--color", type=str, default="#8cc8f3", help="Line color for the plotted data"
    )
    parser.add_argument(
        "varnames",
        type=str,
        help="Comma-separated variables to plot, one panel each. Use 'name:component' "
        "to select a vector component (default 0), "
        "e.g. 'p.density,p.velocity:0,pressure'",
    )
    parser.add_argument("files", type=str, nargs="+", help="Dump(s) to plot")
    parser.add_argument(
        "--nplot", type=int, default=-1, help="Which file (by index) to plot"
    )
    parser.add_argument(
        "--initial", action="store_true", help="Overlay files[0] as the initial condition"
    )
    parser.add_argument(
        "--balsara",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Overlay the exact analytic solution for this Balsara MHD shock tube test "
        "(see inputs/mhd_shocktube.pin and scripts/python/balsara_exact/)",
    )
    args = parser.parse_args()

    varspecs = args.varnames.split(",")
    log = not args.linear
    postfix = ".pdf" if args.pdf else ".png"
    savename = None if args.saveprefix is None else args.saveprefix + postfix
    filename0 = args.files[0] if args.initial else None

    plot_dump(
        args.files[args.nplot],
        varspecs,
        savename=savename,
        x1bounds=args.xbounds,
        filename0=filename0,
        log=log,
        balsara_test=args.balsara,
        color=args.color,
    )
