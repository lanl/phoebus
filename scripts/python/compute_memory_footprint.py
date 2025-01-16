#!/usr/bin/env python

# NOTE: Currently assumes GRMHD, but not radiation
# We need to modify this to account for radiation at
# some point in the future.

# TODO(JMM): Luke suggests making this a class
# and makeing the variable registration
# object-oriented. E.g.,
# mem_size = memory_footprint()
# mem_size.AddINdependent(3, "c.bfield")
# and have compute_footprint be a class method
# This is probably a good way to generalize
# when we move to radiation/etc.
# Also maybe translates nicely to parthenon.

import numpy as np
from argparse import ArgumentParser


def compute_footprint(nb, nx1, nx2, nx3):
    """Compute footrprint for Phoebus given a number nb of meshblocks,
    each of size nx1 * nx2 * nx3
    Output size is in GB. Assumes some set of variables
    and about a 15% overhead due to unaccounted for memory.
    """
    GB2B = 1024 * 1024 * 1024
    B2GB = 1.0 / GB2B

    ng = 4
    bytes_per_num = 8
    nums_per_3d_var = (nx1 + 2 * ng) * (nx2 + 2 * ng) * (nx3 + 2 * ng)
    nums_per_2d_var = (nx1 + 2 * ng) * (nx2 + 2 * ng)
    stages = 4  # base, rhs, k, krhs
    nfaces = 4  # 3 faces + coarse

    # comm buffers
    nums_per_ghost_halo = nums_per_3d_var - nx1 * nx2 * nx3

    # c.energy, c.bfield, c.density, c.momentum
    tot_indep_vars = 1 + 3 + 1 + 3
    indep_3d_nums = tot_indep_vars * (
        nums_per_3d_var * stages + nums_per_3d_var * nfaces + nums_per_ghost_halo
    )
    # p.energy, cell_signal_speed, face_signal_speed, gamma1, p.bfield, ql, qr, temperature, emf, divb, p.velocity, c2p_mu, c2p_scratch, fail, pressure, p.density, node coords
    # ql and qr are nindep*ndim
    dep_3d_nums = (
        1
        + 1
        + 3
        + 3
        + 3
        + tot_indep_vars * (3 + 3)
        + 1
        + 3
        + 1
        + 3
        + 1
        + 1
        + 1
        + 1
        + 1
        + 4
    ) * nums_per_3d_var
    # (centers + faces)*(alpha, dalpha, bcon, gcov, gamcon, detgam, dg, coords)
    dep_2d_nums = 4 * (1 + 4 + 3 + 10 + 6 + 1 + 40 + 4) * nums_per_2d_var

    tot_nums = indep_3d_nums + dep_3d_nums + dep_2d_nums
    tot_bytes = bytes_per_num * tot_nums

    # assume 15% additional memory from stuff I haven't accounted for
    # matches roughly with experimentation
    safety_factor = 1.15

    return safety_factor * tot_bytes * nb * B2GB


def compute_footprint_cubed(nb, nx):
    """Compute memory footprint for Phoebus given a number nb of meshblocks,
    all of size nx^3. Output is in GB.
    """
    return compute_footprint(nb, nx, nx, nx)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute memory footprint for Phoebus")
    parser.add_argument("NB", type=int, help="Number of meshblocks")
    parser.add_argument(
        "NX", type=int, nargs=3, help="Meshblock size. Requires 3 integers."
    )

    args = parser.parse_args()

    footprint = compute_footprint(args.NB, args.NX[0], args.NX[1], args.NX[2])

    outstr = "Memory footprint for {} meshblocks of shape {}x{}x{} = {}."
    print(outstr.format(args.NB, args.NX[0], args.NX[1], args.NX[2], footprint))
