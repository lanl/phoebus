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

PHDF_PATH = (
    "../../external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/"
)
res_low = [8, 16, 32, 64, 128]
res_high = [128, 256, 512]
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

rho0 = 1
ug0 = 1.0
B10 = 1.0
B20 = 0.0
B30 = 0.0
amp = 1.0e-5

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

parser = ArgumentParser(description="Check linear mode convergence")
parser.add_argument(
    "--highres", action="store_true", help="Use higher resolution for convergence"
)
parser.add_argument(
    "-i",
    "--use_initial",
    action="store_true",
    help="Use initial conditions as analytic solution",
)
parser.add_argument("-a", "--lapse", type=float, default=1.0, help="Lapse to use")
parser.add_argument(
    "-b", "--boost", action="store_true", help="boost the coordinate system"
)
parser.add_argument(
    "-p",
    "--physics",
    metavar="P",
    type=str,
    default="hydro",
    choices=["hydro", "mhd"],
    help="What physics to use",
)
parser.add_argument(
    "-m",
    "--mode",
    metavar="M",
    type=str,
    default=None,
    choices=["entropy", "sound", "alfven", "slow", "fast"],
    help="What mode to test. Defaults to 'sound' for hydro, 'fast' for mhd.",
)
parser.add_argument(
    "-r",
    "--recon",
    metavar="R",
    type=str,
    default="weno5",
    help="Reconstruction method",
)
parser.add_argument("-s", "--save", type=str, default=None, help="File to save plot as")
parser.add_argument(
    "-d", "--dim", type=int, default=2, choices=[1, 2], help="Number of dimensions"
)
parser.add_argument(
    "-n",
    "--nblocks",
    type=int,
    default=1,
    help="Number of MeshBlocks per dimension (e.g. for exercising AMR prolongation/"
    "restriction across block boundaries). Resolution at each N must be evenly "
    "divisible by this. Default 1 (single block, matching prior behavior).",
)
parser.add_argument("executable", metavar="E", type=str, help="Executable to run")
parser.add_argument("input_file", metavar="pin", type=str, help="Input file to use")
args = parser.parse_args()

if args.highres:
    res = res_high
else:
    res = res_low

physics = args.physics
if args.mode is not None:
    mode_name = args.mode
else:
    # Default mode depends on physics: "sound" isn't a valid mhd mode (and vice versa
    # for alfven/slow/fast), so the default can't be a single fixed value.
    mode_name = "fast" if physics == "mhd" else "sound"
recon = args.recon
plot_each_wave = True
plot_initial = False


def clean_dump_files():
    # Clean up dump files
    for dump in glob.glob("hydro_modes*.phdf*"):
        os.remove(dump)


mode = {}
mode["lapse"] = args.lapse
mode["k1"] = 2.0 * np.pi
mode["k2"] = 2.0 * np.pi
mode["knorm"] = np.sqrt(mode["k1"] ** 2 + mode["k2"] ** 2)
mode["dim"] = args.dim

u1 = 0  # Boosted mode (for entropy)
if physics == "hydro":
    if mode_name == "entropy":
        mode["omega"] = 0 + (2.0 * np.pi / 10.0) * 1j
        mode["vars"] = ["rho"]
        mode["rho"] = 1.0
        mode["ug"] = 0.0
        u1 = 0.1
    elif mode_name == "sound":
        if mode["dim"] == 1:
            mode["omega"] = 0 + 2.7422068833892093j
            mode["vars"] = ["rho", "ug", "u1"]
            mode["rho"] = 0.5804294924639215
            mode["ug"] = 0.7739059899518946
            mode["u1"] = -0.2533201985524494
        elif mode["dim"] == 2:
            mode["omega"] = 0 + 3.8780661653218766j
            mode["vars"] = ["rho", "ug", "u1", "u2"]
            mode["rho"] = 0.5804294924639213
            mode["ug"] = 0.7739059899518947
            mode["u1"] = 0.1791244302079596
            mode["u2"] = 0.1791244302079596
    else:
        sys.exit(
            'mode "'
            + mode_name
            + '" not valid for physics "hydro" (choose from entropy, sound)'
        )
elif physics == "mhd":
    if mode["dim"] != 2:
        sys.exit("physics \"mhd\" requires --dim 2 (1D mhd modes aren't supported)")
    if mode_name == "alfven":
        mode["omega"] = 0 + 3.44144232573j
        mode["vars"] = ["u3", "B3"]
        mode["u3"] = 0.480384461415
        mode["B3"] = 0.877058019307
    elif mode_name == "slow":
        mode["omega"] = 0 + 2.41024185339j
        mode["vars"] = ["rho", "ug", "u1", "u2", "B1", "B2"]
        mode["rho"] = 0.558104461559
        mode["ug"] = 0.744139282078
        mode["u1"] = -0.277124827421
        mode["u2"] = 0.063034892770
        mode["B1"] = -0.164323721928
        mode["B2"] = 0.164323721928
    elif mode_name == "fast":
        mode["omega"] = 0 + 5.53726217331j
        mode["vars"] = ["rho", "ug", "u1", "u2", "B1", "B2"]
        mode["rho"] = 0.476395427447
        mode["ug"] = 0.635193903263
        mode["u1"] = -0.102965815319
        mode["u2"] = -0.316873207561
        mode["B1"] = 0.359559114174
        mode["B2"] = -0.359559114174
    else:
        sys.exit(
            'mode "'
            + mode_name
            + '" not valid for physics "mhd" (choose from alfven, slow, fast)'
        )
else:
    sys.exit('physics "' + physics + '" not understood')

mode["cs"] = mode["omega"].imag / mode["knorm"]
mode["tf"] = 2.0 * np.pi / mode["omega"].imag
mode["tf"] /= mode["lapse"]
mode["hf"] = 1.0 / mode["tf"]
print("final time = %g. cs = %g. 1/tfinal = %g." % (mode["tf"], mode["cs"], mode["hf"]))

EXECUTABLE = args.executable
TMPINPUTFILE = "tmplinmode.pin"
shutil.copyfile(args.input_file, TMPINPUTFILE)


def get_mode(x, y, t, var):
    k1 = mode["k1"]
    k2 = mode["k2"]
    omega = mode["omega"]
    dvar = mode[var]
    if mode["dim"] == 1:
        ans = np.zeros(x.size)
        for i in range(x.size):
            ans[i] = (amp * dvar * np.exp(1j * (k1 * (x[i] - u1 * t)) - omega * t)).real
    elif mode["dim"] == 2:
        ans = np.zeros([x.size, y.size])
        for i in range(x.size):
            for j in range(y.size):
                ans[j, i] = (
                    amp
                    * dvar
                    * np.exp(1j * (k1 * (x[i] - u1 * t) + k2 * y[j]) - omega * t)
                ).real
    return ans


res = np.array(res)

if args.nblocks > 1:
    bad = [int(N) for N in res if N % args.nblocks != 0]
    if bad:
        sys.exit(
            "--nblocks %d does not evenly divide resolution(s) %s"
            % (args.nblocks, bad)
        )

L1 = {}
for var in mode["vars"]:
    L1[var] = np.zeros(res.size)

for n, N in enumerate(res):
    meshblock_N = N // args.nblocks
    # Process input file
    with open(TMPINPUTFILE, "r") as infile:
        lines = infile.readlines()
        section = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("<") and stripped.endswith(">"):
                section = stripped[1:-1]
            # <parthenon/mesh>'s nx1/nx2 set the full mesh resolution; the identically-
            # named nx1/nx2 under <parthenon/meshblock> set the size of each block, so
            # they need to be told apart by section to support --nblocks > 1.
            in_meshblock = section == "parthenon/meshblock"
            if line.startswith("dt") and "dt_init_fact" not in line:
                lines[i] = "dt = %g" % mode["tf"] + "\n"
            if line.startswith("nx1"):
                lines[i] = "nx1 = %i" % (meshblock_N if in_meshblock else N) + "\n"
            if mode["dim"] == 1:
                if line.startswith("nx2"):
                    lines[i] = "nx2 = 1\n"
            elif mode["dim"] == 2:

                #      if physics == 'mhd':
                if line.startswith("nx2"):
                    lines[i] = "nx2 = %i" % (meshblock_N if in_meshblock else N) + "\n"
                # break
            if line.startswith("alpha"):
                lines[i] = "alpha = %g" % mode["lapse"] + "\n"
            if line.startswith("vx"):
                if args.boost:
                    lines[i] = "vx = %g" % mode["hf"] + "\n"
                else:
                    lines[i] = "vx = %g" % 0 + "\n"
            if line.startswith("vy"):
                if args.boost:
                    lines[i] = "vy = %g" % mode["hf"] + "\n"
                else:
                    lines[i] = "vy = %g" % 0 + "\n"
            if line.startswith("physics"):
                lines[i] = "physics = " + physics + "\n"
            if line.startswith("mode"):
                lines[i] = "mode = " + mode_name + "\n"
            if line.startswith("amplitude"):
                lines[i] = "amplitude = %e" % amp + "\n"
            if line.startswith("recon"):
                lines[i] = "recon = " + recon + "\n"

            if physics == "hydro":
                if line.startswith("mhd"):
                    lines[i] = "mhd = false\n"
            elif physics == "mhd":
                if line.startswith("mhd"):
                    lines[i] = "mhd = true\n"
    with open(TMPINPUTFILE, "w") as outfile:
        outfile.writelines(lines)

    # Run simulation
    start = time.time()
    print("Running problem at %i zones... " % N, end="", flush=True)
    call(
        ["mpirun", "-n", "1", EXECUTABLE, "-i", TMPINPUTFILE],
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    stop = time.time()
    print("done in %g seconds" % (stop - start))

    dumps = np.sort(glob.glob("hydro_modes.out1*.phdf"))

    dump = phdf.phdf(dumps[-1])
    tf_soln = mode["tf"]
    t = dump.Time
    if not args.use_initial and (np.fabs(t - tf_soln) / tf_soln) > 0.05:
        print("Mismatch in expected solution times!")
        print("  Code: ", t)
        print("  Soln: ", tf_soln)
        sys.exit()
    if plot_initial:
        dump = phdf.phdf(dumps[0])
        t = 0
    x = dump.x[0, :]
    y = dump.y[0, :]
    parth = {}
    parth["drho"] = dump.Get("p.density", flatten=False)[0, 0, :, :] - rho0
    parth["dug"] = dump.Get("p.energy", flatten=False)[0, 0, :, :] - ug0
    parth["du1"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 0]
    parth["du2"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 1]
    parth["du3"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 2]
    if physics == "mhd":
        parth["dB1"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 0] - B10
        parth["dB2"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 1] - B20
        parth["dB3"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 2] - B30
    if mode["dim"] == 1:
        for key in parth.keys():
            parth[key] = parth[key][0, :]

    if args.use_initial:
        dump = phdf.phdf(dumps[0])
        mode["rho_soln"] = dump.Get("p.density", flatten=False)[0, 0, :, :] - rho0
        mode["ug_soln"] = dump.Get("p.energy", flatten=False)[0, 0, :, :] - ug0
        mode["u1_soln"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 0]
        mode["u2_soln"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 1]
        mode["u3_soln"] = dump.Get("p.velocity", flatten=False)[0, 0, :, :, 2]
        if physics == "mhd":
            mode["B1_soln"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 0] - B10
            mode["B2_soln"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 1] - B20
            mode["B3_soln"] = dump.Get("p.bfield", flatten=False)[0, 0, :, :, 2] - B30
        if mode["dim"] == 1:
            for key in mode.keys():
                if "_soln" in key:
                    mode[key] = mode[key][0, :]
    else:
        for var in mode["vars"]:
            mode[var + "_soln"] = get_mode(x, y, t, var)

    for var in mode["vars"]:
        L1[var][n] = np.fabs(parth["d" + var] - mode[var + "_soln"]).sum() / (
            N * N * amp
        )

    if plot_each_wave:
        if mode["dim"] == 1:
            fig, axes = plt.subplots(len(mode["vars"]))
            for n, var in enumerate(mode["vars"]):
                if len(mode["vars"]) == 1:
                    ax = axes
                else:
                    ax = axes[n]
                ax.plot(x, parth["d" + var], color="tab:blue")
                ax.plot(x, mode[var + "_soln"], color="k", linestyle="--")
                ax.set_ylabel(var)
            plt.show()
        elif mode["dim"] == 2:
            fig, axes = plt.subplots(len(mode["vars"]), 2)
            for n, var in enumerate(mode["vars"]):
                ax0 = axes[n, 0]
                ax1 = axes[n, 1]
                ax0.pcolormesh(mode[var + "_soln"])
                ax0.set_ylabel(var)
                ax1.pcolormesh(parth["d" + var])
                if n == 0:
                    ax0.set_title("Solution")
                    ax1.set_title("phoebus")
            plt.show()

    clean_dump_files()

# Get fit to see convergence rate
for var in mode["vars"]:
    fit = np.polyfit(np.log(res), np.log(L1[var]), deg=1)
    mode[var + "_fit_slope"] = fit[0]
    mode[var + "_fitx"] = res
    mode[var + "_fity"] = np.exp(fit[0] * np.log(mode[var + "_fitx"]) + fit[1])

plt.figure()
ax = plt.gca()
for n, var in enumerate(mode["vars"]):
    plt.plot(
        res, L1[var], color=colors[n], label="$\\delta$" + var, linestyle="", marker="s"
    )
    plt.plot(
        mode[var + "_fitx"],
        mode[var + "_fity"],
        color=colors[n],
        linestyle="--",
        label="$N^{%g}$" % mode[var + "_fit_slope"],
    )
plt.legend(loc=1)
plt.xlabel("N")
plt.ylabel("L1")
plt.xscale("log")
plt.yscale("log")
plt.title(physics + " " + mode_name)
ax.set_xticks(res)
ticks = []
for N in res:
    ticks.append(str(N))
ax.set_xticklabels(ticks, minor=False)
ax.xaxis.set_tick_params(which="minor", bottom=False)
plt.minorticks_off()
if args.save:
    plt.savefig(args.save, bbox_inches="tight")
else:
    plt.show()
