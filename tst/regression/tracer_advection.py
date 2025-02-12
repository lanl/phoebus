#!/usr/bin/env python

# © 2021-2024. Triad National Security, LLC. All rights reserved.  This
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

import argparse
import os
import sys
import regression_test as rt

parser = argparse.ArgumentParser(description="Run a linear mode as a test")
parser.add_argument("--upgold", dest="upgold", action="store_true")
parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
parser.add_argument("--use_mpiexec", dest="use_mpiexec", action="store_true")
parser.add_argument(
    "--input",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../inputs/advection.pin"
    ),
)
parser.add_argument("--executable", type=str, default=None)
parser.add_argument(
    "--build_type", type=str, default="Release", choices=["Debug, Release"]
)
args = parser.parse_args()

modified_inputs = {}
modified_inputs["parthenon/mesh/nx1"] = 64
modified_inputs["parthenon/mesh/nx2"] = 64
modified_inputs["parthenon/mesh/refinement"] = "off"
modified_inputs["parthenon/meshblock/nx1"] = 32
modified_inputs["parthenon/meshblock/nx2"] = 32

modified_inputs["parthenon/output1/variables"] = "p.ye, p.velocity"
modified_inputs["parthenon/output1/swarms"] = "tracers"
modified_inputs["parthenon/output1/tracers_variables"] = "x, y, z, id"
modified_inputs["parthenon/output1/file_type"] = "hdf5"
modified_inputs["parthenon/output1/dt"] = 1.0


modified_inputs["physics/tracers"] = True
modified_inputs["tracers/num_tracers"] = 128
modified_inputs["tracers/rng_seed"] = 8675309

code = rt.gold_comparison(
    variables=["p.ye", "p.velocity"],
    input_file=args.input,
    modified_inputs=modified_inputs,
    swarm_variables={"tracers": ["swarm.x", "swarm.y", "swarm.z"]},
    executable=args.executable,
    geometry="Minkowski",
    use_gpu=args.use_gpu,
    use_mpiexec=args.use_mpiexec,
    build_type=args.build_type,
    upgold=args.upgold,
    compression_factor=10,
)

sys.exit(code)
