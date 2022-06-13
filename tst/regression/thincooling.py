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

import argparse
import os
import sys
import regression_test as rt

parser = argparse.ArgumentParser(description='Run optically thin cooling as a test')
parser.add_argument('--upgold', dest='upgold', action='store_true')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
parser.add_argument('--use_mpiexec', dest='use_mpiexec', action='store_true')
parser.add_argument('--input', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'inputs/thincooling.pin'))
parser.add_argument('--executable', type=str, default=None)
parser.add_argument('--build_type', type=str, default='Release', choices=['Debug', 'Release'])
args = parser.parse_args()

modified_inputs = {}
modified_inputs['radiation/method'] = 'monte_carlo'

code = rt.gold_comparison(variables=['p.density', 'p.energy'],
                          input_file=args.input,
                          modified_inputs=modified_inputs,
                          executable=args.executable,
                          geometry='Minkowski',
                          use_gpu=args.use_gpu,
                          use_mpiexec=args.use_mpiexec,
                          build_type=args.build_type,
                          upgold=args.upgold)

sys.exit(code)
