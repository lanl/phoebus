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

import argparse
import regression_test as rt

parser = argparse.ArgumentParser(description='Run the Bondi accretion problem as a test')
parser.add_argument('--upgold', dest='upgold', action='store_true')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
parser.add_argument('--executable', type=str, default=None)
args = parser.parse_args()

modified_inputs = {}
modified_inputs['parthenon/mesh/nx1'] = 128

if args.executable == None:
  rt.build_code(geometry="FMKS", use_gpu=args.use_gpu)
  rt.gold_comparison(variables=['p.density', 'p.velocity'],
                     input_file='../../../inputs/bondi.pin',
                     modified_inputs=modified_inputs,
                     upgold=args.upgold,
                     compression_factor=1)
else:
  rt.gold_comparison(variables=['p.density', 'p.velocity'],
                     input_file='../../../inputs/bondi.pin',
                     modified_inputs=modified_inputs,
                     executable=args.executable,
                     upgold=args.upgold,
                     compression_factor=1)
