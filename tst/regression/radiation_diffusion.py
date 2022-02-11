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

parser = argparse.ArgumentParser(description='Run a finite velocity radiation diffusion test')
parser.add_argument('--upgold', dest='upgold', action='store_true')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
parser.add_argument('--save_output', dest='save_output', action='store_true')
parser.add_argument('--executable', type=str, default=None)
args = parser.parse_args()

modified_inputs = {}
modified_inputs['radiation/scattering_fraction'] = 1.0
modified_inputs['radiation/B_fake'] = 0.5
modified_inputs['opacity/gray_kappa'] = 1.e3
modified_inputs['radiation_advection/J'] = 1.0
modified_inputs['radiation_advection/Hx'] = 0.0
modified_inputs['radiation_advection/Hy'] = 0.0
modified_inputs['radiation_advection/Hz'] = 0.0
modified_inputs['radiation_advection/vx'] = 0.3
modified_inputs['radiation_advection/width'] = 0.0333
modified_inputs['radiation_advection/kappas_init'] = 1.e3

if args.executable == None:
  rt.build_code(geometry="Minkowski", use_gpu=args.use_gpu)
  rt.gold_comparison(variables=['r.p.J', 'r.p.H'],
                     input_file='../../../inputs/radiation_advection.pin',
                     modified_inputs=modified_inputs,
                     upgold=args.upgold,
                     compression_factor=10,
                     save_output=args.save_output)
else:
  rt.gold_comparison(variables=['r.p.J', 'r.p.H'],
                     input_file='../../../inputs/radiation_advection.pin',
                     modified_inputs=modified_inputs,
                     executable=args.executable,
                     upgold=args.upgold,
                     compression_factor=10,
                     save_output=args.save_output)
