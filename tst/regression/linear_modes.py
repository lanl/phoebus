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
import os
import sys
import shutil
import glob
from numpy import sort, array, savetxt, loadtxt, fabs
from subprocess import call
sys.path.append("../../external/parthenon/scripts/python/packages/parthenon_tools")
from parthenon_tools import phdf

parser = argparse.ArgumentParser(description='Run a phoebus problem as a test')
parser.add_argument('--upgold', dest='upgold', action='store_true')
args = parser.parse_args()

SCRIPT_NAME=os.path.basename(__file__).split('.py')[0]
COMPRESSION_FACTOR=10
SOURCE_DIR='../../../'
BUILD_DIR='build'
NUM_PROCS=4 # Default values for cmake --build --parallel can overwhelm CI systems
VARIABLES_TO_COMPARE = ['p.density', 'p.velocity']

def soft_equiv(val, ref, tol = 1.e-5):
  if ref < 1.e-100:
    return True

  if fabs(val - ref)/fabs(ref) > tol:
    return False
  else:
    return True

# Build code
if os.path.isdir(BUILD_DIR):
  print("BUILD_DIR already exists! Clean up before calling this script!")
  sys.exit()
os.mkdir(BUILD_DIR)
os.chdir(BUILD_DIR)

configure_options = []
configure_options.append("-DCMAKE_BUILD_TYPE=Release")
configure_options.append("-DPHOEBUS_ENABLE_UNIT_TESTS=OFF")
configure_options.append("-DMAX_NUMBER_CONSERVED_VARS=10")
configure_options.append("-DPHOEBUS_GEOMETRY=Minkowski")
configure_options.append("-DPHOEBUS_CACHE_GEOMETRY=ON")
configure_options.append("-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON")

cmake_call = []
cmake_call.append('cmake')
for option in configure_options:
  cmake_call.append(option)
cmake_call.append(SOURCE_DIR)

call(cmake_call)

call(['cmake', '--build', '.', '--parallel', str(NUM_PROCS)])

# Run test problem
call(['./src/phoebus', '-i', '../../../inputs/linear_modes.pin'])

# Get last dump file
dumpfiles = sort(glob.glob('hydro_modes.*.phdf'))
dump = phdf.phdf(dumpfiles[-1])

# Construct array of results values
import numpy as np
variables = np.empty(shape=(0))
for variable_name in VARIABLES_TO_COMPARE:
  variable = dump.Get(variable_name)
  if len(variable.shape) > 1:
    dim = variable.shape[1]
    for d in range(dim):
      variables = np.concatenate((variables, variable[:,d]))
  else:
    variables = np.concatenate((variables, variable))

# Compress results, if desired
COMPRESSION_FACTOR = int(COMPRESSION_FACTOR)
compressed_variables = np.zeros(len(variables) // COMPRESSION_FACTOR)
for n in range(len(compressed_variables)):
  compressed_variables[n] = variables[COMPRESSION_FACTOR*n]
variables = compressed_variables

# Write gold file, or compare to existing gold file
FAILED = False
gold_name = os.path.join('../', SCRIPT_NAME) + '.gold'
if args.upgold:
  savetxt(gold_name, variables, newline='\n')
else:
  gold_variables = loadtxt(gold_name)
  if not len(gold_variables) == len(variables):
    print("Length of gold variables does not match calculated variables!")
    FAILED = True
  else:
    for n in range(len(gold_variables)):
      if not soft_equiv(variables[n], gold_variables[n]):
        FAILED = True

# Leave build directory
os.chdir('..')

# Clean up build directory, first checking for RELATIVE PATH ONLY
if os.path.isabs(BUILD_DIR):
  print("Absolute paths not allowed for BUILD_DIR -- unsafe when deleting build directory!")
  sys.exit()
try:
  shutil.rmtree(BUILD_DIR)
except:
  print("Error cleaning up build directory!")

# Report success or failure to the system
if args.upgold:
  print(f"Gold file {gold_name} updated!")
  if FAILED:
    print("TEST FAILED")
    sys.exit(os.EX_SOFTWARE)
  else:
    print("TEST PASSED")
    sys.exit(os.EX_OK)
