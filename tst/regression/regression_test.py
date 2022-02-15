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

import os
import sys
import numpy as np
from subprocess import run
import shutil
import glob
sys.path.append("../../external/parthenon/scripts/python/packages/parthenon_tools")
from parthenon_tools import phdf
import __main__

# ------------------------------------------------------------------------------------------------ #
# Constants
#

BUILD_DIR='build'
RUN_DIR='run'
SOURCE_DIR='../../../'
NUM_PROCS=4 # Default values for cmake --build --parallel can overwhelm CI systems
TEMPORARY_INPUT_FILE='test_input.pin'
SCRIPT_NAME=os.path.basename(__main__.__file__).split('.py')[0]

# ------------------------------------------------------------------------------------------------ #
# Utility functions
#

# -- Compare two values up to some floating point tolerance
def soft_equiv(val, ref, tol = 1.e-5):
  if ref < 1.e-100:
    return True

  if np.fabs(val - ref)/np.fabs(ref) > tol:
    return False
  else:
    return True

# -- Modify key in input file, add key (and block) if not present, write new file
def modify_input(dict_key, value, input_file):
  key = dict_key.split('/')[-1]
  block = dict_key.split(key)[0][:-1]

  new_input_file = []

  current_block = None

  input_found = False

  with open(input_file, 'r') as infile:
    lines = infile.readlines()
    for line in lines:
      sline = line.strip()

      # Skip empty lines and comments
      if len(sline) == 0 or sline[0] == '#':
        continue

      # Check for block
      elif sline[0] == '<':
        current_block = sline.split('<')[1].split('>')[0]

      # Check for key
      elif len(sline.split('=')) != 2 or ',' in sline or '&' in sline:
        # Multiple values not supported for modification
        new_input_file.append(line)
        continue

      else:

        current_key = sline.split('=')[0].strip()
        current_value = sline.split('=')[1].strip()

        newline = line
        if block == current_block and key == current_key:
          newline = key + ' = ' + str(value) + '\n'

        new_input_file.append(newline)
        input_found = True
        continue


      new_input_file.append(line)

  index = None
  if input_found == False:
    print(f"Input \"{block}\" \"{key}\" not found!")
    for i, line in enumerate(new_input_file):
      if line == f"<{block}>":
        index = i

  if index is None:
    # Block doesn't exist
    new_input_file.append(f"<{block}>\n")
    new_input_file.append(key + ' = ' + str(value) + '\n')
  else:
    # Block exists but key doesn't
    new_input_file.insert(index, key + ' = ' + str(value) + '\n')

  with open(input_file, 'w') as outfile:
    for line in new_input_file:
      outfile.write(line)

# ------------------------------------------------------------------------------------------------ #
# Common regression test tools
#

# -- Configure and build phoebus with problem-specific options
def build_code(geometry, use_gpu=False, build_type="Release"):
  if os.path.isdir(BUILD_DIR):
    print(f"BUILD_DIR \"{BUILD_DIR}\" already exists! Clean up before calling a regression test script!")
    sys.exit(os.EX_SOFTWARE)
  os.mkdir(BUILD_DIR)
  os.chdir(BUILD_DIR)

  # Base configure options
  configure_options = []
  if build_type == "Release":
    configure_options.append("-DCMAKE_BUILD_TYPE=Release")
  elif build_type == "Debug":
    configure_options.append("-DCMAKE_BUILD_TYPE=Debug")
  else:
    print(f"Build type \"{build_type}\" not known!")
    sys.exit(os.EX_SOFTWARE)
  configure_options.append("-DPHOEBUS_ENABLE_UNIT_TESTS=OFF")
  configure_options.append("-DMAX_NUMBER_CONSERVED_VARS=10")
  configure_options.append("-DPHOEBUS_CACHE_GEOMETRY=ON")
  configure_options.append("-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON")
  if use_gpu:
    configure_options.append("-DPHOEBUS_ENABLE_CUDA=ON")
    configure_options.append("-DKokkos_ARCH_VOLTA70=ON")
    configure_options.append("-DKokkos_ARCH_POWER9=ON")
    configure_options.append("-DKokkos_ENABLE_CUDA=ON")
    configure_options.append("-DCMAKE_CXX_COMPILER=/home/brryan/github/phoebus/external/parthenon/external/Kokkos/bin/nvcc_wrapper")

  # Geometry (problem-dependent)
  configure_options.append(f"-DPHOEBUS_GEOMETRY={geometry}")

  cmake_call = []
  cmake_call.append('cmake')
  for option in configure_options:
    cmake_call.append(option)
  cmake_call.append(SOURCE_DIR)

  # Configure
  run(cmake_call)

  # Compile
  run(['cmake', '--build', '.', '--parallel', str(NUM_PROCS)])

  # Return to base directory
  os.chdir('..')

# -- Clean up working directory
def cleanup():
  if os.getcwd().split(os.sep)[-1] == BUILD_DIR or os.getcwd().split(os.sep)[-1] == RUN_DIR:
    os.chdir('..')

  if os.path.isabs(BUILD_DIR):
    print(f"Absolute paths not allowed for build directory \"{BUILD_DIR}\" -- unsafe when cleaning up!")
    sys.exit(os.EX_SOFTWARE)

  if os.path.isabs(RUN_DIR):
    print(f"Absolute paths not allowed for run directory \"{RUN_DIR}\" -- unsafe when cleaning up!")
    sys.exit(os.EX_SOFTWARE)

  if os.path.exists(BUILD_DIR):
    try:
      shutil.rmtree(BUILD_DIR)
    except:
      print(f"Error cleaning up build directory \"{BUILD_DIR}\"!")

  if os.path.exists(RUN_DIR):
    try:
      shutil.rmtree(RUN_DIR)
    except:
      print(f"Error cleaning up build directory \"{RUN_DIR}\"!")

# -- Run test problem with previously built code, input file, and modified inputs, and compare
#    to gold output
def gold_comparison(variables, input_file, modified_inputs={},
                    executable=os.path.join('..', BUILD_DIR, 'src', 'phoebus'),
                    upgold=False, compression_factor=1, tolerance=1.e-5):

  if os.path.isdir(RUN_DIR):
    print(f"RUN_DIR \"{RUN_DIR}\" already exists! Clean up before calling a regression test script!")
    sys.exit(os.EX_SOFTWARE)
  os.mkdir(RUN_DIR)
  os.chdir(RUN_DIR)

  # Copy test problem and modify inputs
  shutil.copyfile(input_file, TEMPORARY_INPUT_FILE)
  for key in modified_inputs:
    modify_input(key, modified_inputs[key], TEMPORARY_INPUT_FILE)

  # Run test problem
  run(['mpirun', '-np', '1', executable, '-i', TEMPORARY_INPUT_FILE])

  # Get last dump file
  dumpfiles = np.sort(glob.glob('*.phdf'))
  if len(dumpfiles) == 0:
    print("Could not load any dump files!")
    sys.exit(os.EX_SOFTWARE)
  dump = phdf.phdf(dumpfiles[-1])

  # Construct array of results values
  variables_data = np.empty(shape=(0))
  for variable_name in variables:
    variable = dump.Get(variable_name)
    if len(variable.shape) > 1:
      dim = variable.shape[1]
      for d in range(dim):
        variables_data = np.concatenate((variables_data, variable[:,d]))
    else:
      variables_data = np.concatenate((variables_data, variable))

  # Compress results, if desired
  compression_factor = int(compression_factor)
  compressed_variables = np.zeros(len(variables_data) // compression_factor)
  for n in range(len(compressed_variables)):
    compressed_variables[n] = variables_data[compression_factor*n]
  variables_data = compressed_variables

  # Write gold file, or compare to existing gold file
  success = True
  gold_name = os.path.join('../', SCRIPT_NAME) + '.gold'
  if upgold:
    np.savetxt(gold_name, variables_data, newline='\n')
  else:
    gold_variables = np.loadtxt(gold_name)
    if not len(gold_variables) == len(variables_data):
      print("Length of gold variables does not match calculated variables!")
      success = False
    else:
      for n in range(len(gold_variables)):
        if not soft_equiv(variables_data[n], gold_variables[n]):
          success = False

  # Report upgolding, success, or failure
  if upgold:
    print(f"Gold file {gold_name} updated!")
  else:
    if success:
      print("TEST PASSED")
      return os.EX_OK
    else:
      print("TEST FAILED")
      mean_error = np.mean(variables_data - gold_variables)
      print(f"Mean error: {mean_error}")
      return os.EX_SOFTWARE
