#!/usr/bin/env python3

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
import shutil
import stat
from subprocess import call
import regression_test as rt

parser = argparse.ArgumentParser(description='Run regression test suite')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
args = parser.parse_args()

# Compile all the executables we'll need only once
def copy_executable(src, dest):
  shutil.copyfile(src, dest)
  dest_st = os.stat(dest)
  os.chmod(dest, dest_st.st_mode | stat.S_IEXEC)

rt.build_code(geometry="Minkowski", use_gpu=args.use_gpu, build_type="Release")
copy_executable('src/phoebus', '../phoebus_minkowski_release')
rt.cleanup()
rt.build_code(geometry="Minkowski", use_gpu=args.use_gpu, build_type="Debug")
copy_executable('src/phoebus', '../phoebus_minkowski_debug')
rt.cleanup()
rt.build_code(geometry="FMKS", use_gpu=args.use_gpu, build_type="Release")
copy_executable('src/phoebus', '../phoebus_fmks_release')
rt.cleanup()
rt.build_code(geometry="FMKS", use_gpu=args.use_gpu, build_type="Debug")
copy_executable('src/phoebus', '../phoebus_fmks_debug')
rt.cleanup()

# Run individual tests with the relevant executable
def run_container(script, executable):
  RUN_DIR = "run"

  if os.path.isdir(RUN_DIR):
    print(f"RUN_DIR \"{RUN_DIR}\" already exists! Clean up before calling a regression test script!")
    sys.exit()
  os.mkdir(RUN_DIR)
  os.chdir(RUN_DIR)

  code = call([os.path.join('..', script), '--executable', os.path.join('..', executable)])

  os.chdir('..')

  try:
    shutil.rmtree(RUN_DIR)
  except:
    print(f"Error cleaning up run directory \"{RUN_DIR}\"!")

  return code

codes = []
# Debug
codes.append(run_container('linear_modes.py', 'phoebus_minkowski_debug'))
codes.append(run_container('bondi.py', 'phoebus_minkowski_debug'))
#codes.append(run_container('thincooling.py', 'phoebus_minkowski_debug'))
#codes.append(run_container('thincooling_coolingfunction.py', 'phoebus_minkowski_debug'))

# Release
codes.append(run_container('linear_modes.py', 'phoebus_minkowski_release'))
codes.append(run_container('bondi.py', 'phoebus_minkowski_release'))
#codes.append(run_container('thincooling.py', 'phoebus_minkowski_release'))
#codes.append(run_container('thincooling_coolingfunction.py', 'phoebus_minkowski_release'))

for code in codes:
  if code != os.EX_OK:
    print("NOT ALL TESTS PASSED")
    sys.exit(os.EX_SOFTWARE)
print("ALL TESTS PASSED")
sys.exit(os.EX_OK)
