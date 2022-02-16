#!/usr/bin/env python

import os
import sys
import numpy as np
from subprocess import run, check_output, CalledProcessError, STDOUT

try:
  print(check_output(['mpirun', '--version'],
  stderr=STDOUT))
except CalledProcessError as e:
  print("Subprocess error: ", e.output)

try:
  #print(check_output(['mpirun', '-np', '1', executable, '-i', TEMPORARY_INPUT_FILE],
  print(check_output(['mpirun', '-np', '1', 'phoebus', '-i', 'phoebus.pin'],
  stderr=STDOUT))
except CalledProcessError as e:
  print("Subprocess error: ", e.output)
