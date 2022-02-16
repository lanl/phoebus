#!/usr/bin/env python3

import os
import sys
import numpy as np
from subprocess import run, check_output, CalledProcessError, STDOUT

try:
  print(check_output(['mpirun', '--version'],
  stderr=STDOUT))
except CalledProcessError as e:
  print("Subprocess error: ", e.output)
