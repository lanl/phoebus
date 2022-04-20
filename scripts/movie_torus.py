DUMP_NAMES = '/home/brryan/builds/phoebus/torus.out1.*.phdf'

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os
from subprocess import call, DEVNULL
import glob
#sys.path.append(PHDF_PATH)
#import phdf
from parthenon_tools import phdf
import time
from enum import Enum
from multiprocessing import Pool

dfnams = np.sort(glob.glob(DUMP_NAMES))

def worker(pair):
  n, dfnam = pair
  print(f'n: {n} / {len(dfnams)}')
  call(['python', 'plot_torus.py', '--nfinal', str(n), '--savefig', 'True'])

p = Pool()
p.map(worker,enumerate(dfnams))
