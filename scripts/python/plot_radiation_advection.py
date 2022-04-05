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

import sys
import numpy as np 
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import glob
import argparse 
from parthenon_tools import phdf

# Boosted diffusion solution for given frame 3-velocity and opacity
# initial time and position in boosted frame are t0p and x0p 
# x and t are position in time in the observer frame, returns J at 
# these position
def BoostedDiffusion(kappa, x0p, v, t0p, J0, x, t):
  gamma = 1/np.sqrt(1-v*v) 
  tp = gamma*(t - v*x)
  xp = gamma*(x - v*t)
  return J0*np.sqrt(t0p/tp)*np.exp(-3*kappa*(xp-x0p)**2/(4*tp)) 

parser = argparse.ArgumentParser(description='Plot a boosted diffusion wave.')

# Set the parameters defining the initial conditions 
# Defaults should be consistent with inputs/radiation_advection.pin 
parser.add_argument('-v', dest='v', default=0.3, action="store", type=float)
parser.add_argument('-k', '--kappa', dest='kappa', default=1e3, action="store", type=float, 
                    help='Background opacity in comoving frame')
parser.add_argument('-s', '--sigma', dest='sigma', default=0.03333, action="store", type=float, 
                    help='Initial spread of gaussian pulse in comoving frame')
parser.add_argument('--J0', dest='J0', default=1.0, action="store", type=float, 
                    help='Height of pulse at t=t0 in comoving frame')

parser.add_argument('-d', '--dir', dest='dir', default='./', action="store", type=str, help='Directory containing rad_adv*.phdf files') 

parser.add_argument('--savefig', type=bool, default=True, help='Whether to save figure')
parser.add_argument('--analytic', type=bool, default=True, help='Whether to include analytic boosted diffusion in plot')

args = parser.parse_args()

v = args.v
kappa = args.kappa
sigma = args.sigma
J0 = args.J0

# Calculate the initial time in the primed frame based on the initial spread 
# since at t=0 J \propto \delta(x)
t0p = 3/2*kappa*sigma**2
# Initial time in the observer frame is defined to be equal to the initial 
# time in the co-moving frame 
t0 = t0p
# Get the central position of the gaussian in the observer frame at t0 
x0p = (0.5 - v*t0)/np.sqrt(1-v*v)
# Lorentz factor 
W = 1/np.sqrt(1-v*v)

# Read in the files 
base_dir = './'
work_dir = '/'
files = sorted(glob.glob(args.dir + 'rad_adv*.phdf'))
plot_tmax = 100000

# Find the minimum and maximum times of the data
minTime = sys.float_info.max
maxTime = -sys.float_info.max
for file in files: 
  dfile = phdf.phdf(file)
  minTime = min([dfile.Time, minTime]) 
  maxTime = max([dfile.Time, maxTime]) 

maxTime = max([maxTime, minTime + 0.01])
maxTime = min([plot_tmax, maxTime])

# Set up the axes with a time colorbar
cmap = cm.get_cmap('viridis')
fig = plt.figure(figsize=[20,8])
plt_ax = fig.add_axes([0.15, 0.15, 0.68, 0.8])
c_map_ax = fig.add_axes([0.86, 0.2, 0.03, 0.7])
mpl.colorbar.ColorbarBase(c_map_ax, cmap=cmap, 
                          norm=mpl.colors.Normalize(minTime, maxTime), 
                          orientation = 'vertical', 
                          label='Time')

# Plot the data (should work for refinement, but untested)
# Choose the species and y and z locations
# ispec currently has to be fixed to 0 because of tensor issues in Parthenon output
ispec = 0
iz = 0 
iy = 0
for file in files[0::1]: 
  dfile = phdf.phdf(file)
  J = dfile.Get("r.p.J", flatten=False)
  x = dfile.x 
  t = dfile.Time
  
  if (t>maxTime): continue

  color = cmap((t - minTime)/(maxTime - minTime))
  for block in range(dfile.NumBlocks):
    plt_ax.plot(x[block, :], J[block, iz, iy, :, ispec], color=color)
  
  xmin = np.amin(x)
  xmax = np.amax(x) 
  xgrid = np.arange(xmin, xmax, (xmax-xmin)/1000)
  tdiff = t + t0 
  if args.analytic:
    plt_ax.plot(xgrid, BoostedDiffusion(kappa, x0p, v, t0p, J0, xgrid, t + t0p), linestyle='--', color='k')

xl = 0.3
xh = 1.0
yl = -0.1 
yh = 1.05 
plt_ax.set_ylabel('J (arb. units)')
plt_ax.set_xlabel('x (arb. units)')
plt_ax.set_xlim([xl, xh])
plt_ax.set_ylim([yl, yh])

plt_ax.text(0.05*(xh-xl)+xl, 0.95*(xh-xl)+xl, '$\kappa={}$'.format(kappa))

if args.savefig:
  plt.savefig('rad_adv_J.pdf')
else:
  plt.show() 
