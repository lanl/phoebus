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
parser.add_argument('-f', '--files', dest='files', nargs='*', default='rad_adv*.phdf', help='List of input Parthenon hdf files to plot')
parser.add_argument('-o', '--out', dest='out_file', default='rad_adv_J.pdf', help='Plot output file')

# Set the parameters defining the initial conditions
# Defaults should be consistent with inputs/radiation_advection.pin
parser.add_argument('-v', dest='v', default=0.3, action="store", type=float)
parser.add_argument('-k', '--kappa', dest='kappa', default=1e3, action="store", type=float,
                    help='Background opacity in comoving frame')
parser.add_argument('-s', '--sigma', dest='sigma', default=0.03333, action="store", type=float,
                    help='Initial spread of gaussian pulse in comoving frame')
parser.add_argument('--J0', dest='J0', default=1.0, action="store", type=float,
                    help='Height of pulse at t=t0 in comoving frame')


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
files = []
for file in args.files:
  files += glob.glob(file)
files = sorted(files)

# Set up unit conversions
file0 = phdf.phdf(files[0])
L_unit = file0.Params['phoebus/LengthCodeToCGS']
T_unit = file0.Params['phoebus/TimeCodeToCGS']
M_unit = file0.Params['phoebus/MassCodeToCGS']
scale_free = True
if not np.isclose(L_unit, 1.) or not np.isclose(T_unit, 1.) or not np.isclose(M_unit, 1.):
  scale_free = False
E_unit = M_unit*L_unit**2/T_unit**2
UE_unit = E_unit / L_unit**3
J0 *= UE_unit

# Find the minimum and maximum times of the data
minTime = sys.float_info.max
maxTime = -sys.float_info.max
for file in files:
  dfile = phdf.phdf(file)
  minTime = min([dfile.Time, minTime])
  maxTime = max([dfile.Time, maxTime])
maxTime = max([maxTime, minTime + 0.01])

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
  J = dfile.Get("r.p.J", flatten=False)*UE_unit
  x = dfile.x*L_unit
  t = dfile.Time

  if (t>maxTime): continue

  color = cmap((t - minTime)/(maxTime - minTime))
  for block in range(dfile.NumBlocks):
    if J.ndim == 5:
      plt_ax.plot(x[block, :], J[block, iz, iy, :, ispec], color=color)
    else:
      plt_ax.plot(x[block, :], J[block, iz, iy, :], color=color)

  xmin = np.amin(x)
  xmax = np.amax(x)
  xgrid = np.arange(xmin, xmax, (xmax-xmin)/1000)
  tdiff = t + t0
  if args.analytic:
    plt_ax.plot(xgrid, BoostedDiffusion(kappa, x0p, v, t0p, J0, xgrid/L_unit, t + t0p), linestyle='--', color='k')

xl = v*L_unit # 0.3
xh = 1.0*L_unit
yl = -0.1
yh = 1.05*J0
if scale_free:
  plt_ax.set_ylabel('J (arb. units)')
  plt_ax.set_xlabel('x (arb. units)')
else:
  plt_ax.set_ylabel('J (erg cm^-3)')
  plt_ax.set_xlabel('x (cm)')
plt_ax.set_xlim([xl, xh])
plt_ax.set_ylim([yl, yh])

if J.ndim == 5:
  etot = sum(J[0, iz, iy, :, ispec])
else:
  etot = sum(J[0, iz, iy, :])
print("etot: ", etot)

plt_ax.text(0.05*(xh-xl)+xl, 0.95*(yh-yl)+yl, '$\kappa={}$'.format(kappa))

if args.savefig:
  plt.savefig(args.out_file)
else:
  plt.show()
