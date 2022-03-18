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

import numpy as np
from scipy import integrate, optimize, interpolate
from scipy.fft import fft, ifft
import h5py
from glob import glob
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font',size=18)

class Dump:
    def __init__(self,filename):
        with h5py.File(filename,'r') as f:
            self.t = f['Info'].attrs['Time']
            self.xf = f['Locations/x'][0]
            self.xc = 0.5*(self.xf[1:] + self.xf[:-1])
            self.rhoc = f['c.density'][0,0,0,:,0]
            self.ec = f['c.energy'][0,0,0,:,0]
            self.pc = f['c.momentum'][0,0,0,:,0]
            self.rhop = f['p.density'][0,0,0,:,0]
            self.ep = f['p.energy'][0,0,0,:,0]
            self.pv = f['p.velocity'][0,0,0,:,0]
            self.press = f['pressure'][0,0,0,:,0]
            self.src = f['src_terms'][0,0,0,:,:]
            self.flx = f['flux_divergence'][0,0,0,:,:]
            try:
                self.alpha = f['g.c.alpha'][0,0,0,:,0]
                self.detgam = f['g.c.detgam'][0,0,0,:,0]
                self.gcov = f['g.c.gcov'][0,0,0,:,:]
            except:
                pass

filenames = sorted(glob(f'tov.out1.*.phdf'))
data = [Dump(fnam) for fnam in filenames]

plt.loglog(data[0].xc,data[0].press,label='initial time')
plt.loglog(data[-1].xc,data[-1].press,linestyle='--',label=f't={data[-1].t}')
plt.legend()
plt.ylim(1e-9,1e-1)
plt.xlabel(r'$r$')
plt.ylabel('pressure')
plt.savefig('tov_initial_final_comparison.png',bbox_inches='tight')
plt.cla()
plt.clf()

times = []
rho0s = []
for d in data:
    times.append(d.t)
    rho0s.append(d.rhop[0])
times = np.array(times)
rho0s = np.array(rho0s)

drhos = (rho0s - rho0s[0])/rho0s[0]

plt.plot(times,drhos)
plt.xlabel('Time')
plt.ylabel(r'$(\rho_c - \rho_{c,0})/\rho_{c,0}$')
plt.savefig('tov_quasinormal.png',bbox_inches='tight')
plt.cla()
plt.clf()

drhos_fourier = fft(drhos)
dt = times[1] - times[0]
xf = np.arange(len(times))/dt
plt.semilogy(xf,drhos_fourier)
plt.ylabel('power (DB)')
plt.xlabel('freq (1/time)')
plt.xlim(0,30)
plt.savefig('power-spectrum.png',bbox_inches='tight')
plt.cla()
plt.clf()
