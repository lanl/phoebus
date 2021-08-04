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
import h5py
from glob import glob
filenames = sorted(glob('sedov.out1.*.phdf'))

def get_data(i):
    with h5py.File(filenames[i],'r') as f:
        xf = f['Locations']['x'][0]
        xc = 0.5*(xf[1:] + xf[:-1])
        P = f['pressure'][0,0,0,:,0]
        tau = f['p.energy'][0,0,0,:,0]
        rho = f['p.density'][0,0,0,:,0]
        D = f['c.density'][0,0,0,:,0]
        v = f['p.velocity'][0,0,0,:,0]
    return xc, P, tau, rho, D, v

def make_frame_static(i,clear=False):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    xc, P, tau, rho, D, v = get_data(i)
    fig,axarr=plt.subplots(1,2,figsize=(12,6))
    axarr[0].plot(xc,P,marker='o',label=r'$P$')
    axarr[0].plot(xc,tau,marker='o',label=r'$\tau$')
    axarr[0].plot(xc,rho,marker='o',label='$\rho$')
    axarr[0].set_xlabel(r'$r$')
    axarr[0].legend([r'$P$',r'$\tau$',r'$\rho$'])
    axarr[1].loglog(xc,D,marker='o',label=r'$D$')
    axarr[1].loglog(xc,np.abs(v),marker='o',label=r'$|v_r|$')
    axarr[1].legend()
    axarr[0].set_ylim(0.5,2)
    axarr[1].set_ylim(1e-18,1e0)
    plt.savefig('frame_%02d.png' % i)
    if clear:
        plt.clf()


def make_frame_blast(i,clear=False):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    xc, P, tau, rho, D, v = get_data(i)
    plt.plot(xc,rho,label='$\rho$')
    plt.xlabel(r'$r$')
    plt.ylim(0,11)
    plt.savefig('frame_%02d.png' % i)
    if clear:
        plt.clf()

for i in range(len(filenames)):
    print(i)
    make_frame_static(i,True)
