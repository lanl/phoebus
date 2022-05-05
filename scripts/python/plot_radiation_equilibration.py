import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import shutil
import os
from subprocess import call, DEVNULL
import glob
from parthenon_tools import phdf
import time
from enum import Enum
from phoebus_constants import cgs
import phoebus_utils
from phoedf import phoedf

parser = argparse.ArgumentParser(description='Plot neutrino thermalization')
parser.add_argument('-f', '--files', dest='files', nargs='*', default=['rad_eql*.phdf'], help='List of input Parthenon hdf files to plot')
parser.add_argument('--savefig', type=bool, default=False, help='Whether to save figure')
parser.add_argument('--Tg0', type=float, default=1., help='Initial gas temperature (MeV)')
parser.add_argument('--Tr0', type=float, default=0., help='Initial neutrino temperature (MeV)')
parser.add_argument('--numin', type=float, default=1.e14, help='Minimum frequency (Hz)')
parser.add_argument('--numax', type=float, default=1.e22, help='Maximum frequency (Hz)')
parser.add_argument('--nnu', type=int, default=100, help='Number of frequency support points')
args = parser.parse_args()

if args.savefig:
  matplotlib.use('Agg')

# Initial state
rho0 = 1.e10 # g cm^-3
Tg0 = args.Tg0*cgs['MEV']/cgs['KBOL'] # K
Tr0 = args.Tr0*cgs['MEV']/cgs['KBOL'] # K
# TODO(BRR) Add velocity?
gamma = 5./3. # Ideal gas EOS
NSPECIES=3

nnu = args.nnu
lnumin = np.log10(args.numin)
lnumax = np.log10(args.numax)
dlnu = (lnumax - lnumin)/(nnu - 1)
nus = np.zeros(nnu)
for n in range(nnu):
  nus[n] = 10**(lnumin + n*dlnu)

# Read in the files
files = []
print(args.files)
for file in args.files:
  print(f"globbing {file}")
  files += glob.glob(file)
files = sorted(files)
nfiles = len(files)
print(files)

# Set up unit conversions
dfile0 = phoedf(files[0])
params = dfile0.Params
#L_unit = params['phoebus/LengthCodeToCGS']
#T_unit = params['phoebus/TimeCodeToCGS']
#M_unit = params['phoebus/MassCodeToCGS']
eos = dfile0.GetEOS()
opac = dfile0.GetOpacity()

# Get simulation setup
#for param in params:
#  print(param)
#opacity_model = params['opacity/type'].decode('ascii')

#dfile = phoedf(files[0])
#print(dfile.Params)
#print(opac.alphanu(1., 1., 1., 1.))

t = np.zeros(nfiles)
J = np.zeros(nfiles)
Jstd = np.zeros(nfiles)
for n, filename in enumerate(files[0::1]):
  dfile = phoedf(filename)
  t[n] = dfile.Time
  Jfile = dfile.Get("r.p.J", flatten=False)
  J[n] = np.mean(Jfile)
  Jstd[n] = np.std(Jfile)

# ---------------------------------------------------------------------------- #
# -- Calculate analytic solution
# Ensure tmax is divisible by dt for best accuracy!
tmax = 1.e-1*tc
print("Final time: %e s" % tmax)
dt = min(1.e-2*tc, 1.e-2*tmax)
print("dt: %e s" % dt)
t = 0
cycle = 0

ncycle = int(tmax/dt)

t = np.zeros(ncycle+1)
u = np.zeros(ncycle+1)
E = np.zeros(ncycle+1)
Tg = np.zeros(ncycle+1)
Tr = np.zeros(ncycle+1)

t[0] = 0
Tg[0] = Tg0
Tr[0] = Tr0
u[0] = gas_T_to_u(Tg[0])
E[0] = rad_T_to_E(Tr[0])

# Initialize neutrino spectrum
Inu = np.zeros(nnu)
for n in range(nnu):
  Inu[n] = max(1.e-100, Bnu(Tr[0], nus[n]))

for cycle in range(ncycle):
  phoebus_utils.progress_bar((cycle+1)/ncycle)
  import time
  #print(f"cycle = {cycle} t = {t[cycle]}")
  #T = u[cycle]*mu*(gamma - 1.)/(rho0*cgs['KBOL'])
  T = Tg[cycle]
  dE = 0
  dInu = np.zeros(nnu)
  for n in range(nnu):
    nu = nus[n]

    #dInu[n] = cgs['CL']*dt*(jnu(T, nu) - alphanu(T, nu)*Inu[n])
    dInu[n] = (Inu[n] + cgs['CL']*dt*jnu(T, nu))/(1. + cgs['CL']*dt*alphanu(T, nu)) - Inu[n]
    Inu[n] += dInu[n]
    Inu[n] = max(Inu[n], 1.e-100)

  for n in range(nnu - 1):
    dE += 4.*np.pi/cgs['CL']*(dInu[n+1] + dInu[n])/2*(nus[n+1]-nus[n])

  E[cycle+1] = E[cycle] + dE
  u[cycle+1] = u[cycle] - dE
  Tr[cycle+1] = pow(15.*cgs['CL']**3*cgs['HPL']**3*E[cycle+1]/(7.*cgs['KBOL']**4*np.pi**5*NSPECIES),1./4.)
  Tg[cycle+1] = u[cycle+1]*mu*(gamma - 1.)/(rho0*cgs['KBOL'])


  t[cycle+1] = t[cycle] + dt







print(opac.alphanu(1., 1., 1., 1.))

# Calculate analytic solution
def f(t, y, args):
  nubins = args[0]
  nus = args[1]
  eos = args[2]
  opac = args[3]
  rho0 = args[4]
  Ye0 = args[5]
  ug = y[0]
  ur = y[1]
  Tg = eos.T_from_rho_u_Ye(rho0, ug, Ye0)
  Inu = y[2:nubins+2]
  print(t)
  print(y)

  dydt = np.zeros(len(y))
  dydt[0] = 0.
  dydt[1] = 0.


  #rho =
  Ye = 0.5
  for n in range(nubins):
    nu = nus[n]
    #jnu =
    #alphanu =

    dydt[2+n] = cgs['CL']*(opac.jnu(rho0, Tg, Ye0, nu))

    #dInu[n] = cgs['CL']*dt*(jnu(T, nu) - alphanu(T, nu)*Inu[n])
    #dInu[n] = (Inu[n] + cgs['CL']*dt*jnu(T, nu))/(1. + cgs['CL']*dt*alphanu(T, nu)) - Inu[n]
    #Inu[n] += dInu[n]
    #Inu[n] = max(Inu[n], 1.e-100)

  return dydt

from scipy.integrate import ode
r = ode(f).set_integrator('vode', method='bdf')
rho0 = dfile0.Get('p.density', flatten=False)
r.set_initial_value([1., 0., 1., 1.], 0.).set_f_params((2, [1,2], eos, opac, rho0.mean(), 0.5))
print(r.integrate(r.t + 0.1))

import sys
sys.exit()


# Set up the axes
fig, ax = plt.subplots(figsize=[12,8])
ax.plot(t, J, color='r')
ax.set_xlim([0, None])
ax.set_ylim([0, None])
ax.set_xlabel('t')
ax.set_ylabel('J')
plt.show()

sys.exit()


# Burrows-Reddy-Thompson opacity
Fc = 4.543791885043567014 # Fermi coupling constant (erg^-2)
sigma0 = 4.*Fc**2*cgs['HBAR']**2*cgs['ME']**2*cgs['CL']**6/np.pi
gA = -1.23
Deltanp = 2.072126995e-6 # erg
mu = cgs['MP']
zeta3 = 1.2020569031595942853
zeta5 = 1.0369277551433699263
u0 = rho0*cgs['KBOL']*Tg0/(mu*(gamma - 1.))

def rad_E_to_T(E):
  return pow(15.*cgs['CL']**3*cgs['HPL']**3*E/(7.*cgs['KBOL']**4*np.pi**5*NSPECIES),1./4.)

def rad_T_to_E(T):
  return 7.*cgs['KBOL']**4*np.pi**5*NSPECIES*T**4/(15.*cgs['CL']**3*cgs['HPL']**3)

def gas_u_to_T(u):
  return u*mu*(gamma - 1.)/(rho0*cgs['KBOL'])

def gas_T_to_u(T):
  return rho0*cgs['KBOL']*T/(mu*(gamma - 1.))

def Bnu(T, nu, NSPECIES=3):
  x = cgs['HPL']*nu/(cgs['KBOL']*T + np.nextafter(0, 1))
  return NSPECIES*(2*cgs['HPL']*nu**3)/(cgs['CL']**2)/(np.exp(x) + 1.)

def sigmac(T, nu):
  return sigma0*((1. + 3.*gA**2)/4.)*((cgs['HPL']*nu + Deltanp)/(cgs['ME']*cgs['CL']**2))**2

def alphanu(T, nu):
  return rho0/mu*sigmac(T, nu)

def jnu(T, nu):
  return rho0/mu*sigmac(T, nu)*Bnu(T, nu)

def J(T):
  j = (1. + 3.*gA**2)*(cgs['KBOL']*T)**4*rho0*sigma0
  j *= (310.*(cgs['KBOL']*T)**2*np.pi**6 +
        147.*np.pi**4*Deltanp**2 +
        113400.*cgs['KBOL']*T*Deltanp*zeta5)
  j /= (5040.*cgs['CL']**6*cgs['HPL']**3*cgs['ME']**2*mu)
  return 4.*np.pi*j

# Estimate cooling time
tc = u0/J(Tg0)
print("Cooling time: %e s" % tc)

# Ensure tmax is divisible by dt for best accuracy!
tmax = 1.e-1*tc
print("Final time: %e s" % tmax)
dt = min(1.e-2*tc, 1.e-2*tmax)
print("dt: %e s" % dt)
t = 0
cycle = 0

ncycle = int(tmax/dt)

t = np.zeros(ncycle+1)
u = np.zeros(ncycle+1)
E = np.zeros(ncycle+1)
Tg = np.zeros(ncycle+1)
Tr = np.zeros(ncycle+1)

t[0] = 0
Tg[0] = Tg0
Tr[0] = Tr0
u[0] = gas_T_to_u(Tg[0])
E[0] = rad_T_to_E(Tr[0])

# Initialize neutrino spectrum
Inu = np.zeros(nnu)
for n in range(nnu):
  Inu[n] = max(1.e-100, Bnu(Tr[0], nus[n]))

for cycle in range(ncycle):
  phoebus_utils.progress_bar((cycle+1)/ncycle)
  import time
  #print(f"cycle = {cycle} t = {t[cycle]}")
  #T = u[cycle]*mu*(gamma - 1.)/(rho0*cgs['KBOL'])
  T = Tg[cycle]
  dE = 0
  dInu = np.zeros(nnu)
  for n in range(nnu):
    nu = nus[n]

    #dInu[n] = cgs['CL']*dt*(jnu(T, nu) - alphanu(T, nu)*Inu[n])
    dInu[n] = (Inu[n] + cgs['CL']*dt*jnu(T, nu))/(1. + cgs['CL']*dt*alphanu(T, nu)) - Inu[n]
    Inu[n] += dInu[n]
    Inu[n] = max(Inu[n], 1.e-100)

  for n in range(nnu - 1):
    dE += 4.*np.pi/cgs['CL']*(dInu[n+1] + dInu[n])/2*(nus[n+1]-nus[n])

  E[cycle+1] = E[cycle] + dE
  u[cycle+1] = u[cycle] - dE
  Tr[cycle+1] = pow(15.*cgs['CL']**3*cgs['HPL']**3*E[cycle+1]/(7.*cgs['KBOL']**4*np.pi**5*NSPECIES),1./4.)
  Tg[cycle+1] = u[cycle+1]*mu*(gamma - 1.)/(rho0*cgs['KBOL'])


  t[cycle+1] = t[cycle] + dt

fig, ax = plt.subplots(1,2,figsize=(12,5))
B0 = Bnu(Tg0, nus)
B = Bnu(T, nus)
ax[0].plot(nus, Inu, color='r', label='Final Inu')
ax[0].plot(nus, B0, color='k', linestyle=':', label='Initial Bnu')
ax[0].plot(nus, B, color='k', linestyle='--', label='Final Bnu')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim([1.e-10*B.max(), 1.e1*B.max()])
ax[0].set_xlabel('nu (Hz)')
ax[0].set_ylabel('Intensity (erg cm^-2 s^-1 Sr^-1 Hz^-1)')
ax[0].legend(loc=4)
ax[1].plot(t, Tr, label='Tr')
ax[1].plot(t, Tg, label='Tg')
ax[1].set_yscale('log')
ax[1].set_xlabel('t (s)')
ax[1].set_ylabel('Temperature (K)')
ax[1].legend(loc=4)
ax[1].set_xlim([0, t[-1]])
fig.subplots_adjust(wspace=0.5)
plt.show()