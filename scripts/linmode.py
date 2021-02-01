PHDF_PATH = '/home/brryan/rpm/phoebus/external/parthenon/scripts/python/'
res = [16, 32, 64, 128, 256]
#res = [128, 256, 512, 1024]

rho0 = 1
amp = 1.e-5
mode = 'sound'

# Sound

import numpy as np
#import h5py
import sys
import matplotlib.pyplot as plt
import shutil
import os
from subprocess import call, DEVNULL
import glob
sys.path.append(PHDF_PATH)
import phdf

if mode == "entropy":
  omega = 0 + (2.*np.pi/10.)*1j
  drho_mode = 1.
  u1 = 0.1
elif mode == "sound":
  omega = 0 + 0.6568547144496073j
  drho_mode = 0.9944432913027026
  u1 = 0

if len(sys.argv) != 3:
  print("ERROR: format is")
  print("  python linmode.py [executable] [inputfile]")
  sys.exit()

EXECUTABLE = sys.argv[1]
TMPINPUTFILE = 'tmplinmode.pin'
shutil.copyfile(sys.argv[2], TMPINPUTFILE)

k = 2.*np.pi
def rho_mode(x, t):
  #print(amp)
  #print(drho_mode)
  #print(np.exp(1j*k*x - omega*t))
  return (amp*drho_mode*np.exp(1j*k*(x - u1*t) - omega*t)).real

res = np.array(res)
data = {}

L1 = np.zeros(res.size)

for n, N in enumerate(res):
  # Process input file
  with open(TMPINPUTFILE, 'r') as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
      if (line.startswith("nx1")):
        lines[i] = "nx1 = %i" % N + "\n"
        #break
      if (line.startswith("mode")):
        lines[i] = "mode = " + mode + "\n"
      if (line.startswith("amplitude")):
        lines[i] = "amplitude = %e" % amp + "\n"
  with open(TMPINPUTFILE, 'w') as outfile:
    outfile.writelines(lines)

  # Run simulation
  print('Running problem at %i zones... ' % N, end='')
  call([EXECUTABLE, '-i', TMPINPUTFILE], stdout=DEVNULL, stderr=DEVNULL)
  print('done')

  dumps = np.sort(glob.glob('hydro_modes*.phdf'))

  dump = phdf.phdf(dumps[-1])
  t = dump.Time
  x = dump.x[0,:]
  #print(dump.Variables)
  drho = dump.Get('p.density') - rho0


  drho_soln = rho_mode(x, t)

  L1[n] = np.fabs(drho - drho_soln).sum()/(N*amp)
  data[N] = (x, drho, L1)

  plt.figure()
  ax = plt.gca()
  plt.plot(x, drho, color='r', label='drho')
  plt.plot(x, drho_soln, color='k', label='drho solution', linestyle='--')
  plt.xlim([0, 1])
  plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
  plt.legend(loc=1)
  plt.xlabel('x')
  plt.show()

#  if (n > 0):
#    sys.exit()

  # Clean up dump files
  for dump in glob.glob('hydro_modes*.phdf*'):
    os.remove(dump)

print(L1)

fit = np.polyfit(np.log(res), np.log(L1), deg=1)
print(fit)
fitx = res
fity = np.exp(fit[0]*np.log(fitx) + fit[1])

plt.figure()
ax = plt.gca()
plt.plot(res, L1, color='r', label='$\\delta \\rho$', linestyle='', marker='s')
plt.plot(fitx, fity, color='k', linestyle='--', label='$N^{%g}$' % fit[0])
#plt.xlim([0, 1])
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(loc=1)
plt.xlabel('N')
plt.ylabel('L1')
plt.xscale('log')
plt.yscale('log')
#plt.xticks([2, 4, 8, 16, 32, 64])
ax.set_xticks(res)
ticks = []
for N in res:
  ticks.append(str(N))
#ax.set_xticklabels(['2', '4', '8', '16'])
ax.set_xticklabels(ticks)
plt.show()



#os.remove(TMPINPUTFILE)
sys.exit()

with h5py.File(fnam, 'r') as dfile:
  print('File :', fnam)
  print('  Keys: ')
  for key in dfile.keys():
    print('    ', key)
  info = dfile['Info']
  #x = np.array(dfile['x'])
  #print('    ' + info)
  print(np.array(dfile['Locations']))
  rho = np.array(dfile['p.density'][0,0,0,:,0])
  drho = rho - rho0
  print(rho.shape)

x = np.linspace(0,1,drho.size)

plt.figure()
ax = plt.gca()
plt.plot(x, drho, color='r', label='drho', linestyle='', marker='s')
plt.xlim([0, 1])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(loc=1)
plt.xlabel('x')
plt.show()
