### This script converts adm quantities in Phoebus units.



import numpy as np
from sys import exit
import math
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as pl
G=6.68e-8
c=2.989e10
msun=1.989e33

def ConvertData(filename,formattype):
    data=ascii.read(filename, format=formattype)
    rhoc=max(data['mass_density'])
    M0=1./(rhoc)**(0.5)*(c**2./G)**(1.5)
    newdata=Table()
    newdata['r']=data['r']*c**2./G/M0
    #np.save('r',newdata['r'])
    newdata['mass_density']=data['mass_density']*(G/c**2.)**3.*M0**2.
    newdata['temp']=data['temp']
    newdata['Ye']=data['Ye']
    newdata['specific_internal_energy']=data['specific_internal_energy']/c**2.
    newdata['velocity']=data['velocity']/c
    newdata['pressure']=data['pressure']*G**3./c**8.*M0**2.
    dvdr=(newdata['velocity'][1:]-newdata['velocity'][:-1])/(newdata['r'][1:]-newdata['r'][:-1])
    newdata['adm_density']=data['adm_density']*(G/c**2.)**3.*M0**2.
    newdata['adm_momentum']=data['adm_momentum']*G**3./c**7.*M0**2.
    newdata['S_adm']=data['S_adm']*G**3./c**8.*M0**2.
    newdata['Srr_adm']=data['Srr_adm']*G**3./c**8.*M0**2.
    ascii.write(newdata,'converted_'+filename,overwrite=True,format=formattype)
    return


filename='ADM_stellartable.dat'
#filename='ADM_homologouscollapse.dat'
#filename='ADM_tov.dat'
formattype='commented_header'
ConvertData(filename,formattype)
