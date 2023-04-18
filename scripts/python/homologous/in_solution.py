# Code generates solution for homologous collpasing core (Goldreich & Weber). 

import numpy as np
import math
import time
import matplotlib.pyplot as pl
import sys
import glob
from scipy.integrate import odeint
from astropy import constants as const

G=const.G.cgs.value
c=const.c.cgs.value
msun=const.M_sun.cgs.value

def solvef(rs,lmbda,doplot=False):    #Solves the second order equation for density normalization function f (eq. 16)
    def func(u,x):
        return (u[1],-2./x*u[1]-u[0]**3.+lmbda)

    f0=[1,0]
    fs=[]
    r=[]
    us=odeint(func,f0,rs)
    for i in range(len(rs)):
        if (us[i,0]<0):
            break
        fs.append(us[i,0])
        r.append(rs[i])
        
    if (doplot==True):
        pl.plot(rs,fs)
        pl.xlabel(r'$r=R/\lambda_J$')
        pl.ylabel(r'$f$')
        pl.title(r'$\lambda=0.002$')
        pl.savefig('f.png')
        pl.show()
    return np.array(r),fs 

def FindJeansLength(r,R,lmbda,k):    # eq. 15
    a=[R/r[len(r)-1]]
    a=np.array(a)
    t=(a/(6.*lmbda)**(1./3.)*(np.pi*G/k**3.)**(1./6.))**(3./2.)
    adot=2./3.*(6.*lmbda)**(1./3.)*(k**3./(math.pi*G))**(1./6.)*t**(-1./3.)
    return a, adot

def CalculateK(M,f,r):
    I=0
    for i in range(len(r)-1):
        I=I+(f[i+1]**3.*r[i+1]**2.+f[i]**3.*r[i]**2.)/2.*(r[i+1]-r[i])
    k=math.pi*G*(M/(4.*math.pi*I))**(2./3.)
    return k
        
def CalculateDensity(a,f,k): # eq. 10
    rho=np.zeros([len(a),len(f)])
    for i in range(len(a)):
        for j in range(len(f)):
            rho[i][j]=(k/(math.pi*G))**(3./2.)*a[i]**(-3.)*f[j]**3.
    rhoc=max(rho[0,:])
    for i in range(len(a)):
        for j in range(len(f)):
            if(rho[i][j]/rhoc<1.e-5):
                rho[i][j]=1.e-5*rhoc
    return rho

def CalculateVelocity(r,adot): #u=adot*r (r-component)
    u=np.zeros([len(adot),len(r)])
    for i in range(len(adot)):
        for j in range(len(r)):
            u[i][j]=adot[i]*r[j]
    return u

def CalculateGravPotential(r,a,f,lmbda,k):
    phi=np.zeros([len(a),len(r)])
    for i in range(len(a)):
        for j in range(len(r)):
            psi=1/2.*lmbda*r[j]**2.-3.*f[j]
            phi[i][j]=4./3.*(k**3/(math.pi*G))**(1./2.)*psi/a[i]
    return phi

def CalculateSpecificInternalEnergy(rho,k,gamma):
    eps=np.zeros([len(rho[:,0]),len(rho[0,:])])
    for i in range(len(rho[:,0])):
        for j in range(len(rho[0,:])):
            eps[i][j]=k/(gamma-1.)*rho[i][j]**(gamma-1.)
    return eps

def CalculatePressure(rho,k,gamma):
    press=k*rho**gamma
    return press

def CalculateTemperature(cv,eps):
    return eps/cv

def GenerateProfile(savedata=False):

    ##Setup
    M=1.4*msun
    R=3.e8
    t=1.*np.ones(1);
    lmbda=0.002
    rs=np.linspace(0.001,10.,10000)
    cv=1.0
    gamma=4./3.

    r,f=solvef(rs,lmbda)
    k=CalculateK(M,f,r)
    a,adot=FindJeansLength(r,R,lmbda,k)
    
    vel=CalculateVelocity(r,adot) 
    rho=CalculateDensity(a,f,k)
    phi=CalculateGravPotential(r,a,f,lmbda,k) 
    eps=CalculateSpecificInternalEnergy(rho,k,gamma)
    press=CalculatePressure(rho,k,gamma)
    temp=CalculateTemperature(cv,eps)
    rhoe=np.zeros(len(r))
    r_data=r*a[0] ### dimensionful
    for n in range(len(r)):
        rhoe[n]=rho[0,n]+rho[0,n]*eps[0,n]/c**2.
    mass=0
    for i in range(len(r)-1):
        mass=mass+2.*np.pi*(rho[0,i+1]*(a[0]*r[i+1])**2.+rho[0,i]*(a[0]*r[i])**2.)*a[0]*(r[i+1]-r[i])
    
    if (savedata==True):
        np.save("r",r_data)
        np.save("v",-vel[0,:])
        np.save("v_ang",np.zeros(len(r)))
        np.save("p",press[0,:])
        np.save("rho_m",rho[0,:])
        np.save("eps",eps[0,:])
        np.save("ye",0.5*np.ones(len(r)))
        np.save("temp",temp[0,:])
        np.save("rho",rhoe)
    return 


GenerateProfile(True)
    
     
