#THE MAIN CODE FOR GR-SOLVER THAT SOLVES FOR METRIC IN 3+1 FORMALISM. THE STARTING POINT IS NEWTONIAN APPROXIMATION FOR METRIC. BASED ON THE NEWTONIAN METRIC, WE FIND ADM QUANTITIES AND THEN SOLVE FOR METRIC. ITERATE UNTIL EXACT. CODE SOLVES FOR TOV STAR AS WELL AS FOR DATA INPUT FROM STELLAR EVOLUTION TABLE.  


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
from sys import exit
import math
from astropy.io import ascii
from astropy.table import Table
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp 
from scipy import optimize
from numpy import diff
from seos import CalculateInternalEnergy


G=6.68e-8
c=2.989e10

class GR_Solver:
        def __init__(self,prob,R,Ni):
                self.problem=prob
                self.R=R
                self.Ni=Ni
                return

        # IMPORT DATA
        def Data(self):
                if (self.problem=='homologouscollapse'):
                        loc='/home/mari/Work/Phoebus/phoebus/Tests/HomologousCollapse/'
                        r_data=np.load(loc+'r.npy')
                        v=np.load(loc+'v.npy')
                        v_ang=np.load(loc+'v_ang.npy')
                        rho_m=np.load(loc+'rho_m.npy')
                        rho=np.load(loc+'rho.npy')
                        p=np.load(loc+'p.npy')
                        eps=np.load(loc+'eps.npy')
                        temp=np.load(loc+'temp.npy')
                        ye=np.load(loc+'ye.npy')
                        
                if (self.problem=='tov'):
                        r_data=np.load('TOV_DATA/tov_r.npy')*1.e2
                        v=np.zeros(len(r_data))
                        v_ang=np.zeros(len(r_data))
                        rho_m=np.load('TOV_DATA/tov_density.npy')/G  ### mass density
                        rho=np.load('TOV_DATA/tov_energydensity.npy')*c**2./G  ### energy density
                        p=np.load('TOV_DATA/tov_pressure.npy')*c**4./G
                        eps=np.load('TOV_DATA/tov_specificinternalenergy.npy')*c**2.   ### specific internal energy
                        temp=np.load('TOV_DATA/tov_temp.npy')
                        ye=np.load('TOV_DATA/tov_ye.npy')
                        a2=np.load('TOV_DATA/a2.npy')
                        alpha2=np.load('TOV_DATA/alpha2.npy')
                        
                if (self.problem=='stellartable'):
                        r_data=[]
                        v=[]
                        v_ang=[]
                        rho_m=[]   ### mass density
                        p=[]
                        ye=[]
                        temp=[]
                        sie=[]
                        #name='s'+m+'_presn'
                        name='s12.0_presn'
                        loc='/home/mari/Work/Phoebus/phoebus/Tests/GRsolver/'+name
                        data0 =ascii.read(loc,header_start=1,data_start=2,delimiter="\t",guess=False)
                        for line in data0:
                                spl=line[0].split()
                                r_data.append(spl[2])
                                v.append(spl[3])
                                v_ang.append(spl[9])
                                rho_m.append(spl[4])
                                p.append(spl[6])
                                ye.append(spl[11])
                                temp.append(spl[5])
                                sie.append(spl[7])
                ind=np.where(np.array(r_data,dtype=float)>max(self.R))[0][0]
                r_data=np.array(r_data[0:ind],dtype=float)
                v=np.array(v[0:ind],dtype=float)
                v_ang=np.array(v_ang[0:ind],dtype=float)
                rho_m=np.array(rho_m[0:ind],dtype=float)
                p=np.array(p[0:ind],dtype=float)
                temp=np.array(temp[0:ind],dtype=float)
                sie=np.array(sie[0:ind],dtype=float)
                ye=np.array(ye[0:ind],dtype=float)
                if (self.problem=='stellartable'):
                        eosfilename='SFHo.h5'
                        eps,u,temp1=CalculateInternalEnergy(rho_m,ye,p,1e8,1e12,eosfilename)
                        rho=rho_m+u/c**2.   ### energy density
                        #eps=u/rho_m    ### specific internal energy
                        print('temp_ratio=',temp1/temp)
                        print('eps_ratio=',eps/sie)

        # RETURN ORIGINAL DATA FOR GRID FROM DATA
                self.r0=r_data
                self.rho=rho
                if (self.problem=='tov'):
                        self.a2_int=interp1d(r_data,a2)
                        self.alpha2_int=interp1d(r_data,alpha2)
        # RETURN INTERPOLATE FUNCTIONS TO BE ABLE TO CALCULATE THESE QUANTITIES FOR ANY GIVEN RADIUS VECTOR    
                self.v_int=interp1d(r_data,v,fill_value='extrapolate')
                self.v_ang_int=interp1d(r_data,v_ang,fill_value='extrapolate')
                self.rho_int=interp1d(r_data,rho,fill_value='extrapolate')
                self.p_int=interp1d(r_data,p,fill_value='extrapolate')
                self.rho_m_int=interp1d(r_data,rho_m,fill_value='extrapolate')
                self.ye_int=interp1d(r_data,ye,fill_value='extrapolate')
                self.temp_int=interp1d(r_data,temp,fill_value='extrapolate')
                self.eps_int=interp1d(r_data,eps,fill_value='extrapolate')
                return 
                                
        
        # METRIC FOR NEWTONIAN APPROXIMATION
        def CalculateMetricForNewtonian(self,r): #### INCLUDES COORDINATE TRANSFORMATION TO ARIAL COORDINATES.
                phi=np.zeros(len(r))
                dphi=np.zeros(len(r))
                for i in range(len(r)):
                        I=-2.*np.pi*G*self.rho*self.r0/r[i]*(r[i]+self.r0-abs(r[i]-self.r0))
                        phi[i]=np.trapz(I,x=self.r0)
                dphi=np.gradient(phi,r)
                alpha2=1.+2.*phi/c**2.
                a2=1.+2.*r*dphi/c**2.
                return  alpha2,a2       
                
        def CalculateInialADM(self,rad):
                gamma2=1./(1.-(self.v_int(rad)**2.+(self.v_ang_int(rad)*rad)**2.)/c**2.)
                        
                if(self.problem=='tov'):
                        rho_adm=self.alpha2_int(rad)*self.rho_int(rad)*gamma2
                        P_adm=np.zeros(len(rad))   # P_adm=alpha*(rho+p)*gamma2*v
                        S_adm=(self.alpha2_int(rad)*gamma2-1.)*self.rho_int(rad)*c**2.+(self.alpha2_int(rad)*gamma2+2.)*self.p_int(rad)
                        
                if (self.problem=='stellartable') or (self.problem=='homologouscollapse'):
                        alpha2,a2=self.CalculateMetricForNewtonian(rad)
                        rho_adm=alpha2*self.rho_int(rad)*gamma2
                        P_adm=np.sqrt(alpha2)*gamma2*(self.rho_int(rad)+self.p_int(rad)/c**2.)*self.v_int(rad)  ## r-component
                        S_adm=(alpha2*gamma2-1.)*self.rho_int(rad)*c**2.+(alpha2*gamma2+2.)*self.p_int(rad)
                        self.alpha2_int=interp1d(rad,alpha2)
                        self.a2_int=interp1d(rad,a2)
                return rho_adm, P_adm,S_adm 
                

        def CalculateMetric(self,r,rho_adm, P_adm,S_adm):
################# SOlve for a and Krr:
                
                ## INTERPOLATE
                rho_adm_int=interp1d(r,rho_adm)
                j_adm_int=interp1d(r,P_adm)
                ## INTEGRANDS
                def f(x,V):
                        da=V[0]/8./x*(32.*np.pi*x**2.*V[0]**2*rho_adm_int(x)*G/c**2.+3.*x**2.*V[0]**2.*V[1]**2.+4.-4.*V[0]**2.)
                        dK=8.*np.pi*V[0]**2.*j_adm_int(x)*G/c**3.-3./x*V[1]
                        return [da,dK]
                        
                ## INITIAL CONDITIONS
                V0=[1,0]   # at r=0
                
                ## SOLVE COUPLED DIFFERENTIAL EQUATIONS FOR EXTRINSIC CURVATURE
                
                sol=solve_ivp(f,[min(r),max(r)],V0,t_eval=r,vectorized=True)
                result=sol.y
                a=result[0]
                K=result[1]



################# Solve for lapse                                       
                ## INITIALIZE MATRIX
                da=a/8./r*(32.*np.pi*r**2.*a**2.*rho_adm*G/c**2.+3.*r**2*a**2*K**2+4-4.*a**2)
                        ## BOUNDARY CONDITIONS
        
                M=np.zeros([len(r),len(r)])
                V=np.zeros(len(r))  # <--------  MX+V=0
                eps=r[1]-r[0]
                A0=1./a[0]**2/eps**2+da[0]/(2*a[0]**3.*eps)-1./(a[0]**2.*r[0]*eps)
                B0=-(2./(a[0]**2.*eps**2)+4*np.pi*G/c**2.*(S_adm[0]/c**2.+rho_adm[0])+3./2.*K[0]**2.)
                C0=1./(a[0]**2.*eps**2.)-da[0]/(2.*a[0]**3.*eps)+1./(a[0]**2.*r[0]*eps)
                ##V[0]=A0
                M[0,0]=A0+B0
                M[0,1]=C0
                AL=1/a[len(r)-1]**2/eps**2+da[len(r)-1]/(2*a[len(r)-1]**3.*eps)-1./(a[len(r)-1]**2.*r[len(r)-1]*eps)
                BL=-(2./(a[len(r)-1]**2.*eps**2)+4*np.pi*G/c**2.*(S_adm[len(r)-1]/c**2+rho_adm[len(r)-1])+3./2.*K[len(r)-1]**2.)
                CL=1./(a[len(r)-1]**2.*eps**2.)-da[len(r)-1]/(2.*a[len(r)-1]**3.*eps)+1./(a[len(r)-1]**2.*r[len(r)-1]*eps)
                V[len(r)-1]=eps/r[len(r)-1]*CL
                M[len(r)-1,len(r)-2]=AL
                M[len(r)-1,len(r)-1]=BL+(1.-eps/r[len(r)-1])*CL

                
                ## THE REST OF THE MATRIX       
                for i in range(1,len(r)-1):
                        A=1/a[i]**2/eps**2+da[i]/(2*a[i]**3.*eps)-1./(a[i]**2.*r[i]*eps)
                        B=-(2./(a[i]**2.*eps**2)+4*np.pi*G/c**2.*(S_adm[i]/c**2.+rho_adm[i])+3./2.*K[i]**2.)
                        C=1./(a[i]**2.*eps**2.)-da[i]/(2.*a[i]**3.*eps)+1./(a[i]**2.*r[i]*eps)
                        M[i,i-1]=A
                        M[i,i]=B
                        M[i,i+1]=C
                def f(x):
                        return np.dot(M,x)+V
                sol = optimize.root(f,100*np.ones(len(r)))
                alpha=sol.x
        
        
################# Solve for shift
                beta=-a**2./2.*alpha*r*K                                                
                return a,K,alpha,beta
                
        def CalculateADM(self,r,a,K,alpha,beta):        
                rho_adm=np.zeros(len(r))
                P_adm=np.zeros(len(r))
                S_adm=np.zeros(len(r))
                Srr_adm=np.zeros(len(r))

                for i in range(len(r)):
                        #upper metric
                        g00=-1./alpha[i]**2.   
                        g0r=beta[i]/alpha[i]**2.                
                        gamma2=1./(1-self.v_int(r[i])**2./c**2.)
                        rho_adm[i]=alpha[i]**2.*((self.rho_int(r[i])+self.p_int(r[i])/c**2.)*gamma2-self.p_int(r[i])/c**2.*g00)
                        P_adm[i]=alpha[i]*beta[i]*((self.rho_int(r[i])+self.p_int(r[i])/c**2.)*gamma2+self.p_int(r[i])/c**2.*g00)+alpha[i]*((self.rho_int(r[i])+self.p_int(r[i])/c**2.)*gamma2*self.v_int(r[i])+self.p_int(r[i])/c**2.*g0r)   ## r-component  (upper index)
                        S_adm[i]=(alpha[i]**2.*gamma2-1.)*self.rho_int(r[i])*c**2.+(alpha[i]**2.*gamma2+2.)*self.p_int(r[i])
                        Srr_adm[i]=self.rho_int(r[i])*self.v_int(r[i])**2.*gamma2*a[i]**4.+self.p_int(r[i])*a[i]**4.*(1./a[i]**2-beta[i]**2./alpha[i]**2.+self.v_int(r[i])**2./c**2.*gamma2)
                return rho_adm,P_adm,S_adm,Srr_adm


        def Iterate(self,doplot):
                ### INITIAL ADM QUANTITIES
                rho_adm, P_adm,S_adm=self.CalculateInialADM(self.R)
                rho_adm=rho_adm
                r=self.R
                alpha_prev=np.sqrt(self.alpha2_int(r))
                for i in range(self.Ni):
                        print(i)
                        a,K,alpha,beta=self.CalculateMetric(r,rho_adm, P_adm,S_adm)
                        if (np.max(abs(alpha_prev-alpha))<1.e-12):
                                break
                        if (i==99):
                                print("ERROR: HAS NOT CONVERGED AFTER 100 ITERATIONS")
                        alpha_prev=alpha
                        rho_adm,P_adm,S_adm,Srr_adm=self.CalculateADM(r,a,K,alpha,beta)
                        if (doplot==True):
                                pl.plot(r/1.e5,alpha)
                if (doplot==True):
                        pl.xlabel(r'$r$'+' '+'[km]')
                        pl.ylabel(r'$\alpha$')
                        if (self.problem=='tov'):
                                pl.savefig('tov_alpha.png')
                        if (self.problem=='stellartable'):
                                pl.savefig('st_a.png')
                        if (self.problem=='homologouscollapse'):
                                pl.savefig('hom_alpha.png')
        
                return rho_adm,P_adm,S_adm,Srr_adm, a, K, alpha

##### EXTRAPOLATE DATA TO INCLUDE r=0 
        def ExtrapolateData(self):
                r=np.linspace(0,max(self.R),10000)
                rho_adm0,P_adm0,S_adm0,Srr_adm0,a0,K0,alpha0=self.Iterate(doplot=False)
                rho_adm=interp1d(self.R,rho_adm0,fill_value='extrapolate')(r)
                P_adm=interp1d(self.R,P_adm0,fill_value='extrapolate')(r)
                S_adm=interp1d(self.R,S_adm0,fill_value='extrapolate')(r)
                Srr_adm=interp1d(self.R,Srr_adm0,fill_value='extrapolate')(r)
                a=interp1d(self.R,a0,fill_value='extrapolate')(r)
                K=interp1d(self.R,K0,fill_value='extrapolate')(r)
                alpha=interp1d(self.R,alpha0,fill_value='extrapolate')(r)
                #np.save('a',a)
                #np.save('K',K)
                #np.save('alpha',alpha)

                return r, rho_adm, P_adm, S_adm, Srr_adm

        def SaveFinalData(self,filename):
                pl.clf()
                r, rho_adm, P_adm, S_adm, Srr_adm=self.ExtrapolateData()
                
                from astropy.io import ascii
                from astropy.table import Table
                data=Table()
                data['r']=r
                data['mass_density']=self.rho_m_int(r)
                data['temp']=self.temp_int(r)
                data['Ye']=self.ye_int(r)
                data['specific_internal_energy']=self.eps_int(r)
                data['velocity']=self.v_int(r)
                data['pressure']=self.p_int(r)
                data['adm_density']=rho_adm
                data['adm_momentum']=P_adm
                data['S_adm']=S_adm
                data['Srr_adm']=Srr_adm
                ascii.write(data,filename,overwrite=True,format='commented_header')

                return


def main():
        # INITIALIZE PROBLEM
        #problem='homologouscollapse'
        problem='stellartable'
        if (problem=='tov'):
                r=np.linspace(0.00001,2500,1000)*1.e2
        if(problem=='stellartable'):
                r=np.linspace(45,10000,1000)*1.e5
        if (problem == 'homologouscollapse'):
                r=np.load("/home/mari/Work/Phoebus/phoebus/Tests/HomologousCollapse/r.npy")

        Ni=100
        # INITIALIZE CLASS 
        GRsolver=GR_Solver(problem,r,Ni)
        GRsolver.Data()
        filename='ADM_'+problem+'.dat'
        GRsolver.SaveFinalData(filename)
        return
        
        
        
if (__name__=='__main__'):
        main()
