from os import system
import numpy as np
import matplotlib.pyplot as pl
import h5py
import re
from glob import glob

class Dump1D:
    def __init__(self, filename):
        with h5py.File(filename, "r") as f:
            #print(f.keys())
            #print(f["Info"].attrs.keys())
            self.nx,self.ny,self.nz = f["Info"].attrs["MeshBlockSize"]
            self.NumMB = f["Info"].attrs["NumMeshBlocks"]
            #print(f["Params"].attrs.keys())
            #print(np.shape(f["Params"].attrs["monopole_gr/shift"]))
            self.varkeys = list(f.keys())[8:]
            #print(f["Locations"].keys())
            self.t = f["Info"].attrs["Time"]
            self.xf = f["Locations/x"][:,:]
            self.xc = 0.5 * (self.xf[:,1:] + self.xf[:,:-1])
            self.yf = f["Locations/y"][:,:]
            self.yc = 0.5 * (self.yf[:,1:] + self.yf[:,:-1])
            self.zf = f["Locations/z"][:,:]
            self.zc = 0.5 * (self.zf[:,1:] + self.zf[:,:-1])
            #self.rhoc = f["c.density"]
            #print(self.rhoc.shape)
            self.var = {}
            for key in self.varkeys:
                self.var[key] = f[key][:,0,0,:] #ib,iz,iy,ix

            try:
                self.var['monopole_gr/lapse_h'] = f["Params"].attrs['monopole_gr/lapse_h']
                self.var['monopole_gr/hypersurface_h']=f["Params"].attrs["monopole_gr/hypersurface_h"]
                self.var['monopole_gr/shift']=f["Params"].attrs["monopole_gr/shift"]
                nxgr = np.size(self.var['monopole_gr/lapse_h'])
                rout = f["Params"].attrs['monopole_gr/rout']
                rin = f["Params"].attrs['monopole_gr/rin']
                self.var['monopole_gr/radius'] = np.linspace(rin,rout,num=nxgr)
            except:
                pass
        return

class DumpGR:
    def __init__(self, filename):
        with h5py.File(filename, "r") as f:
            #print(f["Params"].attrs.keys())
            #print(np.shape(f["Params"].attrs["monopole_gr/shift"]))
            #exit()
            self.t = f["Info"].attrs["Time"]
            self.var = {}

            try:
                self.var['monopole_gr/rhoadm'] = f["Params"].attrs['monopole_gr/matter_h'][0,:]
                self.var['monopole_gr/lapse_h'] = f["Params"].attrs['monopole_gr/lapse_h']
                self.var['monopole_gr/hypersurface_h']=f["Params"].attrs["monopole_gr/hypersurface_h"]
                self.var['monopole_gr/shift']=f["Params"].attrs["monopole_gr/shift"]
                nxgr = np.size(self.var['monopole_gr/lapse_h'])
                rout = f["Params"].attrs['monopole_gr/rout']
                rin = f["Params"].attrs['monopole_gr/rin']
                self.var['monopole_gr/radius'] = np.linspace(rin,rout,num=nxgr)
            except:
                pass
        return

def PlotResolution1D(data,idump=0,iofile='PlotResolution1D.png'):

    pl.clf()
    dx = data[idump].xf[:,1:] - data[idump].xf[:,:-1]
    for ib in range(data[idump].NumMB):
        pl.plot(data[idump].xc[ib,:],dx[ib,:],color='black')
    pl.xlabel('x')
    pl.ylabel('dx')
    pl.ylim(bottom=0.)
    pl.savefig(iofile)
    
    return

def Movie1D(data,varname='p.density',anax=None,anay=None,ylim=None,xlim=None):
    #Arguments:
    #data = list of Data1D instances; there is one for each file.
    #anax = if there is an analytic solution to plot, then 
    ndata = len(data)
    for i in range(ndata):
        iofile=f'img{i:04d}.png'
        print(iofile,data[i].t)
        pl.clf()
        for ib in range(data[i].NumMB):
            pl.plot(data[i].xc[ib,:],data[i].var[varname][ib,:],color='red')
        if (anax is not None):
            pl.plot(anax,anay,linestyle='dashed',color='black')
        if (ylim is not None):
            pl.ylim(ylim)
        if (xlim is not None):
            pl.xlim(xlim)
        pl.yscale('log')
        pl.ylabel(varname)
        pl.xlabel('xc')
        pl.savefig(iofile)


    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p movie.{varname}.mpeg")
    system("rm img????.png")
    
    return

def MovieGR(data,varname='monopole_gr/rhoadm',anax=None,anay=None,ylim=None,xlim=None):
    #Arguments:
    #data = list of Data1D instances; there is one for each file.
    #anax = if there is an analytic solution to plot, then 
    ndata = len(data)
    for i in range(ndata):
        iofile=f'img{i:04d}.png'
        print(iofile,data[i].t)
        pl.clf()
        pl.plot(data[i].var['monopole_gr/radius'],data[i].var[varname],color='red')
        if (anax is not None):
            pl.plot(anax,anay,linestyle='dashed',color='black')
        if (ylim is not None):
            pl.ylim(ylim)
        if (xlim is not None):
            pl.xlim(xlim)
        pl.ylabel(varname)
        pl.xlabel('radius')
        pl.savefig(iofile)


    mvname = varname.replace('/','-')
    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p movieGR.{mvname}.mpeg")
    system("rm img????.png")
    
    return

def ReadHistory(fname='tov.hst'):
    #Get Data
    histdata = np.loadtxt(fname)
    #Get Variable Names
    f = open(fname,"r")
    line = f.readline()
    line = f.readline()
    vars = line.split("=")
    vars2 = [re.sub(r'\[.*\]','',var).strip() for var in vars[1:]]
    print(vars2)
    print(vars[1:])
    f.close()
    
    return histdata
    
def main():
    #List of outfile names
    filenames = sorted(glob(f"*.out1.*.phdf"))
    nfiles = len(filenames)
    #List of data dumps for each file
    data = [Dump1D(fnam) for fnam in filenames]
    Movie1D(data)
    #for i in range(nfiles):
    #    print(data[i].rhop)
    
if (__name__=="__main__"):
    main()
