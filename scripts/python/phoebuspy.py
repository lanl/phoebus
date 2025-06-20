from os import system
import numpy as np
import matplotlib.pyplot as pl
import h5py
import re
from glob import glob

class Dump1D:
    def __init__(self, filename):
        with h5py.File(filename, "r") as f:
            #The various variables and states are embedded within several datasets.  The following print commands show the various keys.
            #print(f.keys())
            #print(f["Info"].attrs.keys())
            #print(f["Params"].attrs.keys())
            #print(f["Locations"].keys())
            self.nx,self.ny,self.nz = f["Info"].attrs["MeshBlockSize"]
            self.NumMB = f["Info"].attrs["NumMeshBlocks"]
            self.varkeys = list(f.keys())[8:]
            self.t = f["Info"].attrs["Time"]
            self.xf = f["Locations/x"][:,:]
            self.xc = 0.5 * (self.xf[:,1:] + self.xf[:,:-1])
            self.yf = f["Locations/y"][:,:]
            self.yc = 0.5 * (self.yf[:,1:] + self.yf[:,:-1])
            self.zf = f["Locations/z"][:,:]
            self.zc = 0.5 * (self.zf[:,1:] + self.zf[:,:-1])
            self.var = {}
            for key in self.varkeys:
                self.var[key] = f[key][:,0,0,:] #ib,iz,iy,ix

            try:
                self.var['monopole_gr/lapse_h'] = f["Params"].attrs['monopole_gr/lapse_h']
                self.var['monopole_gr/hypersurface_h']=f["Params"].attrs["monopole_gr/hypersurface_h"]
                self.var['monopole_gr/shift']=f["Params"].attrs["monopole_gr/shift"]
                self.var['monopole_gr/rhoadm'] = f["Params"].attrs['monopole_gr/matter_h'][0,:]
                nxgr = np.size(self.var['monopole_gr/lapse_h'])
                rout = f["Params"].attrs['monopole_gr/rout']
                rin = f["Params"].attrs['monopole_gr/rin']
                self.var['monopole_gr/radius'] = np.linspace(rin,rout,num=nxgr)
            except:
                pass
        return

class Dump3D:
    #So far this, the only difference between this and Dump1D is
    #the size of var[key].  They are inherently 1D in Dump1D and
    #3D in Dump3D.  Maybe we can unify this into one class at some point.
    def __init__(self, filename,extractvars=None):
        with h5py.File(filename, "r") as f:
            #The various variables and states are embedded within several datasets.  The following print commands show the various keys.
            #print(f.keys())
            #print(f["Info"].attrs.keys())
            #print(f["Params"].attrs.keys())
            #print(f["Locations"].keys())
            self.nx,self.ny,self.nz = f["Info"].attrs["MeshBlockSize"]
            self.NumMB = f["Info"].attrs["NumMeshBlocks"]
            if (extractvars is None):
                self.varkeys = ['p.density']
            elif (extractvars == 'all'):
                self.varkeys = list(f.keys())[8:]
            else:
                self.varkeys = extractvars
            self.t = f["Info"].attrs["Time"]
            self.xf = f["Locations/x"][:,:]
            self.xc = 0.5 * (self.xf[:,1:] + self.xf[:,:-1])
            self.yf = f["Locations/y"][:,:]
            self.yc = 0.5 * (self.yf[:,1:] + self.yf[:,:-1])
            self.zf = f["Locations/z"][:,:]
            self.zc = 0.5 * (self.zf[:,1:] + self.zf[:,:-1])
            #Making xgrid,ygrid,zgrid
            X = self.xc[:, np.newaxis, np.newaxis, :]  # shape: (Nb, 1, 1, Nx)
            Y = self.yc[:, np.newaxis, :, np.newaxis]  # shape: (Nb, 1, Ny, 1)
            Z = self.zc[:, :, np.newaxis, np.newaxis]  # shape: (Nb, Nz, 1, 1)
            # Now broadcast to (Nb, Nz, Ny, Nx)
            self.xgrid = np.broadcast_to(X, (self.NumMB, self.nz, self.ny, self.nx))
            self.ygrid = np.broadcast_to(Y, (self.NumMB, self.nz, self.ny, self.nx))
            self.zgrid = np.broadcast_to(Z, (self.NumMB, self.nz, self.ny, self.nx))
            self.var = {}
            for key in self.varkeys:
                print(f"Reading {key} from {filename}...",end="")
                self.var[key] = f[key][:,:,:,:] #ib,iz,iy,ix
                print(f"Done.")
            try:
                self.var['monopole_gr/lapse_h'] = f["Params"].attrs['monopole_gr/lapse_h']
                self.var['monopole_gr/hypersurface_h']=f["Params"].attrs["monopole_gr/hypersurface_h"]
                self.var['monopole_gr/shift']=f["Params"].attrs["monopole_gr/shift"]
                self.var['monopole_gr/rhoadm'] = f["Params"].attrs['monopole_gr/matter_h'][0,:]
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

def Movie1D(data,varname='p.density',moviename=None,anax=None,anay=None,ylim=None,xlim=None,yscale='linear',xscale='linear'):
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
        pl.yscale(yscale)
        pl.xscale(xscale)
        pl.ylabel(varname)
        pl.xlabel('xc')
        pl.savefig(iofile)


    if (moviename  is None):
        moviename = f'movie.{varname}.mpeg'
        
    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p {moviename}")
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

def PlotScatter3DvsRadius(ax,data,varname=None):
    if (varname is None):
        varname = 'p.density'

    print("Calculating Radius...",end="")
    radius = np.sqrt(data.xgrid**2 + data.ygrid**2 + data.zgrid**2)
    print("Done.")
    nradbins = 100
    radedges = np.linspace(0.,np.max(radius),nradbins+1)
    varpercentiles = np.zeros((3,nradbins))
    #Consider reducing the complexity by looping over meshblocks.
    for ibin in range(nradbins):
        idx = np.where((radius > radedges[ibin]) & (radius < radedges[ibin+1]))[0]
        ninbin = np.size(idx)
        print(ninbin)
    exit()
    print("Plotting Scatter Plot...",end="")
    ax.scatter(radius,data.var[varname])
    print("Done.")
        
    return

def ReadHistory(fname=None):
    if (fname is None):
        fname = glob(f"*.hst")[0] #This assumes that there is only one .hst file.
    #Get Data
    tempdata = np.loadtxt(fname)
    #Get Variable Names
    f = open(fname,"r")
    line = f.readline()
    line = f.readline()
    vars = line.split("=")
    varkeys = [re.sub(r'\[.*\]','',var).strip() for var in vars[1:]]
    print(f"The variables in {fname} are: {varkeys}")

    histdata = {}
    nkeys = len(varkeys)
    for i in range(nkeys):
        key = varkeys[i]
        histdata[key] = tempdata[:,i]
    f.close()
    
    return histdata

def PlotHistory(histdata,varname='maximum density'):
    pl.clf()
    pl.plot(histdata['time'],histdata[varname])
    pl.ylabel(varname)
    pl.xlabel('Time')
    varn = varname.replace(" ","")
    fout = f'History.{varn}.png'
    pl.savefig(fout)
    pl.close()

    return
    
def main():
    #List of outfile names
    filenames = sorted(glob(f"*.out1.*.phdf"))
    nfiles = len(filenames)
    #List of data dumps for each file
    data = [Dump1D(fnam) for fnam in filenames]
    Movie1D(data)
    
if (__name__=="__main__"):
    main()
