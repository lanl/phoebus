from os import system
import numpy as np
import numexpr as ne
import matplotlib.pyplot as pl
import h5py
import re
from glob import glob

#On Polaris, you need to create a conda environment via...
#conda create -n phoebuspy numpy numexpr matplotlib h5py scipy -y

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

def CalcScatter3DvsRadius(data,nradbins=100,varname=None):
    if (varname is None):
        varname = 'p.density'

    import time
    #start = time.time()
    #radius = np.sqrt(data.xgrid**2 + data.ygrid**2 + data.zgrid**2)
    #print("Numpy took", time.time() - start)

    #start = time.time()
    #x=data.xgrid
    #y=data.ygrid
    #z=data.zgrid
    #radius_ne = ne.evaluate("sqrt(x**2 + y**2 + z**2)")
    #print("NumExpr took", time.time() - start)
    #exit()
    #print("Calculating Radius...",end="")
    x=data.xgrid
    y=data.ygrid
    z=data.zgrid
    radius = ne.evaluate("sqrt(x**2 + y**2 + z**2)")
    
    #from scipy.stats import binned_statistic
    ## Flatten everything
    #flat_radius = radius.ravel()
    #flat_var = data.var[varname].ravel()
    #radius = np.sqrt(data.xgrid**2 + data.ygrid**2 + data.zgrid**2)
    #print("Done.")
    
    radmax_overall = np.max(radius)
    radedges = np.linspace(0.,radmax_overall,nradbins+1)
    radcenters = 0.5 * (radedges[:-1] + radedges[1:])

    ## Compute percentiles
    #p20, _, _ = binned_statistic(flat_radius, flat_var, statistic=lambda v: np.percentile(v, 20), bins=radedges)
    #p50, _, _ = binned_statistic(flat_radius, flat_var, statistic=lambda v: np.percentile(v, 50), bins=radedges)
    #p80, _, _ = binned_statistic(flat_radius, flat_var, statistic=lambda v: np.percentile(v, 80), bins=radedges)

    ## Stack for plotting
    #varpercentiles = np.vstack([p20, p50, p80])
    
    varpercentiles = np.zeros((3,nradbins))
    varhopper = [[] for _ in range(nradbins)]
    for im in range(data.NumMB):
        irads = (radius[im,:,:,:]/radmax_overall*nradbins).astype(int)
        irads[irads == nradbins] = nradbins - 1
        data_mb = data.var[varname][im,:,:,:]

        # Flatten both arrays
        irads_flat = irads.ravel()
        data_flat = data_mb.ravel()
    
        # Sort bin assignments and values together
        sort_idx = np.argsort(irads_flat)
        irads_sorted = irads_flat[sort_idx]
        data_sorted = data_flat[sort_idx]

        # Now group data into bins in a single pass
        start = 0
        for ir in np.unique(irads_sorted):
            end = np.searchsorted(irads_sorted, ir + 1, side='left')
            varhopper[ir].append(data_sorted[start:end])
            start = end 
        
        ##irmin = np.min(irads)
        ##irmax = np.max(irads)
        ##irmax = min(irmax,nradbins-1)
        #ir_used = np.unique(irads)

        #print(f'Adding vars to varhopper for meshblock {im}...',end="")
        ##for ir in range(irmin,irmax+1):
        #for ir in ir_used:
        #    mask = (irads == ir)
        #    vars_in_bin = data_mb[mask]
        #    #idx = np.where(ir == irads)
        #    #vars_in_bin = data_mb[idx]
        #    varhopper[ir].extend(vars_in_bin.tolist())
        #print("Done.")

    for i, bin_chunks in enumerate(varhopper):
        if bin_chunks:
            values = np.concatenate(bin_chunks)
            varpercentiles[:, i] = np.percentile(values, [20, 50, 80])
        else:
            varpercentiles[:, i] = np.nan
            
    #for i,vars_in_bin in enumerate(varhopper):
    #    ncells = len(vars_in_bin)
    #    print(f"{ncells} cells in radial bin {i}.")
    #    percentiles = np.percentile(np.array(vars_in_bin), [20, 50, 80])
    #    varpercentiles[:, i] = percentiles

    #radbins = 0.5*(radedges[0:-1]+radedges[1:])
        
    return radcenters,varpercentiles

def Make2DSlice(data,sliceaxis=1,slice=0.):
    #sliceaxis=1 => z
    #sliceaxis=2 => y
    #sliceaxis=3 => x

    if (sliceaxis==1):
        w=data.zgrid
    elif(slice==2):
        w=data.ygrid
    elif(slice==3):
        w=data.zgrid
        
    for im in range(data.NumMB):
        if ()
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

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--Movie1D', action='store_true')
    parser.add_argument('--CalcOneDProfiles', action='store_true')
    parser.add_argument('--varname', type=str, default='p.density')
    parser.add_argument('--MakeSlices', action='store_true')
    args= parser.parse_args()

    #Split varname into a list if needed
    varnames = args.varname.split(',')
    varnames = [varname.strip() for varname in varnames]
    args.varname = varnames

    if (args.CalcOneDProfiles):
        #List of outfile names                                                      
        filenames = sorted(glob(f"*.out1.*.phdf"))
        nfiles = len(filenames)
        for i in range(nfiles):
            data = Dump3D(filenames[i],extractvars=args.varname)
            for var in args.varname:
                iofile=f'OneDProfile.{var}.{i:04d}.npz'
                print(f"Making {iofile}")
                radbins,varpercentiles = CalcScatter3DvsRadius(data,nradbins=400,varname=var)
                np.savez(iofile,radbins=radbins,varpercentiles=varpercentiles)
                
    if (args.MakeSlices):
        #List of outfile names                                                      
        filenames = sorted(glob(f"*.out1.*.phdf"))
        nfiles = len(filenames)
        for i in range(nfiles):
            data = Dump3D(filenames[i],extractvars=args.varname)
            for var in args.varname:
                iofile=f'TwoDSlice.{var}.{i:04d}.npz'
                print(f"Making {iofile}")
                dataslice=Make2DSlice(data)
                np.save(iofile,dataslice)
                
    if (args.Movie1D):
        #List of outfile names
        filenames = sorted(glob(f"*.out1.*.phdf"))
        nfiles = len(filenames)
        #List of data dumps for each file
        data = [Dump1D(fnam) for fnam in filenames]
        Movie1D(data)
    
if (__name__=="__main__"):
    main()
