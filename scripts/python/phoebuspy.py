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

def ReadParameterFile(FileName='Params.dat'):
    params = {}

    with open(FileName) as fobj:
    
        for line in fobj:
            line = line.strip()
            if not line or line.startswith('#'):
                continue #skip blank lines and comments
            key_value = line.split('=')
            if len(key_value) == 2:
                #params[key_value[0].strip()] = key_value[1].strip()
                key = key_value[0].strip()
                val = key_value[1].strip()
                #Detect lists: comma-separated
                if ',' in val:
                    val = [v.strip() for v in val.split(',')]
                params[key] = val

    #convert to float
    floatparams = ['slice','varmin','varmax']
    for fpar in floatparams:
        if(fpar in params):
            val = params[fpar]
            if isinstance(val,list):
                params[fpar] = [float(v) for v in val]
            else:
                params[fpar] = float(val)

    #convert to int
    intparams = ['sliceaxis','nslices','keepaxis']
    for ipar in intparams:
        if(ipar in params):
            val = params[ipar]
            if (isinstance(val,list)):
                params[ipar] = [int(v) for v in val]
            else:
                params[ipar] = int(val)

    #convert to boolean
    boolparams = ['plotlog']
    for bpar in boolparams:
        if (bpar in params):
            val = params[bpar]
            if (isinstance(val,list)):
                params[bpar] = [(v.lower()=='true') for v in val]
            else:
                params[bpar] = (val.lower() == 'true')

    ##Split param into a list if needed
    #splitparams = ['varname']
    #for spar in splitparams:
    #    if (spar in params):
    #        temp = params[spar]
    #        temp = temp.split(',')
    #        temp = [s.strip() for s in temp]
    #        params[spar] = temp

    #Default Values
    if (('varname' in params) == False):
        params['varname'] = 'p.density'
    if (('sliceaxis' in params) == False):
        params['sliceaxis'] = 1
    if (('keepaxis' in params) == False):
        params['keepaxis'] = 3
    if (('varmin' in params) == False):
        params['varmin'] = -16.
    if (('varmax' in params) == False):
        params['varmax'] = -3.
    if (('plotlog' in params) == False):
        params['plotlog'] = False
        
    params['varname'] = params['varname'] if isinstance(params['varname'],list) else [params['varname']]
    
    return params

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

def Make2DSlice(data,sliceaxis=1,slice=0.,extractvars=['p.density']):
    #sliceaxis=1 => z
    #sliceaxis=2 => y
    #sliceaxis=3 => x
    
    if (sliceaxis==1):
        w=data.zf
    elif(sliceaxis==2):
        w=data.yf
    elif(sliceaxis==3):
        w=data.xf
    else:
        raise ValueError("sliceaxis must be 1 (z), 2 (y), or 3 (x)")

    slice_data = []
    
    for im in range(data.NumMB):
        # w[im] shape: (nz+1), (ny+1), or (nx+1) depending on sliceaxis
        w_local = w[im, :]

        # Find zones where the slice falls between w_local edges
        mask = (w_local[:-1] <= slice) & (slice < w_local[1:])
        
        if (np.any(mask)):
            assert np.sum(mask) == 1, f"Multiple indices match in meshblock {im}"

            idx = np.where(mask)[0][0]
            # Extract 2D slice by fixing idx along the chosen axis
            if sliceaxis == 1:
                slice_vars = {key: data.var[key][im, idx, :, :] for key in extractvars}
                #coords = (data.xgrid[im, idx, :, :], data.ygrid[im, idx, :, :])
                coords = (data.xf[im, :], data.yf[im, :])
            elif sliceaxis == 2:
                slice_vars = {key: data.var[key][im, :, idx, :] for key in extractvars}
                #coords = (data.xgrid[im, :, idx, :], data.zgrid[im, :, idx, :])
                coords = (data.xf[im, :], data.zf[im, :])
            elif sliceaxis == 3:
                slice_vars = {key: data.var[key][im, :, :, idx] for key in extractvars}
                #coords = (data.ygrid[im, :, :, idx], data.zgrid[im, :, :, idx])
                coords = (data.yf[im, :], data.zf[im, :])

            slice_data.append({
                'time': data.t,
                'sliceaxis': sliceaxis,
                'slice': slice,
                'meshblock': im,
                'index': idx,
                'slice_vars': slice_vars,
                'coords': coords
            })

    return slice_data

def Make1DSlice(data,keepaxis=1,slice=[0.,0.],extractvars=['p.density']):
    #keepaxis=1 => z
    #keepaxis=2 => y
    #keepaxis=3 => x

    if (keepaxis==1):
        w=[data.yf,data.xf]
    elif(keepaxis==2):
        w=[data.zf,data.xf]
    elif(keepaxis==3):
        w=[data.zf,data.yf]
    else:
        raise ValueError("keepaxis must be 1 (z), 2 (y), or 3 (x)")

    slice_data = []
    
    for im in range(data.NumMB):
        # w[im] shape: (nz+1), (ny+1), or (nx+1) depending on sliceaxis
        w0_local = w[0][im, :]
        w1_local = w[1][im, :]

        # Find zones where the slice falls between w_local edges
        mask0 = (w0_local[:-1] <= slice[0]) & (slice[0] < w0_local[1:])
        mask1 = (w1_local[:-1] <= slice[1]) & (slice[1] < w1_local[1:])
        
        if (np.any(mask0) & np.any(mask1)):
            assert np.sum(mask0) == 1, f"Multiple indices match in meshblock {im}"
            assert np.sum(mask1) == 1, f"Multiple indices match in meshblock {im}"

            idx0 = np.where(mask0)[0][0]
            idx1 = np.where(mask1)[0][0]
            # Extract 1D slice by fixing idx0 and idx1 along the chosen axes
            if keepaxis == 1:
                slice_vars = {key: data.var[key][im, :, idx0, idx1] for key in extractvars}
                coords = data.zc[im, :]
            elif keepaxis == 2:
                slice_vars = {key: data.var[key][im, idx0, :, idx1] for key in extractvars}
                coords = data.yc[im, :]
            elif keepaxis == 3:
                slice_vars = {key: data.var[key][im, idx0, idx1, :] for key in extractvars}
                coords = data.xc[im, :]

            slice_data.append({
                'time': data.t,
                'keepaxis': keepaxis,
                'slice': slice,
                'meshblock': im,
                'index': [idx0,idx1],
                'slice_vars': slice_vars,
                'coords': coords
            })

    return slice_data

def Make2DSlicesVsTime(params):
    import pickle
    #List of outfile names                                                      
    filenames = sorted(glob(f"*.out1.*.phdf"))
    nfiles = len(filenames)
    for i in range(nfiles):
        data = Dump3D(filenames[i],extractvars=params['varname'])
        iofile=f'TwoDSlice{i:04d}.pkl'
        print(f"Making {iofile}")
        slice_data=Make2DSlice(data,sliceaxis=params['sliceaxis'],extractvars=params['varname'])
        #Saving the slice with python's pickel.  We might consider using hdf5 instead.
        with open(iofile, "wb") as f:
            pickle.dump(slice_data, f)

    return

def Make2DSlicesVsCoord(params):
    import pickle
    #List of outfile names                                                      
    filename = params['phdffile']
    data = Dump3D(filename,extractvars=params['varname'])
    
    sliceaxis = params['sliceaxis']
    if (sliceaxis == 1):
        wmin = np.min(data.zf[:,:-1])
        wmax = np.max(data.zf[:,:-1])
    elif (sliceaxis == 2):
        wmin = np.min(data.yf[:,:-1])
        wmax = np.max(data.yf[:,:-1])
    elif (sliceaxis == 3):
        wmin = np.min(data.xf[:,:-1])
        wmax = np.max(data.xf[:,:-1])

    nslices = params['nslices']
    wslices = np.linspace(wmin,wmax,num=nslices)
    
    for i in range(nslices):
        iofile=f'TwoDSlice{i:04d}.pkl'
        print(f"Making {iofile}")
        slice_data=Make2DSlice(data,sliceaxis=params['sliceaxis'],slice=wslices[i],extractvars=params['varname'])
        #Saving the slice with python's pickel.  We might consider using hdf5 instead.
        with open(iofile, "wb") as f:
            pickle.dump(slice_data, f)

    return

def Make1DSlicesVsTime(params):
    import pickle
    #List of outfile names                                                      
    filenames = sorted(glob(f"*.out1.*.phdf"))
    nfiles = len(filenames)
    for i in range(nfiles):
        data = Dump3D(filenames[i],extractvars=params['varname'])
        if (params['keepaxis']==1):
            iofile=f'OneDSliceZ_T{i:04d}.pkl'
        elif (params['keepaxis']==2):
            iofile=f'OneDSliceY_T{i:04d}.pkl'
        elif (params['keepaxis']==3):
            iofile=f'OneDSliceX_T{i:04d}.pkl'
        print(f"Making {iofile}")
        slice_data=Make1DSlice(data,keepaxis=params['keepaxis'],slice=params['slice'],extractvars=params['varname'])
        #Saving the slice with python's pickel.  We might consider using hdf5 instead.
        with open(iofile, "wb") as f:
            pickle.dump(slice_data, f)

    return

def Movie2DSlices(varname='p.density',plotlog=True,varbounds=[-16.,-3.]):
    import pickle
    filenames = sorted(glob(f"*TwoDSlice*.pkl"))
    nfiles = len(filenames)
    minvar = np.inf
    maxvar = -np.inf
    moviename=f'Movie2Dslices.{varname}.mpeg'
    for i in range(nfiles):
        with open(filenames[i], "rb") as f:
            slice_data = pickle.load(f)
        iofile=f'img{i:04d}.png'
        print(f"Making {iofile}")
        pl.clf()
        ax = pl.gca()
        for block in slice_data:
            x, y = block['coords']
            var = block['slice_vars'][varname]
            if (plotlog):
                var =  np.log10(var)
            minvar = min(minvar,np.min(var))
            maxvar = max(maxvar,np.max(var))
            ax.pcolormesh(x, y, var, shading='auto',vmin=varbounds[0],vmax=varbounds[1])
        print(f'min and max = {minvar}, {maxvar}')

        dy = 0.05
        xleft = 0.05
        ybot = 0.05
        time = block['time']
        ttext = f'time = {time}'
        ax.text(xleft,ybot,ttext,transform=ax.transAxes,color='white')
        
        slice = block['slice']
        if (block['sliceaxis']==1):
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            stext = f'z = {slice}'
        elif(block['sliceaxis']==2):
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            stext = f'y = {slice}'
        elif(block['sliceaxis']==3):
            ax.set_xlabel('y')
            ax.set_ylabel('z')
            stext = f'x = {slice}'
        ax.text(xleft,ybot+dy,stext,transform=ax.transAxes,color='white')
        
        pl.colorbar(ax.collections[0], ax=ax, label=varname)
        pl.savefig(iofile)
            
    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p {moviename}")
    
    return

def Movie1DSlicesVsTime(varname=['p.density'],plotlog=True,varbounds=[-16.,-3.]):
    import pickle
    filenamesX = sorted(glob(f"*OneDSliceX_T*.pkl"))
    filenamesY = sorted(glob(f"*OneDSliceY_T*.pkl"))
    filenamesZ = sorted(glob(f"*OneDSliceZ_T*.pkl"))
    nfilesX = len(filenamesX)
    nfilesY = len(filenamesY)
    nfilesZ = len(filenamesZ)

    if not (nfilesX == nfilesY == nfilesZ):
        print(f"Error: nfilesX={nfilesX}, nfilesY={nfilesY}, nfilesZ={nfilesZ} must be equal.")
        exit(1)
        
    moviename=f'Movie1DslicesVsTime.{varname[0]}.mpeg'

    colors = ['black','blue','green']
    def PlotSliceData(ax,slice_data,color='black'):
        minvar = np.inf
        maxvar = -np.inf
        for block in slice_data:
            x = block['coords']
            var = block['slice_vars'][varname[0]]
            minvar = min(minvar,np.min(var))
            maxvar = max(maxvar,np.max(var))
            ax.scatter(x, var,color=color,marker='.')
        print(f'min and max = {minvar}, {maxvar}')
        
    for i in range(nfilesX):
        with open(filenamesX[i], "rb") as fX:
            slice_dataX = pickle.load(fX)
        with open(filenamesY[i], "rb") as fY:
            slice_dataY = pickle.load(fY)
        with open(filenamesZ[i], "rb") as fZ:
            slice_dataZ = pickle.load(fZ)
        iofile=f'img{i:04d}.png'
        print(f"Making {iofile}")
        pl.clf()
        ax = pl.gca()
        PlotSliceData(ax,slice_dataX,color=colors[0])
        PlotSliceData(ax,slice_dataY,color=colors[1])
        PlotSliceData(ax,slice_dataZ,color=colors[2])

        if (plotlog):
            ax.set_yscale('log')
        ax.set_ylim(varbounds)
        dy = 0.05
        xleft = 0.05
        ybot = 0.05
        block = slice_dataX[0]
        time = block['time']
        ttext = f'time = {time}'
        ax.text(xleft,ybot,ttext,transform=ax.transAxes,color='black')
        
        block = slice_dataX[0]
        keepaxis = block['keepaxis']
        slice = block['slice']
        ttext = f'[{slice[0]},{slice[1]},:]'
        ax.text(xleft,ybot+3.*dy,ttext,transform=ax.transAxes,color=colors[0])

        block = slice_dataY[0]
        keepaxis = block['keepaxis']
        slice = block['slice']
        ttext = f'[{slice[0]},:,{slice[1]}]'
        ax.text(xleft,ybot+2.*dy,ttext,transform=ax.transAxes,color=colors[1])

        block = slice_dataZ[0]
        keepaxis = block['keepaxis']
        slice = block['slice']
        ttext = f'[:,{slice[0]},{slice[1]}]'
        ax.text(xleft,ybot+1.*dy,ttext,transform=ax.transAxes,color=colors[2])

        ttext = '[Z,Y,X]'
        ax.text(xleft,ybot+4.*dy,ttext,transform=ax.transAxes,color='black')
                
        pl.savefig(iofile)
            
    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p {moviename}")
    
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
    parser.add_argument('--CalcScatterProfiles', action='store_true')
    #parser.add_argument('--varname', type=str, default='p.density')
    parser.add_argument('--Make2DSlicesVsTime', action='store_true')
    parser.add_argument('--Make2DSlicesVsCoord', action='store_true')
    parser.add_argument('--Make1DSlicesVsTime', action='store_true')
    parser.add_argument('--Movie2DSlices', action='store_true')
    parser.add_argument('--Movie1DSlicesVsTime', action='store_true')
    args= parser.parse_args()

    params = ReadParameterFile()

    
    if (args.CalcScatterProfiles):
        #List of outfile names                                                      
        filenames = sorted(glob(f"*.out1.*.phdf"))
        nfiles = len(filenames)
        for i in range(nfiles):
            data = Dump3D(filenames[i],extractvars=params['varname'])
            for var in params['varname']:
                iofile=f'ScatterProfile.{var}.{i:04d}.npz'
                print(f"Making {iofile}")
                radbins,varpercentiles = CalcScatter3DvsRadius(data,nradbins=400,varname=var)
                np.savez(iofile,radbins=radbins,varpercentiles=varpercentiles)

                
    if (args.Make2DSlicesVsTime):
        Make2DSlicesVsTime(params)
                
    if (args.Make1DSlicesVsTime):
        Make1DSlicesVsTime(params)
                
    if (args.Make2DSlicesVsCoord):
        Make2DSlicesVsCoord(params)
                
    if (args.Movie2DSlices):
        #This will only make a Movie2DSlice for the first varname if there is a list.
        Movie2DSlices(varname=params['varname'],plotlog=params['plotlog'],varbounds=[params['varmin'],params['varmax']])
        
    if (args.Movie1DSlicesVsTime):
        #This will only make a Movie2DSlice for the first varname if there is a list.
        Movie1DSlicesVsTime(varname=params['varname'],plotlog=params['plotlog'],varbounds=[params['varmin'],params['varmax']])
        
            
    if (args.Movie1D):
        #List of outfile names
        filenames = sorted(glob(f"*.out1.*.phdf"))
        nfiles = len(filenames)
        #List of data dumps for each file
        data = [Dump1D(fnam) for fnam in filenames]
        Movie1D(data)
    
if (__name__=="__main__"):
    main()
