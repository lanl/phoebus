from os import system
import numpy as np
import matplotlib.pyplot as pl
import h5py
from glob import glob

class Dump1D:
    def __init__(self, filename):
        with h5py.File(filename, "r") as f:
            #print(f.keys())
            #print(f["Params"].attrs.keys())
            #print(np.shape(f["Params"].attrs["monopole_gr/hypersurface_h"]))
            #exit()
            self.varkeys = list(f.keys())[8:]
            #print(f["Locations"].keys())
            self.t = f["Info"].attrs["Time"]
            self.xf = f["Locations/x"][0]
            self.xc = 0.5 * (self.xf[1:] + self.xf[:-1])
            self.yf = f["Locations/y"][0]
            self.yc = 0.5 * (self.yf[1:] + self.yf[:-1])
            self.zf = f["Locations/z"][0]
            self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
            self.nx = len(self.xc)
            self.ny = len(self.yc)
            self.nz = len(self.zc)
            #self.rhoc = f["c.density"]
            #print(self.rhoc.shape)
            self.var = {}
            for key in self.varkeys:
                self.var[key] = f[key][0,0,0,:]

            try:
                self.var['monopole_gr/lapse_h'] = f["Params"].attrs['monopole_gr/lapse_h']
                self.var['monopole_gr/hypersurface_h']=f["Params"].attrs["monopole_gr/hypersurface_h"]

                nxgr = np.size(self.var['monopole_gr/lapse_h'])
                rout = f["Params"].attrs['monopole_gr/rout']
                rin = f["Params"].attrs['monopole_gr/rin']
                self.var['monopole_gr/radius'] = np.linspace(rin,rout,num=nxgr)
            except:
                pass
        return

def Movie1D(data,varname='p.density',anax=None,anay=None,ylim=None,xlim=None):
    #Arguments:
    #data = list of Data1D instances; there is one for each file.
    #anax = if there is an analytic solution to plot, then 
    ndata = len(data)
    for i in range(ndata):
        iofile=f'img{i:04d}.png'
        print(iofile)
        pl.clf()
        pl.plot(data[i].xc,data[i].var[varname],color='red')
        if (anax is not None):
            pl.plot(anax,anay,linestyle='dashed',color='black')
        if (ylim is not None):
            pl.ylim(ylim)
        if (xlim is not None):
            pl.xlim(xlim)
        pl.ylabel(varname)
        pl.xlabel('xc')
        pl.savefig(iofile)


    system(f"ffmpeg -r 10 -f image2 -i img%04d.png -vcodec mpeg2video -crf 25 -pix_fmt yuv420p movie.{varname}.mpeg")
    system("rm img????.png")
    
    return

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
