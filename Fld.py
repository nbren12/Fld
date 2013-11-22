#!/usr/bin/env python
# Author: Noah D. Brenowitz
# Email : noah@cims.nyu.edu
#
# This file contains classes for reading the parallel format of
# Nek5000 files. Moreover, when used as a command line script it enables
# writing a folder of fld files into an easily readable hdf5 file.
#
# Requirements:
# Python 2.7, numpy, h5py
#
# Installation:
#
# Install the Requirements above, then run
#    python setup.py install
#
# Alternatively, you can just place this file in your working directory and run
# it as below.
#
# Usage:
#
# Type python Fld.py -h to see the command line usage
#
# Here is an example:
#
# python Fld.py -n 2 -p ray -d ./ -o out.hdf5
#
# The dimension of the arrays (in fortran order) are (nz, ny, nx, nelt, nt).
#
# Caveats:
#
# The scripts uses globing to match all relevant files in the specified folder.
# Therefore, the first file in the directory (e.g. ray15.f00001) should contain
# the mesh. This code does not guess which array stores the mass matrix, and
# merely transcribes the data.
#
# API:
#
# The classes for reading Fld are not documented, but should be pretty easy to
# modify and use.
#
# Disclaimer:
#
# This code almost surely contains bugs, and is provided with no warranty
# whatsoever. Use at your own risk.

from __future__ import print_function
import numpy as np
import os
import re
import glob
###########################################################################
#                           Fld Folder Classes                            #
###########################################################################

def get_file_list(folder,nameBase,parallel=False):
    flds = glob.glob(os.path.join(folder,nameBase)+'*')

    if parallel:
        reg =re.compile(r'\.f(\d\d\d*)$')
    else:
        reg =re.compile(r'\.fld(\d\d*)$')

    matches = [ (x,reg.search(x)) for x in flds]
    matches = filter(lambda x :   x[1] is not None, matches)

    matches.sort(key=lambda x : int(x[1].group(1)))

    file_list,_ = zip(*matches)

    return file_list

class FldFolder:
    def __init__(self,folder='fld',nameBase='ray15'):
        self.folder = folder
        self.nameBase = nameBase
        self.file_list = self.get_file_list()
        self.nt = len(self.file_list)

    def __call__(self,i):
        return self.get_data(i)

    def get_file_list(self):
        return get_file_list(self.folder,self.nameBase,parallel =False)

    def get_data(self,i):
        fld = Fld(self.file_list[i-1])
        out = dict(data=fld.get_data(),
                time=fld.time,
                cycle=fld.cycle,
                nelt =fld.nelt,
                )

        return out

    def get_grid(self):
        gridFld = GridFld(self.file_list[0])
        return  gridFld.get_grid()

    def get_times(self):
        t = np.array(list(self(i)['time'] for i in xrange(1,self.nt+1)))
        return t

    def iter(self,start=1,end=-1):
        if end == -1 : end = self.nt
        return (self(i) for i in xrange(start,end))

class ParallelFldFolder(FldFolder):
    def get_file_list(self):
        return get_file_list(self.folder,self.nameBase,parallel =True)

    def get_data(self,i):
        fld = ParallelFld( self.file_list[i-1] )
        out = dict(data=fld.get_data(),
                time=fld.time,
                cycle=fld.cycle,
                nelt =fld.nelt,
                )
        return out

###########################################################################
#                            Fld File Parsers                             #
###########################################################################

class ParallelFld:
    def __init__(self, fname='./fld/ray15.fld01'):
        self.fname = fname
        self.read_header()

    def get_data(self):
        n = self.samples_per_block
        nelt = self.nelt
        fields = self.fields
        nx,ny,nz = self.block_shape

        with open(self.fname,'rb') as f:
            f.seek(132)
            endian = f.read(4)
            blocks = np.fromfile(f,dtype=np.int32,count=nelt)-1

            #   Invert the block permutation
            iperm  = np.zeros(nelt,dtype=np.int32)
            for i,p in enumerate(blocks):
                iperm[p] = i


            data = np.fromfile(f,dtype=np.float32)
            nf = data.shape[0]/(nx * ny * nz * nelt)
            ntot = nf * (nx * ny * nz * nelt)
            rs   = np.reshape(data[:ntot],(-1,nx,ny,nz))

            #   Create the slices to slice the data blocks by
            slices = {}
            l = 0
            if "X" in fields:
                if nz > 1:
                    slices['x'] = np.arange(0,3*nelt,3)[iperm]
                    slices['y'] = np.arange(1,3*nelt,3)[iperm]
                    slices['z'] = np.arange(2,3*nelt,3)[iperm]
                    l += 3*nelt
                else:
                    slices['x'] = np.arange(0,2*nelt,2)[iperm]
                    slices['y'] = np.arange(1,2*nelt,2)[iperm]
                    l +=2*nelt

            if "U" in fields:
                if nz > 1:
                    slices['uX'] = np.arange(l,l+3*nelt,3)[iperm]
                    slices['uY'] = np.arange(l+1,l+3*nelt,3)[iperm]
                    slices['uZ'] = np.arange(l+2,l+3*nelt,3)[iperm]
                    l += 3*nelt
                else:
                    slices['uX'] = np.arange(l,l+2*nelt,2)[iperm]
                    slices['uY'] = np.arange(l+1,l+2*nelt,2)[iperm]
                    l += 2*nelt


            if "P" in fields:
                slices['P'] = np.arange(l,l+nelt)[iperm]
                l+= nelt

            if "T" in fields:
                slices['T'] = np.arange(l,l+nelt)[iperm]

            # data = {k:np.reshape(rs[l,:,:,:].transpose((0,2,1)),(-1,ny)) for k,l in slices.iteritems()}
            data = {k:rs[l,:,:,:] for k,l in slices.iteritems()}
        return data

    def read_header(self):
        if not os.path.exists(self.fname) : raise IOError

        with open(self.fname,'r') as f:
            #   If new version use 132 bytes for header
            s= f.read(132)

        s_split = filter(lambda x : len(x) != 0 , s.split(' '))
        self.wordsize = s_split[1]
        H = dict((
            ('nelt',int(s_split[5])),
            ('block_shape',[ int(ss) for ss in s_split[2:5] ]),
            ('header_string',s),
            ('fields',list(s_split[-1]))
            ))

        time = float(s_split[7])
        cycle = int(s_split[8])


        self.__dict__.update(H)

        self._header = H
        self.time = time
        self.cycle = cycle
        self.samples_per_block = np.prod(self.block_shape)

class Fld:
    def __init__(self, fname='./fld/ray15.fld01'):
        self.fname = fname
        self.read_header()

    def read_header(self):
        if not os.path.exists(self.fname) : raise IOError

        with open(self.fname,'r') as f:
            #   If new version use 132 bytes for header
            s= f.read(80)

        s_split = filter(lambda x : len(x) != 0 , s.split(' '))

        fields = []
        for tok in s_split[6:]:
            try:
                int(tok)
                break
            except:
                fields.append(tok)


        H = dict((
            ('nelt',int(s_split[0])),
            ('block_shape',[ int(ss) for ss in s_split[1:4] ]),
            ('header_string',s),
            ('fields', fields)
            ))


        H['nxyz'] = np.prod(H['block_shape'])

        time = float(s_split[4])
        cycle = int(s_split[5])

        self.__dict__.update(H)

        self.time = time
        self.cycle = cycle
        self._header = H

    def get_data(self):
        if not hasattr(self,'_header'):
            self.read_header()

        nelt = self.nelt
        fields = self.fields
        nx,ny,nz = self.block_shape

        dat = {}
        with  open(self.fname,'r') as f:
            f.seek(84)

            data = np.fromfile(f,dtype=np.float32)

            nf = data.shape[0]/(nx * ny * nz * nelt)
            ntot = nf * (nx * ny * nz * nelt)
            rs   = np.reshape(data[:ntot],(nx,ny,nz, nf, nelt), order='F')
            del data


        i = 0
        for k in fields:
            if k == 'X':
                dat['x'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

            elif k == 'Y':
                dat['y'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

            elif k == 'Z':
                dat['z'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

            elif k == 'U':
                dat['uX'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

                dat['uY'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

                if nz > 1:
                    dat['uZ'] = rs[:,:,:,i,:].transpose((3,0,1,2))
                    i+=1

            elif k!='P':
                dat[k] = rs[:,:,:,i,:].transpose((3,0,1,2))
                i+=1

        return dat

###########################################################################
#                             Other Functions                             #
###########################################################################

def save_mean():
    import argparse

    parser = argparse.ArgumentParser(description='Plot some modes')
    parser.add_argument('iStart',type=int,nargs=1)
    parser.add_argument('iEnd',type=int,nargs=1)
    parser.add_argument('-d',dest='fld_dir',default='./fld')

    args = parser.parse_args()

    fd = FldFolder(folder=args.fld_dir)
    grid = fd.get_grid()
    dat = fd.get_data(1000)

    nelt = dat['nelt']

    iStart = args.iStart[0]
    iEnd   = args.iEnd[0] +1


    def add(x,y):
        for key in x:
            y[key] += x[key]
        return y


    mu    = dict(zip(('uX','uY','T'), [np.zeros(fd.get_shape())]*3 ))

    datgen = (fd.get_data(i)['data'] for i in xrange(iStart,iEnd))
    mu = reduce(add,datgen)

    for key in mu:
        mu[key] /= (iEnd-iStart)

    out = {}
    out['muU'] = np.reshape(mu['uX'],(-1,nelt),order='F')
    out['muW'] = np.reshape(mu['uY'],(-1,nelt),order='F')
    out['muT'] = np.reshape(mu['T'],(-1,nelt),order='F')

    from scipy.io import savemat

    savemat('./ray15_steady_%d_1_%d_py.mat'%(iStart,iEnd-1),out)
    return 0

def grid2vtk():
    fd = FldFolder()
    vtkfolder= 'vtk'

    from evtk.hl import gridToVTK
    grid = fd.get_grid()
    x = grid['x']
    y = grid['y']

    nx,ny,nz,ne = x.shape


    x = grid_to_3d(x)[:,0,0]
    y = grid_to_3d(y)[0,:,0]
    z = np.array([0.0])

    dat = fd('2000')['data']
    dat ={key:grid_to_3d( dat[ key ]  ) for key in  dat}

    gridToVTK('./vtk/test',x,y,z,pointData = dat)

def fld_to_hdf5():
    import argparse

    parser = argparse.ArgumentParser(description='Fld to hdf')
    parser.add_argument('-n',dest='nt',type=int)
    parser.add_argument('-p',dest='prefix', type=str)
    parser.add_argument('-d',dest='fld',default='./')
    parser.add_argument('-o',dest='out',default='./')
    parser.add_argument('--serial', dest = "serial", action='store_true', default =False )

    args = parser.parse_args()

    fld = args.fld
    prefix = args.prefix
    nt   = args.nt
    out   = args.out
    serial = args.serial

    import h5py

    if serial:
        fd = FldFolder(fld, nameBase=prefix)
    else:
        fd = ParallelFldFolder(fld, nameBase=prefix)

    with h5py.File(out) as f:
        dat = fd(1)

        # Write Grid
        # f.create_dataset('w', data=dat['data']['uX'])
        f.create_dataset('x', data=dat['data']['x'])
        f.create_dataset('y', data=dat['data']['y'])
        if 'z' in dat['data']:
            f.create_dataset('z', data=dat['data']['z'])

        f.flush()

        names  = fd(2)['data'].keys()

        # Find Shape
        nf = len(fd.file_list)
        nt = min(nf, nt)


        outsh = [nt]
        outsh.extend(dat['data']['x'].shape)

        for k in names:
            f.create_dataset(k, shape=outsh)

    times = np.zeros(nt)

    startind = 1
    num_sample = nt
    chunk_size = min( 10000, num_sample )
    for i_start in range(startind, startind+num_sample, chunk_size):
        if serial:
            fd = FldFolder(fld, nameBase=prefix)
        else:
            fd = ParallelFldFolder(fld, nameBase=prefix)
        i_end =  min(i_start + chunk_size,num_sample + i_start )
        ns_cur = i_end-i_start
        h5_start = i_start - startind
        h5_end   = h5_start + i_end

        tmpshp = outsh
        tmpshp[0] = ns_cur

        tmp = {k:[0]*(np.empty(tmpshp)) for k in names}

        for i in range(i_start, i_end):
            fdi = fd(i)
            data = fdi['data']
            t = fdi['time']
            for k,l in data.iteritems():
                if k in names:
                    tmp[k][i-i_start, ...] = l
                    times[i-startind] = t


        del fdi, fd, data, l
        print ( "Flushing  Iteration %d"%(i) )
        for k,l in tmp.iteritems():
            with h5py.File(out) as f:
                if len(l) > 0:
                    ns = len(l)
                    f[k][h5_start:h5_end,...] = l
                    f.flush()
        print("done")

    with h5py.File(out) as f:
        f.create_dataset('t', data=times)

###########################################################################
#                                  Main                                   #
###########################################################################

if __name__=='__main__':
    import sys
    sys.exit(fld_to_hdf5())
