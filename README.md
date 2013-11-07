# Fld

A script and classes to convert Parallel nek5000 fld files to hdf5.

## Requirements:

 Python 2.7, numpy, h5py

## Installation:

 Install the Requirements above, then run
    python setup.py install

 Alternatively, you can just place this file in your working directory and run
 it as below.

## Usage:

Type python Fld.py -h to see the command line usage

Here is an example:

python Fld.py -n 2 -p ray -d ./ -o out.hdf5

The dimension of the arrays (in fortran order) are (nz, ny, nx, nelt, nt).

## Caveats:

The scripts uses globing to match all relevant files in the specified folder.
Therefore, the first file in the directory (e.g. ray15.f00001) should contain
the mesh. This code does not guess which array stores the mass matrix, and
merely transcribes the data.

## API:

The classes for reading Fld are not documented, but should be pretty easy to
modify and use.

## Disclaimer:

This code almost surely contains bugs, and is provided with no warranty
whatsoever. Use at your own risk.
