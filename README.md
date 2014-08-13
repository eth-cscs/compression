USI Compression
===============

This repository contains a Matlab and C++ implementation of the compression
algorithm developped at USI by Prof. Illia Horenko.


Usage
-----

The program has three different variants using different backends for the
linear algebra computations.
One variant uses the [Eigen library](http://eigen.tuxfamily.org/) while the
other two use [minlin](https://github.com/bcumming/minlin) with either OpenMP
or CUDA for the computations. The usage is identical for all variants.

```
usi_compression_VARIANT INPUT_FILE [OPTIONS]

Options:
--help              Display a usage message and exit.
--version           Display the version number and exit.

-v, --variables     The NetCDF variables that will be compressed. REQUIRED
-c, --compressed    The NetCDF dimensions along which the data is compressed.
                    default: lon, lat
-i, --indexed       The NetCDF dimensions where a single entry will be selected
                    and compressed (e.g. to compress only one time step).
                    format: dim=# (e.g. time=1), default: none
-K, --clusters      The number of clusters used for the compression algorithm.
                    default: 10
-M, --eigenvectors  The number of eigenvectors used for the final compression.
                    default: 5
-h, --horizontal-stacking  By default, the variables are stacked along the
                    compressed dimension if multiple variables are selected.
                    By specifying this option, this can be changed to stacking
                    the variables along the distributed (horizontal) dimensions.
-o, --output-file   The path to the file where the reconstructed data is written.
                    default: INPUT_FILE_reconstructed.nc4
-a, --append        The data is appended to the output file instead of overwriting it.
```


Setup
-----

After cloning this repository and updating the submodules, the code can be
compiled with cmake. Make sure all the requirements are available. On the CSCS
systems, these can be loaded with the modulefile in the folder 'util'.

```bash
git clone https://github.com/eth-cscs/compression.git
cd compression
git submodule init     # these two lines initialize the minlin
git submodule update   # submodule and update its repository

mkdir build            # with cmake, you should do out-of-source builds
cd build
cmake ..

# load the required modules on CSCS systems (may not work on other systems)
module load ../util/usi_compression_module

make                   # this builds all variants, alternatively select only
                       # one variant with 'make eigen', 'make minlin_host' or
                       # 'make minlin_device'

aprun usi_compression_minlin_host INPUT_DATA -v VAR1 VAR2 ... # see 'Usage'
```


Requirements
------------

* Eigen: This is only needed for the Eigen variant and is included in this
  repository. (template library, header-only)
* minlin: This is only needed for the minlin variants and is included as a
  git submodule in this repository (see 'Setup'). (template library, header-only)
* Thrust: Minlin uses thrust for providing the underlying host and device
  vectors. We use the version from cudatoolkit and do not specify include or
  library paths so they should be provided by the 'cudatoolkit' module.
* MKL: Some BLAS routines are used as a fallback in the minlin variants where
  minlin does not provide certain functionality. For this, we rely on the
  $MKLROOT environment variable, provided by the 'intel' module.
* CUBLAS: Some CUBLAS routines are used as a fallback in the minlin_device
  variant where minlin does not provide certain functionality. We do not
  specify include or library paths so they should be provided by the
  'cudatoolkit' module.
* Boost: We use the program_options component for parsing the command line
  options. We do not specify include or library paths in the CMakeList so they
  should be provided by the 'boost' module.
* MPI: We use MPI for communication between processes running in parallel. We
  do not specify include or library paths so they should be provided by the
  'cray-mpich' module.
* NetCDF: We use the parallel variant of the NetCDF library to read and write
  data files. We do not specify include or library paths so they should be
  provided by the 'cray-netcdf-hdf5parallel' and 'cray-hdf5-parallel' modules.
* Standard C++ library: We use several parts of the standard library.


License
-------
Copyright (c) 2014, Università della Svizzera italiana (USI) & Centro Svizzero di Calcolo Scientifico (CSCS)  
All rights reserved.

This software may be modified and distributed under the terms of the BSD license. See the [LICENSE file](LICENSE.md) for details.

The Eigen and minlin projects are subject to their respective licensing conditions.
