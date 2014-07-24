# This script sets up the modules and LD_LIBRARY_PATH needed
# for compiling and running the usi_compression program.

# load modules
module swap PrgEnv-cray PrgEnv-gnu
module load boost
module load cudatoolkit
module load cmake
module load craype-accel-nvidia35
module load cray-netcdf-hdf5parallel
module load cray-hdf5-parallel

# add ld library path
export LD_LIBRARY_PATH=/opt/intel/14.0.1.106/composer_xe_2013_sp1.1.106/mkl/lib/intel64:$LD_LIBRARY_PATH
