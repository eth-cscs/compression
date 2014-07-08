MINLINROOT=/users/wsawyer/Software/USI_COMPRESSION/minlin/include
EIGENROOT=/users/wsawyer/Software/USI_COMPRESSION/usi_compression/cxx/eigen 
MKLROOT=/opt/intel/14.0.1.106/composer_xe_2013_sp1.1.106/mkl
BOOSTROOT=/apps/daint/boost/1.54.0/gnu_473/include
THRUSTROOT=/scratch/daint/bcumming/github/thrust

MINLIN_INCLUDE=-I$(MINLINROOT)
EIGEN_INCLUDE=-I$(EIGENROOT)
INCLUDE=-I$(MKLROOT)/include -I$(BOOSTROOT) -I$(THRUSTROOT)

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
LIBS=$(MKLLIBS)

MINLIN_ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -D__host__=" " -D__device__=" " -DUSE_MINLIN -std=c++11 -DFULL_EIGENSOLVE
EIGEN_ARGS= -DUSE_EIGEN -DEIGEN_EIGENSOLVE -std=c++11 -shared -fPIC

all: minlin_host eigen

minlin_host: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_minlin_host -fopenmp -O3 $(INCLUDE) $(MINLIN_INCLUDE) $(LIBS) $(MINLIN_ARGS)

eigen: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_eigen -fopenmp -O3 $(INCLUDE) $(EIGEN_INCLUDE) $(LIBS) $(EIGEN_ARGS)
