MINLINROOT=../minlin/include
EIGENROOT=../eigen 
THRUSTROOT=$(CUDATOOLKIT_HOME)/include/thrust

MINLIN_INCLUDE=-I$(MINLINROOT)
EIGEN_INCLUDE=-I$(EIGENROOT)
NVCC_INCLUDE=-I$(MPICH_DIR)/include -I$(NETCDF_DIR)/include \
	     $(BOOST_INCLUDE_OPTS:-I%=-isystem%)
INCLUDE=-I$(MKLROOT)/include -I$(THRUSTROOT)

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
LIBS=$(MKLLIBS) -lboost_program_options

MINLIN_ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP \
	    -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -D__host__=" " \
	    -D__device__=" " -DUSE_MINLIN -std=c++11 #-g -DMINLIN_DEBUG
MINLIN_DEVICE_ARGS=-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -DUSE_MINLIN \
		   -DUSE_GPU -g
EIGEN_ARGS= -DUSE_EIGEN -std=c++11

all: minlin_host minlin_device eigen

minlin_host: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_minlin_host -fopenmp -O3 $(INCLUDE) $(MINLIN_INCLUDE) $(LIBS) $(MINLIN_ARGS)

minlin_device: usi_compression.cpp
	nvcc -c usi_compression.cu -arch=sm_35 --compiler-options "-fopenmp -O3 -Wno-vla $(NVCC_INCLUDE)" $(INCLUDE) $(MINLIN_INCLUDE) $(LIBS) $(MINLIN_DEVICE_ARGS)
	CC usi_compression.o -o usi_compression_minlin_device -fopenmp $(LIBS) -lcublas -g
	rm -f usi_compression.o

eigen: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_eigen -fopenmp -O3 $(INCLUDE) $(EIGEN_INCLUDE) $(LIBS) $(EIGEN_ARGS)

clean:
	rm -f usi_compression_eigen usi_compression_minlin_host usi_compression_minlin_device *.o
