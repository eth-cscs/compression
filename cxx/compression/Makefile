MINLINROOT=../minlin/include
EIGENROOT=../eigen 
THRUSTROOT=$(CUDATOOLKIT_HOME)/include/thrust

MINLIN_INCLUDE=-I$(MINLINROOT)
EIGEN_INCLUDE=-I$(EIGENROOT)
NVCC_INCLUDE=-I$(MPICH_DIR)/include -I$(NETCDF_DIR)/include $(BOOST_INCLUDE_OPTS)
INCLUDE=-I$(MKLROOT)/include -I$(BOOSTROOT) -I$(THRUSTROOT)

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
LIBS=$(MKLLIBS) -lboost_program_options -lcublas

MINLIN_ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP \
	    -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -D__host__=" " \
	    -D__device__=" " -DUSE_MINLIN -std=c++11 #-g -DMINLIN_DEBUG
MINLIN_DEVICE_ARGS=-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -DUSE_MINLIN -DUSE_GPU -g
EIGEN_ARGS= -DUSE_EIGEN -DEIGEN_EIGENSOLVE -std=c++11

all: minlin_host eigen

minlin_host: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_minlin_host -fopenmp -O3 \
		$(INCLUDE) $(MINLIN_INCLUDE) $(LIBS) $(MINLIN_ARGS)

minlin_device: usi_compression.cpp
	nvcc -c usi_compression.cu --compiler-options "-fopenmp -O3" \
		$(INCLUDE) $(MINLIN_INCLUDE) $(NVCC_INCLUDE) $(LIBS) $(MINLIN_DEVICE_ARGS)
	CC usi_compression.o -o usi_compression_minlin_device -fopenmp $(LIBS) -g

eigen: usi_compression.cpp
	CC usi_compression.cpp -o usi_compression_eigen -fopenmp -O3 $(INCLUDE) $(EIGEN_INCLUDE) $(LIBS) $(EIGEN_ARGS)

clean:
	rm -f usi_compression_eigen usi_compression_minlin_host usi_compression_minlin_device
