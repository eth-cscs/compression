#--------------------------------------------------
# C++ implementation of USI compression
#--------------------------------------------------
project(USI_COMPRESSION)
cmake_minimum_required(VERSION 2.8)

# set general g++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

# define variables for boost & mkl include directories
string(REPLACE "-I" "" BOOST_INCLUDEDIR $ENV{BOOST_INCLUDE_OPTS})
find_package(Boost REQUIRED COMPONENTS program_options)
set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
link_directories($ENV{MKLROOT}/lib/intel64)

# minlin include location and defines
set(MINLIN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/minlin/include)
set(MINLIN_HOST_DEFS
	THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
	THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	IN_MEMORY_MAPPING
	__host__=\ 
	__device__=\ 
	USE_MINLIN)
set(MINLIN_DEVICE_DEFS
	-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	-DUSE_GPU
	-DIN_MEMORY_MAPPING
	-DUSE_MINLIN)

# eigen include location and defines
set(EIGEN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
set(EIGEN_DEFS USE_EIGEN IN_MEMORY_MAPPING)

# MINLIN HOST EXECUTABLE
add_executable(minlin_host src/usi_compression.cpp)
target_link_libraries(minlin_host mkl_core mkl_gnu_thread mkl_rt boost_program_options)
set_property(
	TARGET minlin_host
	APPEND
	PROPERTY INCLUDE_DIRECTORIES
	${MINLIN_INCLUDE})
set_target_properties(minlin_host PROPERTIES
	OUTPUT_NAME usi_compression_minlin_host
	COMPILE_DEFINITIONS "${MINLIN_HOST_DEFS}"
	)

# MINLIN DEVICE EXECUTABLE
include(FindCUDA)
set(CUDA_PROPAGATE_HOST_FLAGS off)
set(NVCC_INCLUDE "-I$ENV{MPICH_DIR}/include -I$ENV{NETCDF_DIR}/include -isystem${Boost_INCLUDE_DIR}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"-fopenmp -O3 -Wno-vla ${NVCC_INCLUDE}\"")
cuda_include_directories(${MINLIN_INCLUDE})
cuda_add_executable(minlin_device src/usi_compression.cu)
target_link_libraries(minlin_device mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
set_target_properties(minlin_device PROPERTIES
	OUTPUT_NAME usi_compression_minlin_device
	)

# EIGEN EXECUTABLE
add_executable(eigen src/usi_compression.cpp)
target_link_libraries(eigen mkl_core mkl_gnu_thread mkl_rt boost_program_options)
set_property(
	TARGET eigen
	APPEND
	PROPERTY INCLUDE_DIRECTORIES
	${EIGEN_INCLUDE})
set_target_properties(eigen PROPERTIES
	OUTPUT_NAME usi_compression_eigen
	COMPILE_DEFINITIONS "${EIGEN_DEFS}"
	)
