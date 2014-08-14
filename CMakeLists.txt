#--------------------------------------------------
# C++ implementation of USI compression
#--------------------------------------------------
project(USI_COMPRESSION)
cmake_minimum_required(VERSION 2.8)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()

# set general g++ flags
# note: we always need -O3 because minlin_host doesn't compile without it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

# define variables for mkl include directories
set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
link_directories($ENV{MKLROOT}/lib/intel64)

# add debug definitions to compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# eigen include location and defines
set(EIGEN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
set(EIGEN_DEFS USE_EIGEN)

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

# minlin include location and defines
set(MINLIN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/minlin/include)
set(MINLIN_HOST_DEFS
	THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
	THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	__host__=\ 
	__device__=\ 
	USE_MINLIN)
set(MINLIN_DEVICE_DEFS # we use -D here because this isn't added automatically
	-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	-DUSE_GPU
	-DUSE_MINLIN)

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
set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options
cuda_include_directories(${MINLIN_INCLUDE})
cuda_add_executable(minlin_device src/usi_compression.cu
	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
target_link_libraries(minlin_device mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
set_target_properties(minlin_device PROPERTIES
	OUTPUT_NAME usi_compression_minlin_device
	)
