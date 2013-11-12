project(COMPRESSION)
cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------
# Find MPI
#--------------------------------------------------
find_package(MPI)
include_directories(${MPI_INCLUDE_DIRS})

#--------------------------------------------------
# Find HDF5
#--------------------------------------------------
if(WIN32)
  find_package(HDF5 NO_MODULE)
else(WIN32)
  find_package(HDF5)
endif()
include_directories(${HDF5_INCLUDE_DIR})

#--------------------------------------------------
# Find NetCDF
#--------------------------------------------------
if(WIN32)
  find_package(netcdf NO_MODULE)
else(WIN32)
  find_package(netcdf)
endif()
include_directories(${netcdf_INCLUDE_DIRS})

#--------------------------------------------------
# Find Boost
#--------------------------------------------------
find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

#--------------------------------------------------
# 
#--------------------------------------------------
find_package(ViennaCL NO_MODULE PATHS ${CMAKE_CURRENT_BINARY_DIR}/../ViennaCL)
#message("ViennaCL_FOUND is ${ViennaCL_FOUND}")
#add_definitions(-DVIENNACL_WITH_OPENCL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../viennaCL)

add_definitions(-DUSE_PARALLEL)

set(COMPRESSION_SRC_FILES
  netcdf_get_data_eigen.cpp 
)

add_executable(getdata 
  ${COMPRESSION_SRC_FILES}
)

target_link_libraries(getdata ${VIENNACL_LIBRARIES} hdf5 netcdf ${MPI_CXX_LIBRARIES})