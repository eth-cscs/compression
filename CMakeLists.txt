#--------------------------------------------------
# C++ implementation of USI compression
#--------------------------------------------------
project(USI_COMPRESSION)
cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------
# configure testing, 
# optionally support the Eigen tests
#--------------------------------------------------
option(BUILD_TESTING OFF)
if(BUILD_TESTING)
  enable_testing()
  option(ENABLE_EIGEN_TESTING OFF)
  option(ENABLE_VIENNACL_TESTING OFF)
endif()

option(ENABLE_VIENNACL_EXAMPLES OFF)

#--------------------------------------------------
# Add Eigen sub project
# disable any Qt based tests
#--------------------------------------------------
set(EIGEN_TEST_NOQT 1 CACHE BOOL "" FORCE)
add_subdirectory(eigen)

#--------------------------------------------------
# Add ViennaCL sub project
#--------------------------------------------------
if(WIN32)
  link_directories("C:/Program Files/boost_1_54_0/lib")
endif()
add_subdirectory(viennaCL)

#--------------------------------------------------
# Add compression project
#--------------------------------------------------
add_subdirectory(compression)
