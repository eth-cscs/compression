#--------------------------------------------------
# C++ implementation of USI compression
#--------------------------------------------------
project(USI_COMPRESSION)
cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------
# configure testing, eigen supports tests
#--------------------------------------------------
option(BUILD_TESTING OFF)
if(BUILD_TESTING)
  enable_testing()
endif()

#--------------------------------------------------
# Add Eigen sub project
# disable any Qt based tests
#--------------------------------------------------
set(EIGEN_TEST_NOQT 1 CACHE BOOL "" FORCE)
add_subdirectory(eigen)

