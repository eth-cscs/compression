#include <Eigen/Dense>
#include <vector>
//
//
#include "mpi.h"
#include <netcdf_par.h>
#include <netcdf.h>
//
using namespace Eigen;
//
template <typename ScalarType>
Matrix<ScalarType,Dynamic,Dynamic> read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields );

#include "read_timeseries_matrix.txx"
