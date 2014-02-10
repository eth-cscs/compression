#include <vector>
//
//
#include "mpi.h"
#include <netcdf_par.h>
#include <netcdf.h>
//
template <typename ScalarType>
ScalarType* read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields );

#include "read_timeseries_matrix.txx"
