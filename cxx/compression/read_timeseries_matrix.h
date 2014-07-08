#include <vector>
//
//
#include "mpi.h"
#include <netcdf_par.h>
#include <netcdf.h>
//
template <typename ScalarType>
ScalarType* read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields, const int iam_in_x, const int iam_in_y, const int pes_in_x, const int pes_in_y, int &rows, int &cols, size_t **start, size_t **count, int *ncid_out, int *varid_out );

#include "read_timeseries_matrix.txx"
