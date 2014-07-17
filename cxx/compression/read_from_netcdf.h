#include <vector>
#include "mpi.h"
#include <netcdf_par.h>
#include <netcdf.h>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

/**
	   Write description of function here.
	   The function should follow these comments.
	   Use of "brief" tag is optional. (no point to it)
	   
	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     filename    Filename (string)
	   @param[in]     fields      List of the fields to be extracted (vector of strings0
	   @return                    vector of concatenated field values
	 */

     /* Handle errors by printing an error message and exiting with a
      * non-zero status. */

/*
get total number of dimensions from netcdf
make sure this is the same as the sum of compressed & distributed dimensions

decide on number of subdivisions in each compressed direction

create vectors/arrays start, count, imap of length ndims
create vector/array stride = ones of length ndims

set inter-element-distance = 1

go through compressed dimensions
get id and length for dimension
write start[id] = 0
write count[id] = length
write imap[id] = inter-element-distance
set inter-element-distance *= length

go through distributed dimensions
get id and length for dimension
calculate start and count for dimension, depending on rank
write these to start[id] and count[id]
write imap[id] = inter-element-distance
set inter-element-distance *= count

read array from file with nc_get_varm_double(ncid, varid, start, count, stride, imap, data)
create column major matrix from data
*/


template <typename ScalarType>
GenericMatrix read_from_netcdf(const std::string filename,
                               const std::string variable,
                               const std::vector<std::string> compressed_dimensions,
                               const std::vector<std::string> distributed_dimensions)
{

  // get MPI information
  int my_rank, mpi_processes;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Info mpi_info = MPI_INFO_NULL;
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );

  // open NetCDF file for read-only access
  int retval, netcdf_id; // retval is used for error handling within the whole function
  if ((retval = nc_open_par(filename.c_str(), NC_NOWRITE|NC_MPIIO, mpi_comm, mpi_info, &netcdf_id))) ERR(retval);
  if (!my_rank) std::cout << "Processing: " << filename.c_str() << " SEARCHING FOR " << variable.c_str() << std::endl;

  // build list of number of processes along each distributed dimension
  // note: we assume mpi_processes to be a power of 2
  int p = mpi_processes;
  int i = 0;
  std::vector<int> process_distribution(distributed_dimensions.size(), 1);
  while (p > 1) {
    if (p%2 != 0) {
      std::cout << "Error: The number of processes must be a power of 2" << std::endl;
      exit(1);
    }
    p /= 2;
    process_distribution[i] *= 2;
    i = (i+1)%process_distribution.size(); // restart at the beginning if end is reached
  }

  // get number of dimensions for selected variable
  int variable_id, n_dimensions;
  if ((retval = nc_inq_varid(netcdf_id, variable.c_str(), &variable_id))) ERR(retval);
  if ((retval = nc_inq_varndims(netcdf_id, variable_id, &n_dimensions))) ERR(retval);
  if (!my_rank) std::cout << "ndims " << ndims << std::endl;
  assert(n_dimensions == compressed_dimensions.size() + distributed_dimensions.size());

  // get IDs of dimensions used for the variable
  int* dimension_ids = new int[sizeof(int)*n_dimensions];
  if ((retval = nc_inq_vardimid(netcdf_id, variable_id, dimension_ids))) ERR(retval);

  // set up arrays used as arguments for reading the data
  size_t* start = new size_t[sizeof(size_t)*n_dimensions];
  size_t* count = new size_t[sizeof(size_t)*n_dimensions];
  ptrdiff_t* imap = new ptrdiff_t[sizeof(ptrdiff_t)*n_dimensions];

  // the inter-element distance is needed for building up the 'imap' vector
  int interelement_distance = 1;
  int dimension_id, vardim_id;

  // fill up entries in start, count, and imap belonging to compressed dimensions
  if (!my_rank) std::cout << "Compressed dimensions:" << std::endl;
  for (i=0, i<compressed_dimensions.size(), i++) {

    // find out where in the variable dimensions the current dimension is located
    // TODO: check if there is a better way for this
    if ((retval = nc_inq_dimid(netcdf_id, compressed_dimensions[i].c_str(), &dimension_id))) ERR(retval);
    vardim_id = -1;
    for j=0, j<n_dimensions, j++ {
      if (dimension_ids[j] == dimension_id) {
        vardim_id = j;
        break;
      }
    }
    assert(vardim_id >= 0); // make sure the dimension has been found

    // get length of dimension
    size_t dim_length;
    if ((retval = nc_inq_dimlen(netcdf_id, dimension_id, &dim_length))) ERR(retval);
    if (!my_rank) std::cout << "  dimension '" << compressed_dimensions[i] << "': length " << dim_length << std::endl;

    // write values into arrays
    start[vardim_id] = 0;
    count[vardim_id] = dim_length;
    imap[vardim_id] = interelement_distance;

    // the next variable has to change slower in the output array
    interelement_distance *= dim_length;  
  }

  int N_rows = interelement_distance;

  // fill up entries in start, count, and imap belonging to distributed dimensions
  if (!my_rank) std::cout << "Distributed dimensions:" << std::endl;
  for (i=0, i<distributed_dimensions.size(), i++) {

    // find out where in the variable dimensions the current dimension is located
    // TODO: check if there is a better way for this
    if ((retval = nc_inq_dimid(netcdf_id, distributed_dimensions[i].c_str(), &dimension_id))) ERR(retval);
    vardim_id = -1;
    for j=0, j<n_dimensions, j++ {
      if (dimension_ids[j] == dimension_id) {
        vardim_id = j;
        break;
      }
    }
    assert(vardim_id >= 0); // make sure the dimension has been found

    // get length of dimension
    size_t dim_length;
    if ((retval = nc_inq_dimlen(netcdf_id, dimension_id, &dim_length))) ERR(retval);
    if (!my_rank) std::cout << "  dimension '" << distributed_dimensions[i] << "': length " << dim_length << std::endl;

    // write values into arrays
    start[vardim_id] = 0;           // TODO: calculate correct value based on splitting
    count[vardim_id] = dim_length;  // TODO: calculate correct value based on splitting
    imap[vardim_id] = interelement_distance;

    // the next variable has to change slower in the output array
    interelement_distance *= dim_length; // TODO: change to correct value based on splitting
  }

  int N_cols = interelement_distance / N_rows;

  // read values for output
  GenericColMatrix output_matrix(N_rows, N_columns);
  ScalarType *data = GET_POINTER(output_matrix);
  if ((retval = nc_get_varam_double(netcdf_id, variable_id, start, count, NULL, imap, data))) ERR(retval);

  // delete working arrays
  delete[] dimension_ids;
  delete[] start;
  delete[] count;
  delete[] imap;

  if ((retval = nc_close(netcdf_id))) ERR(retval);

  return output_matrix;

}






  int processes_in_x;
  int processes_in_y = 1;
  for ( processes_in_x = mpi_processes; processes_in_x > processes_in_y; processes_in_x /= 2 ) { processes_in_y *= 2; }

  const int iam_in_x = my_rank % processes_in_x;
  const int iam_in_y = my_rank / processes_in_x;
  std::cout << "Decomposition:  processes_in_x " << processes_in_x << " processes_in_y " << processes_in_y << " iam x " << iam_in_x << " iam_in_y " << iam_in_y << std::endl;
  if ( processes_in_x * processes_in_y != mpi_processes ) { std::cout << "mpi_processes " << mpi_processes << " not power of two; aborting " << std::endl; abort(); }
  


  
  if (dims[ndims-2] % pes_in_x != 0) {
      if (!my_rank) std::cout << "dims[ndims-2] " << dims[ndims-2] << " is not divisible by pes_in_x = " << pes_in_x << std::endl;
      return NULL;
  } 
  start[ndims-2] = iam_in_x * ( dims[ndims-2] / pes_in_x );
  count[ndims-2] = dims[ndims-2] / pes_in_x; 

  if (dims[ndims-1] % pes_in_y != 0) {
      if (!my_rank) std::cout << "dims[ndims-1] " << dims[ndims-1] << " is not divisible by pes_in_y = " << pes_in_y << std::endl;
      return NULL;
  } 
  start[ndims-1] = iam_in_y * ( dims[ndims-1] / pes_in_y ) ;
  count[ndims-1] = dims[ndims-1] / pes_in_y; 


  /* Read the slab this process is responsible for. */
  std::cout << "Rank: " << my_rank << " reading count " << count[0] << " " << count[1] << " " << count[2] << " " << count[3] << " start " << start[0] << " " << start[1] << " " << start[2] << " " << start[3] << std::endl;
