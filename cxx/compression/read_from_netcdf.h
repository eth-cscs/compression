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

template <typename Scalar>
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
  //if ((retval = nc_open(filename.c_str(), NC_NOWRITE|NC_MPIIO, &netcdf_id))) ERR(retval); // sequential version
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
  if (!my_rank) std::cout << "Number of dimensions: " << n_dimensions << std::endl;
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

  // COMPRESSED DIMENSIONS
  // fill up entries in start, count, and imap
  if (!my_rank) std::cout << "Compressed dimensions:" << std::endl;
  for (int i=0; i<compressed_dimensions.size(); i++) {

    // find out where in the variable dimensions the current dimension is located
    // TODO: check if there is a better way for this
    if ((retval = nc_inq_dimid(netcdf_id, compressed_dimensions[i].c_str(), &dimension_id))) ERR(retval);
    vardim_id = -1;
    for (int j=0; j<n_dimensions; j++) {
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

  // DISTRIBUTED DIMENSIONS
  // fill up entries in start, count, and imap
  int r = my_rank; // needed for calculating index along dimension
  int d = mpi_processes; // needed for calculating index along dimension
  if (!my_rank) std::cout << "Distributed dimensions:" << std::endl;
  for (int i=0; i<distributed_dimensions.size(); i++) {

    // find out where in the variable dimensions the current dimension is located
    // TODO: check if there is a better way for this
    if ((retval = nc_inq_dimid(netcdf_id, distributed_dimensions[i].c_str(), &dimension_id))) ERR(retval);
    vardim_id = -1;
    for (int j=0; j<n_dimensions; j++) {
      if (dimension_ids[j] == dimension_id) {
        vardim_id = j;
        break;
      }
    }
    assert(vardim_id >= 0); // make sure the dimension has been found

    // get length of dimension
    size_t dim_length;
    if ((retval = nc_inq_dimlen(netcdf_id, dimension_id, &dim_length))) ERR(retval);
    if (!my_rank) std::cout << "  dimension '" << distributed_dimensions[i] << "': length " 
        << dim_length << " divided into " << process_distribution[i] << " parts" << std::endl;

    // calculate index along dimension
    // note: this is not immediately obvious, change with care!
    d /= process_distribution[i];
    int dim_index = r / d;
    r %= d;

    int size_along_dim = dim_length / process_distribution[i];

    // write values into arrays
    start[vardim_id] = dim_index * size_along_dim;
    count[vardim_id] = size_along_dim;
    if (dim_index == process_distribution[i] - 1) {
      // if we are the last process along a dimension, add the remainder
      count[vardim_id] += dim_length % process_distribution[i];
    }
    imap[vardim_id] = interelement_distance;

    // the next variable has to change slower in the output array
    interelement_distance *= count[vardim_id];
  }

  int N_cols = interelement_distance / N_rows;

  // print which values are read by the current rank
  //std::cout << "Rank " << my_rank << " has data";
  //for (int i=0; i<n_dimensions; i++) {
  //  std::cout << " " << start[i] << "-" << start[i]+count[i]-1 << " (imap: " << imap[i] << ")  ";
  //}
  //std::cout << std::endl;

  // read values for output
  GenericMatrix output_matrix(N_rows, N_cols);
#if defined(USE_GPU)
  // if we want to use the GPU, we first need to read the matrix to the
  // host and copy it over to the device
  HostMatrix<Scalar> temp_matrix(N_rows, N_cols);
  if ((retval = nc_get_varm_double(netcdf_id, variable_id, start, count, NULL,
          imap, temp_matrix.pointer()))) ERR(retval);
  output_matrix = temp_matrix;
#else
  if ((retval = nc_get_varm_double(netcdf_id, variable_id, start, count, NULL,
          imap, GET_POINTER(output_matrix)))) ERR(retval);
#endif

  // delete working arrays
  delete[] dimension_ids;
  delete[] start;
  delete[] count;
  delete[] imap;

  if ((retval = nc_close(netcdf_id))) ERR(retval);
  if (!my_rank) std::cout << "Data successfully read from NetCDF file" << std::endl;

  return output_matrix;

}
