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
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

template <typename ScalarType>
ScalarType* read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields, const int iam_in_x, const int iam_in_y, const int pes_in_x, const int pes_in_y, int &rows, int &cols, size_t **start_out, size_t **count_out, int *ncid_out, int *varid_out )
{
  int ncid, varid;
  int ndims;

  /* error handling */
  int retval ;

  int    *dimids, *dimids_out;
  size_t *dims;
  size_t *p;

  std::string input_filename = filename;
  std::string output_filename = filename.substr(0,filename.length()-4) + "_" + fields[0] + ".nc4";

  int my_rank, mpi_processes;
  int slab_size ; // is the number of entries in one slab to be read in
  ScalarType *data;
  char dim_name[NC_MAX_NAME+1];

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );   
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );
  // Assumptions: mpi_processes is power of 2

  /* Open the file. NC_NOWRITE tells netCDF we want read-only access to the file.*/

  if ( my_rank == 0 ) std::cout << "Processing: " << input_filename.c_str() << " SEARCHING FOR " << fields[0].c_str() << std::endl;

  if ((retval = nc_open_par(input_filename.c_str(), NC_NOWRITE|NC_MPIIO, comm, info, &ncid))) ERR(retval);

  if ((retval = nc_create_par(output_filename.c_str(), NC_CLOBBER | NC_NETCDF4 | NC_MPIIO, comm, info, ncid_out))) ERR(retval);

  //  sequential: if ((retval = nc_open(input_filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

  if ((retval = nc_inq_varid(ncid, fields[0].c_str(), &varid))) ERR(retval);

  if ((retval = nc_inq_varndims(ncid, varid, &ndims))) ERR(retval);
  
  if ( my_rank == 0 ) std::cout << "ndims " << ndims << std::endl;

  dimids = (int*)    malloc (sizeof(int)*ndims);
  dimids_out = (int*)    malloc (sizeof(int)*ndims);
  *start_out  = (size_t*) malloc (sizeof(size_t)*ndims);   // Starting points for parallel implementation
  *count_out  = (size_t*) malloc (sizeof(size_t)*ndims);   // Slab sizes
  dims   = (size_t*) malloc (sizeof(size_t)*ndims);

  size_t *start = *start_out;
  size_t *count = *count_out;

  if ((retval = nc_inq_vardimid(ncid, varid, dimids))) ERR(retval);

  p = dims;
  for (int i=0; i<ndims; ++i ) {
    if ((retval = nc_inq_dim(ncid, dimids[i], dim_name, p))) ERR(retval);
    if ( my_rank == 0 ) std::cout << "dimension = " << dim_name << " length " << *p << std::endl;
// OUTPUT
    if ((retval = nc_def_dim(*ncid_out, dim_name, *p, &dimids_out[i]))) ERR(retval);   // Define dimensions
    *p++;
    start[i] = 0;
  }


// OUTPUT
  if ((retval = nc_def_var(*ncid_out, fields[0].c_str(), NC_DOUBLE, ndims, dimids_out, varid_out))) ERR(retval);   // Define variable
  if ((retval = nc_enddef(*ncid_out))) ERR(retval);   // End of definition

  for (int i=0; i < ndims-2; ++i) {   // First dimensions are not distributed 
    start[i] = 0;
    count[i] = dims[i];
  }
  // Last two dimensions are distributed, assumes that dimensions are divisible by decomposition

  
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

  slab_size = 1;
  for (int i=0; i < ndims; ++i) { slab_size *= count[i]; }

  data  = (ScalarType*) malloc (sizeof(ScalarType)*slab_size);

  /* Read the slab this process is responsible for. */
  std::cout << "Rank: " << my_rank << " reading count " << count[0] << " " << count[1] << " " << count[2] << " " << count[3] << " start " << start[0] << " " << start[1] << " " << start[2] << " " << start[3] << std::endl;
  /* Read one slab of data. */
  if ((retval = nc_get_vara_double(ncid, varid, start, count, data))) ERR(retval);

// OUTPUT
//  if ((retval = nc_put_vara_double(*ncid_out, *varid_out, start, count, data))) ERR(retval);
// OUTPUT FILE will be closed in main program
//  if ((retval = nc_close(*ncid_out))) ERR(retval);

  switch (ndims) 
  { 
    case 2:  cols = count[0]; cols = count[1]; break;
    case 3:  rows = count[0]; cols = count[1]*count[2]; break;
    case 4:  rows = count[0]*count[1]; cols = count[2]*count[3]; break;
    default: std::cout << "Number dimensions " << ndims << " not supported " << std::endl; break;
  }

  free( dims );
  free( dimids );
//  free( start );
//  free( count );

  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);

  return data;
}
