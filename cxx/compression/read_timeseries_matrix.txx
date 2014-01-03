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
MatrixXX read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields, const int iam_in_x, const int iam_in_y, const int pes_in_x, const int pes_in_y )
{
  int ncid, varid;
  int ndims;

  /* error handling */
  int retval ;

  int    *dimids;
  size_t *dims;
  size_t *p;
  size_t *start, *count;
  int my_rank, mpi_processes;
  int slab_size ; // is the number of entries in one slab to be read in
  ScalarType *data;

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );   
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );
  // Assumptions: mpi_processes is power of 2

  /* Open the file. NC_NOWRITE tells netCDF we want read-only access to the file.*/

  if ( my_rank == 0 ) std::cout << "Processing: " << filename.c_str() << " SEARCHING FOR " << fields[0].c_str() << std::endl;

  if ((retval = nc_open_par(filename.c_str(), NC_NOWRITE|NC_MPIIO, comm, info, &ncid))) ERR(retval);
  //  sequential: if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

  if ((retval = nc_inq_varid(ncid, fields[0].c_str(), &varid))) ERR(retval);

  if ((retval = nc_inq_varndims(ncid, varid, &ndims))) ERR(retval);
  
  if ( my_rank == 0 ) std::cout << "ndims " << ndims << std::endl;

  dimids = (int*)    malloc (sizeof(int)*ndims);
  start  = (size_t*) malloc (sizeof(size_t)*ndims);   // Starting points for parallel implementation
  count  = (size_t*) malloc (sizeof(size_t)*ndims);   // Slab sizes
  dims   = (size_t*) malloc (sizeof(size_t)*ndims);

  if ((retval = nc_inq_vardimid(ncid, varid, dimids))) ERR(retval);

  p = dims;
  for (int i=0; i<ndims; ++i ) {
    if ((retval = nc_inq_dimlen(ncid, dimids[i], p))) ERR(retval);
    if ( my_rank == 0 ) std::cout << "dimension = " << *p << std::endl;
    *p++;
    start[i] = 0;
  }

  for (int i=0; i < ndims-2; ++i) {   // First dimensions are not distributed 
    start[i] = 0;
    count[i] = dims[i];
  }
  // Last two dimensions are distributed, assumes that dimensions are divisible by decomposition\

  
  if (dims[ndims-2] % pes_in_x != 0) {
      if (!my_rank) std::cout << "dims[ndims-2] " << dims[ndims-2] << " is not divisible by pes_in_x = " << pes_in_x << std::endl;
      static Matrix<ScalarType,0,0> empty_vector;
      return empty_vector;
  } 
  start[ndims-2] = iam_in_x * ( dims[ndims-2] / pes_in_x );
  count[ndims-2] = dims[ndims-2] / pes_in_x; 

  if (dims[ndims-1] % pes_in_y != 0) {
      if (!my_rank) std::cout << "dims[ndims-1] " << dims[ndims-1] << " is not divisible by pes_in_y = " << pes_in_y << std::endl;
      static Matrix<ScalarType,0,0> empty_vector;
      return empty_vector;
  } 
  start[ndims-1] = iam_in_y * ( dims[ndims-1] / pes_in_y ) ;
  count[ndims-1] = dims[ndims-1] / pes_in_y; 

  slab_size = 1;
  for (int i=0; i < ndims; ++i) { slab_size *= count[i]; }

  data  = (ScalarType*) malloc (sizeof(ScalarType)*slab_size);

  /* Read the slab this process is responsible for. */
  std::cout << "Rank: " << my_rank << " reading count " << count[0] << count[1] << count[2] << count[3] << " start " << start[0] << start[1] << start[2] << start[3] << std::endl;
  /* Read one slab of data. */
  if ((retval = nc_get_vara_double(ncid, varid, start, count, data))) ERR(retval);

  int dim1, dim2;

  switch (ndims) 
  { 
    case 2:  dim1 = count[0]; dim2 = count[1]; break;
    case 3:  dim1 = count[0]; dim2 = count[1]*count[2]; break;
    case 4:  dim1 = count[0]*count[1]; dim2 = count[2]*count[3]; break;
    default: std::cout << "Number dimensions " << ndims << " not supported " << std::endl; break;
  }

  // std::cout << "Creating eigen matrix " << dim1 << " X " << dim2 << std::endl;
  Map<MatrixXX> output(data,dim2,dim1); 
  free( dims );
  free( dimids );
  free( start );
  free( count );
  //  free( data );   // Cannot free data, since it is now bound to the Matrix instantiation "output"

  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);

  return output.transpose();   // We want time x levels to be in the first dimension; horizontal dimensions in second
}
