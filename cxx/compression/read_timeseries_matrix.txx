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
MatrixXX read_timeseries_matrix(const std::string filename, const std::vector<std::string> fields )
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

  /* Open the file. NC_NOWRITE tells netCDF we want read-only access to the file.*/

  std::cout << "Processing: " << filename.c_str() << " SEARCHING FOR " << fields[0].c_str() << std::endl;

  if ((retval = nc_open_par(filename.c_str(), NC_NOWRITE|NC_MPIIO, comm, info, &ncid))) ERR(retval);
  //  sequential: if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

  if ((retval = nc_inq_varid(ncid, fields[0].c_str(), &varid))) ERR(retval);

  if ((retval = nc_inq_varndims(ncid, varid, &ndims))) ERR(retval);
  
  std::cout << "ndims " << ndims << std::endl;

  dimids = (int*)    malloc (sizeof(int)*ndims);
  start  = (size_t*) malloc (sizeof(size_t)*ndims);   // Starting points for parallel implementation
  count  = (size_t*) malloc (sizeof(size_t)*ndims);   // Slab sizes
  dims   = (size_t*) malloc (sizeof(size_t)*ndims);

  if ((retval = nc_inq_vardimid(ncid, varid, dimids))) ERR(retval);

  p = dims;
  for (int i=0; i<ndims; ++i ) {
    if ((retval = nc_inq_dimlen(ncid, dimids[i], p))) ERR(retval);
    std::cout << "dimension = " << *p++ << std::endl;
    start[i] = 0;
  }

  if (dims[0] % mpi_processes != 0)
    {
      if (!my_rank) std::cout << "dims[0] " << dims[0] << " is not evenly divisible by mpi_size= " << mpi_processes << std::endl;
      static Matrix<ScalarType,0,0> empty_vector;
      return empty_vector;
    }
  
  count[0] = dims[0] / mpi_processes;         // Number of 2D slices, domain decomposed in first dimension for now

  slab_size = count[0];
  for (int i=1; i<ndims; ++i ) { 
    start[i] = 0; count[i] = dims[i]; slab_size *= dims[i]; 
    std::cout << i << " start " << start[i] << " count " << count[i] << std::endl;
  }

  data  = (ScalarType*) malloc (sizeof(ScalarType)*slab_size);

  /* Read the slab this process is responsible for. */
  start[0] = dims[0] / mpi_processes * my_rank;
  std::cout << "Rank: " << my_rank << " reading slab with size" << slab_size << std::endl;
  /* Read one slab of data. */
  if ((retval = nc_get_vara_double(ncid, varid, start, count, data))) ERR(retval);

  //  std::vector<ScalarType> output(data, data+slab_size);

  typedef MatrixXX MatrixX;

  
  int dim1, dim2;

  switch (ndims) 
  { 
    case 2:  dim1 = count[0]; dim2 = count[1]; break;
    case 3:  dim1 = count[0]; dim2 = count[1]*count[2]; break;
    case 4:  dim1 = count[0]*count[1]; dim2 = count[2]*count[3]; break;
  default: std::cout << "Number dimensions " << ndims << " not supported " << std::endl; break;
  }

  std::cout << "Creating eigen matrix " << dim1 << " X " << dim2 << std::endl;
  Map<MatrixX> output(data,dim1,dim2); 
  free( dims );
  free( dimids );
  free( start );
  free( count );
  //  free( data );   // Cannot free data, since it is now bound to the Matrix instantiation "output"

  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);
     
  return output;
}
