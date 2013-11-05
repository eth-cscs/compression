/*
#include <iostream>
#include <stdlib.h>
#include <stdio.h>


#include <boost/numeric/ublas/io.hpp>
*/
#if defined(VIENNACL_WITH_OPENCL) || defined(VIENNACL_WITH_OPENMP) || defined(VIENNACL_WITH_CUDA)
#define VIENNACL
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#endif
     
     /* Handle errors by printing an error message and exiting with a
      * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp> 
using namespace boost::numeric;

#include "mpi.h"
#include <netcdf_par.h>
#include <netcdf.h>

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



template <typename ScalarType>
std::vector<ScalarType> netcdf_read_timeseries(const std::string filename, const std::vector<std::string> fields )
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
    count[i] = dimids[i];
    start[i] = 0;
  }

  if (dims[0] % mpi_processes != 0)
    {
      if (!my_rank) std::cout << "dims[0] " << dims[0] << " is not evenly divisible by mpi_size= " << mpi_processes << std::endl;
      static std::vector<ScalarType> empty_vector;
      return empty_vector;
    }
  
  count[0] = dims[0] / mpi_processes;         // Number of 2D slices.  

  slab_size = count[0];
  for (int i=0; i<ndims; ++i ) { slab_size *= count[i]; }

  data  = (ScalarType*) malloc (sizeof(ScalarType)*slab_size);

  /* Read the slab this process is responsible for. */
  start[0] = dims[0] / mpi_processes * my_rank;
  std::cout << "Rank: " << my_rank << " reading slab: " << start[0] << std::endl;
  /* Read one slab of data. */
  if ((retval = nc_get_vara_double(ncid, varid, start, count, data))) ERR(retval);

  std::vector<ScalarType> output(data, data+slab_size);

  free( dims );
  free( dimids );
  free( start );
  free( count );
  free( data );

  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);
     
  return output;
}


int main(int argc, char *argv[])
//****************************************************************************80
//
//  Purpose:
//
//    Example of parallel NetCDF functionality
//
//  Discussion:
//
//    This program demonstrates parallel NetCDF functionality.  It reads from
//    a NetCDF file a specified, specified as the first and second arguments of
//    the function.  
//
//    This is the first step toward implementing a compression backend which 
//    reads a NetCDF stream, and compresses the time series data in parallel
//    using the approaches of Horenko, et al.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//     Starting October 2013
//
//  Author:
//
//    William Sawyer (CSCS)
//
//  Reference:
//    
//    Horenko, Klein, Dolaptchiev, Schuette
//    Automated Generation of Reduced Stochastic Weather Models I:
//    simultaneous dimension and model reduction for time series analysis
//    XXX  
//
//  Example execution:
//
//    aprun -n 2 ./netcdf_test /project/csstaff/outputs/echam/echam6/echam_output/t31_196001.01_echam.nc seaice
//
{
  using namespace std;

  typedef double               ScalarType;     //feel free to change this to 'double' if supported by your hardware

  /* This will be the netCDF ID for the file and data variable. */
  int ncid, varid, dimid;
  int ndims, nvars_in, ngatts_in, unlimdimid_in;
  int my_rank, mpi_processes;

  /* Loop indexes, and error handling. */
  int x, y, retval ;
  int slab_size ; // is the number of entries in one slab to be read in

  double *data_double;
  int    *dimids;
  size_t *dims;
  size_t *p;
  size_t *start, *count;

  viennacl::ocl::set_context_device_type(1, viennacl::ocl::gpu_tag());   // Does not find the GPU

//
//  Initialize MPI.
//
  MPI_Init ( &argc, &argv );
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );   
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );

  if (argc <= 2)
    {
      std::cout << "Usage: " << argv[0] << " <Filename>" << " <field name>" << std::endl;
      exit(1);
    }


   typedef std::vector< viennacl::ocl::platform > platforms_type;
   platforms_type platforms = viennacl::ocl::get_platforms();
   viennacl::ocl::platform pf = viennacl::ocl::get_platforms()[1];

   //    for (platforms_type::iterator platform_iter = platforms.begin();
   //                                 platform_iter != platforms.end();
   //                                ++platform_iter)
   if (!my_rank)
   {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = pf.devices(CL_DEVICE_TYPE_ALL);

    //
    // print some platform info
    //
    std::cout << "# =========================================" << std::endl;
    std::cout << "# Platform Information " << std::endl;
    std::cout << "# =========================================" << std::endl;

    std::cout << "#" << std::endl;
    std::cout << "# Vendor and version: " << pf.info() << std::endl;
    std::cout << "#" << std::endl;

    //
    // traverse the devices and print the information
    //
    std::cout << "# " << std::endl;
    std::cout << "# Available Devices: " << std::endl;
    std::cout << "# " << std::endl;
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
        std::cout << std::endl;

        std::cout << " -----------------------------------------" << std::endl;
        std::cout << iter->info();
        std::cout << " -----------------------------------------" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "###########################################" << std::endl;
    std::cout << std::endl;
   }


  std::cout << "Parallel execution: my_rank " << my_rank << " out of " << mpi_processes << " processes" << std::endl;


  std::string              filename(argv[1]);
  std::vector<std::string> fields(argv+2, argv+argc);

  std::vector<ScalarType>      X = netcdf_read_timeseries<ScalarType>( filename, fields);
  viennacl::vector<ScalarType> Xvcl(X.size());

  viennacl::copy(X.begin(), X.end(), Xvcl.begin() );
  
  
  retval =  0;
  std::cout << "retval " << retval << std::endl;

//
//  Terminate MPI.
//
  MPI::Finalize ( );

  return 0;
}
