/*
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

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


#include "read_timeseries_matrix.hpp"

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

#if defined(VIENNACL_WITH_OPENCL) || defined(VIENNACL_WITH_OPENMP) || defined(VIENNACL_WITH_CUDA)
  viennacl::ocl::set_context_device_type(1, viennacl::ocl::gpu_tag());   // Does not find the GPU
#endif

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


#if defined(VIENNACL_WITH_OPENCL) || defined(VIENNACL_WITH_OPENMP) || defined(VIENNACL_WITH_CUDA)
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
#endif


  std::cout << "Parallel execution: my_rank " << my_rank << " out of " << mpi_processes << " processes" << std::endl;


  std::string              filename(argv[1]);
  std::vector<std::string> fields(argv+2, argv+argc);

  Eigen::Matrix<ScalarType,Dynamic,Dynamic>      X = read_timeseries_matrix<ScalarType>( filename, fields);
  VectorXd                                       b = VectorXd::Random(X.cols());
  VectorXd                                       Xb = X * b;

  std::cout << "matrix rows " << X.rows() << " cols " << X.cols() << std::endl;
  std::cout << "norm rand vect " << b.norm() << " norm matrix * random vector " << Xb.norm() << std::endl;
  
  retval =  0;
  std::cout << "retval " << retval << std::endl;

//
//  Terminate MPI.
//
  MPI::Finalize ( );

  return 0;
}
