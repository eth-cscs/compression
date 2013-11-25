/*
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

#include <boost/numeric/ublas/io.hpp>
*/

#define MAX_ITER 100
#define TOL 1.0e-3

#include <random>
#include <algorithm>

#if defined(VIENNACL_WITH_OPENCL) || defined(VIENNACL_WITH_OPENMP) || defined(VIENNACL_WITH_CUDA)
#define VIENNACL
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#endif

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp> 
using namespace boost::numeric;

#include "usi_compression.h"
#include "read_timeseries_matrix.h"
#include "gamma_zero.h"
#include "gamma_s.h"
#include "find.h"
#include "theta_s.h"
#include "lanczos_correlation.h"
#include "L_value.h"
//

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


  /* This will be the netCDF ID for the file and data variable. */
  int ncid, varid, dimid;
  int ndims, nvars_in, ngatts_in, unlimdimid_in;
  int my_rank, mpi_processes;

  /* Loop indexes, and error handling. */
  int x, y, retval ;
  int slab_size ; // is the number of entries in one slab to be read in

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

  typedef Array<int, Dynamic, 1> ArrayX1i;

  MatrixXX   X = read_timeseries_matrix<ScalarType>( filename, fields);



  /*
  VectorXd  b = VectorXd::Random(X.cols());
  VectorXd  Xb = X * b;
  std::cout << "matrix rows " << X.rows() << " cols " << X.cols() << std::endl;
  std::cout << "norm rand vect " << b.norm() << " norm matrix * random vector " << Xb.norm() << std::endl;
  */
  
  //
  // want vector of length X.rows() of random values between {0,k-1}
  //

#define K 10

  // create a blank vector of length X.rows()
  const int Ntl = static_cast<int>(X.rows());
  const int nl = static_cast<int>(X.cols());
  // std::cout << "First row X " << X.block(0,0,Ntl,5) << std::endl;
  ArrayX1i gamma_ind = gamma_zero(nl, K );
  MatrixXX theta = MatrixXX::Zero(Ntl,K);       // Time series means (one for each k), allocate outside loop
  MatrixXX TT(Ntl,K);                 // Eigenvectors (one for each k), allocate outside loop
  MatrixXX Xtranslated( Ntl, nl ) ;   // Maximum size for worst case (all nl in one K)
  MatrixXX eigenvectors( Ntl, 1 ) ;   // Only one eigenvector at this stage

  ScalarType L_value_old = 1.0e19;   // Very big value
  ScalarType L_value_new;  
  for ( int iter = 0; iter < MAX_ITER; iter++ ) {
    theta_s<ScalarType>(gamma_ind, X, theta);       // Determine X column means for each active state denoted by gamma_ind
    // if ( iter > 0 ) { std::cout << "L value after theta determination " << L_value( gamma_ind, TT, X, theta ) << std::endl; }
    for(int k = 0; k < K; k++) {              // Principle Component Analysis
      std::vector<int> Nonzeros = find( gamma_ind, k );
      //      std::cout << " For k = " << k << " nbr nonzeros " << Nonzeros.size() << std::endl;
      for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
      {
	Xtranslated.col(m) = X.col(Nonzeros[m]) - theta.col(k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
      }
      lanczos_correlation(Xtranslated.block(0,0,Ntl,Nonzeros.size()), 1, 1.0e-7, 20, eigenvectors);
      TT.col(k) = eigenvectors.col(0);
    }
    std::cout << "L value after PCA " << L_value( gamma_ind, TT, X, theta ) << std::endl;
    gamma_s( X, theta, TT, gamma_ind );
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    std::cout << "L value after gamma minimization " << L_value_new << std::endl;
    if ( L_value_new > L_value_old ) { 
      std::cout << "New L_value " << L_value_new << " larger than old: " << L_value_new << " aborting " << std::endl;
      break;
    }
    else if ( (L_value_old - L_value_new) < L_value_new*TOL ) {
      std::cout << " Converged: to tolerance " << TOL << std::endl;
      break;
    }
    L_value_old = L_value_new;
  }

//  auto result1 = std::find_if(gamma_ind.data(), gamma_ind.data()+gamma_ind.rows(), std::bind2nd (std::equal_to<int>(), 4));
//  std::cout << "result of find operation is " << std::endl 
//    << result1.format(CommaInitFmt) << std::endl;

/*
  for(int k = 0; k < theta.cols(); k++) {
    std::cout << "index set for k =  " << k << std::endl ;
    std::vector<int> found_items = find( gamma_ind, k );

    copy(found_items.begin(), found_items.end(), ostream_iterator<int>(cout, ", "));
    std::cout << std::endl ;
  }
*/

//
//  Terminate MPI.
//
  retval =  0;
  std::cout << "retval " << retval << std::endl;

  MPI::Finalize ( );

  return 0;
}
