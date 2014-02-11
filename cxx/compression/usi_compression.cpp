  // Size of space
#define KSIZE 10
  // Number of EOF used in final compression
#define MSIZE 5

#define MAX_ITER 100
#define TOL 1.0e-7

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
#include "find.h"
#include "theta_s.h"
#include "L_value.h"

#if defined( USE_EIGEN )

#include "gamma_s.h"
#include "lanczos_correlation.h"

#endif

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
//    Ben Cumming (CSCS)
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
//    aprun -n 2 ./netcdf_get_data /project/csstaff/outputs/echam/echam6/echam_output/t31_196001.01_echam.nc seaice
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

  if (argc <= 2) {
    std::cout << "Usage: " << argv[0] << " <Filename>" << " <field name>" << std::endl;
    exit(1);
  }

  int pes_in_x;
  int pes_in_y = 1;

  for ( pes_in_x = mpi_processes; pes_in_x > pes_in_y; pes_in_x /= 2 ) { pes_in_y *= 2; }

  const int iam_in_x = my_rank % pes_in_x;
  const int iam_in_y = my_rank / pes_in_x;

  std::cout << "Decomposition:  pes_in_x " << pes_in_x << " pes_in_y " << pes_in_y << " iam x " << iam_in_x << " iam_in_y " << iam_in_y << std::endl;

  if ( pes_in_x * pes_in_y != mpi_processes ) { std::cout << "mpi_processes " << mpi_processes << " not power of two; aborting " << std::endl; abort(); }




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


   // std::cout << "Parallel execution: my_rank " << my_rank << " out of " << mpi_processes << " processes" << std::endl;


  std::string              filename(argv[1]);
  std::vector<std::string> fields(argv+2, argv+argc);
  int  Xrows, Xcols;

  ScalarType*  data = read_timeseries_matrix<ScalarType>( filename, fields, iam_in_x, iam_in_y, pes_in_x, pes_in_y, Xrows, Xcols );

  std::cout << "Creating Time Series Matrix " << Xrows << " X " << Xcols << std::endl;
#if defined( USE_EIGEN )
  Map<MatrixXXrow> X(data,Xrows,Xcols);       // Needs to be row-major to mirror NetCDF output
#elif defined( USE_MINLIN )
  HostMatrix<ScalarType> X(Xrows,Xcols);
  for(int i=0; i<Xrows; i++)
    for(int j=0; j<Xcols; j++, data++)
       X(i,j) = *data;
#endif


  //
  // want vector of length X.rows() of random values between {0,k-1}
  //

  // create a blank vector of length X.rows()

  const int Ntl = Xrows;
  const int nl =  Xcols;

  int *nl_global;
  nl_global = (int*) malloc (sizeof(int)*mpi_processes);
  MPI_Allgather( &nl, 1, MPI_INT, nl_global, 1, MPI_INT, MPI_COMM_WORLD);
  // std::cout << "nl sizes "; for ( int rank=0; rank < mpi_processes; rank++ ) { std::cout << nl_global[rank] << " "; } 
  // std::cout << std::endl;

  std::vector<int> gamma_ind = gamma_zero(nl_global, my_rank, KSIZE ); // Needs to be generated in a consistent way for any PE configuration
#if defined( USE_EIGEN )
  // ArrayX1i gamma_ind(gamma_initial.data());  // Error:  YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR
  // ArrayX1i gamma_ind(gamma_initial.size()) ;
  // std::copy(gamma_initial.begin(), gamma_initial.end(), gamma_ind.data());
  MatrixXX theta(Ntl,KSIZE);                   // Time series means (one for each k), allocate outside loop
  // MatrixXX theta = MatrixXX::Zero(Ntl,KSIZE);                   // Time series means (one for each k), allocate outside loop
  MatrixXX eigenvectors( Ntl, 1 ) ;    // Only one eigenvector for the iteration stage
  std::vector<MatrixXX> TT(KSIZE, MatrixXX::Zero(Ntl,1));
  std::vector<MatrixXX> EOFs(KSIZE, MatrixXX::Zero(Ntl,MSIZE));
#elif defined( USE_MINLIN )

#if defined( USE_GPU )
#else
  // HostVector<int> gamma_ind(gamma_initial.size());   // Needs to be generated in a consistent way for any PE configuration
  // std::copy(gamma_initial.begin(), gamma_initial.end(), gamma_ind.pointer());
  HostMatrix<ScalarType> theta(Ntl,KSIZE);      // Time series means (one for each k), allocate outside loop
  theta(all) = 0.;   // Check if this is necessary
  HostMatrix<ScalarType> eigenvectors( Ntl, 1 ) ;   // Only one eigenvector for the iteration stage

  std::vector<HostMatrix<ScalarType>> TT(KSIZE,HostMatrix<ScalarType>(Ntl,1) );
  TT[0](all) = 0.;
  std::vector<HostMatrix<ScalarType>> EOFs(KSIZE,HostMatrix<ScalarType>(Ntl,MSIZE) );
#endif
  
#endif

  ScalarType L_value_old = 1.0e19;   // Very big value
  ScalarType L_value_new;  

  for ( int iter = 0; iter < MAX_ITER; iter++ ) {
    theta_s<ScalarType>(gamma_ind, X, theta);       // Determine X column means for each active state denoted by gamma_ind
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    if (!my_rank) std::cout << "L value after Theta calc " << L_value_new << std::endl;
    // Not clear if there should be monotonic decrease here:  new theta_s needs new TT, right?
    // if ( iter > 0 ) { std::cout << "L value after theta determination " << L_value( gamma_ind, TT, X, theta ) << std::endl; }
    for(int k = 0; k < KSIZE; k++) {              // Principle Component Analysis
      std::vector<int> Nonzeros = find( gamma_ind, k );
      GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
      // if (!my_rank) std::cout << " For k = " << k << " nbr nonzeros " << Nonzeros.size() << std::endl;
      for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
      {
#if defined( USE_EIGEN )
	Xtranslated.col(m) = X.col(Nonzeros[m]) - theta.col(k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
#elif defined( USE_MINLIN )
	Xtranslated(all,m) = X(all,Nonzeros[m]) - theta(all,k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
#else
        ERROR:  must USE_EIGEN or USE_MINLIN
#endif
      }
      //      lanczos_correlation(Xtranslated.block(0,0,Ntl,Nonzeros.size()), 1, 1.0e-13, 50, TT[k], true);
      lanczos_correlation(Xtranslated, 1, 1.0e-13, 50, TT[k], true);
    }
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    if (!my_rank) std::cout << "L value after PCA " << L_value_new << std::endl;
#if defined( USE_EIGEN )
    gamma_s( X, theta, TT, gamma_ind );
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    if (!my_rank) std::cout << "L value after gamma minimization " << L_value_new << std::endl;
    if ( L_value_new > L_value_old ) { 
      if (!my_rank) std::cout << "New L_value " << L_value_new << " larger than old: " << L_value_old << " aborting " << std::endl;
      break;
    }
    else if ( (L_value_old - L_value_new) < L_value_new*TOL ) {
      if (!my_rank) std::cout << " Converged: to tolerance " << TOL << " after " << iter << " iterations " << std::endl;
      break;
    }
    else if ( iter+1 == MAX_ITER ) {
      if (!my_rank) std::cout << " Reached maximum number of iterations " << MAX_ITER << " without convergence " << std::endl;
    }
    L_value_old = L_value_new;
#endif
  }

#if defined( USE_EIGEN)
  theta_s<ScalarType>(gamma_ind, X, theta);       // Determine X column means for each active state denoted by gamma_ind
  for(int k = 0; k < KSIZE; k++) {              // Principle Component Analysis
    std::vector<int> Nonzeros = find( gamma_ind, k );
    GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
    for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
    {
      Xtranslated.col(m) = X.col(Nonzeros[m]) - theta.col(k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
    }
    // lanczos_correlation(Xtranslated.block(0,0,Ntl,Nonzeros.size()), MSIZE, 1.0e-8, Ntl, EOFs[k], true);
    lanczos_correlation(Xtranslated, MSIZE, 1.0e-8, Ntl, EOFs[k], true);
  }
  L_value_new =  L_value( gamma_ind, EOFs, X, theta );
  if (!my_rank) std::cout << "L value final " << L_value_new << std::endl;

//
//  Terminate MPI.
//
#endif
  retval =  0;
  if (!my_rank) std::cout << "retval " << retval << std::endl;

  MPI_Finalize ( );

  return 0;
}
