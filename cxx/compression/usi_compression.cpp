  // Size of space
#define KSIZE 10
  // Number of EOF used in final compression
#define MSIZE 5

#define MAX_ITER 100
#define TOL 1.0e-7
#define RANDOM_SEED 123456

#include <random>
#include <algorithm>
#include <mpi.h>
#include <iostream> 

#include "usi_compression.h"
#include "read_timeseries_matrix.h"
#include "gamma_zero.h"
#include "find.h"
#include "theta_s.h"
#include "L_value.h"
#include "lanczos_correlation.h"
#include "gamma_s.h"
#include "reduction.h"
#include "reconstruction.h"

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

  /* This will be the netCDF ID for the file and data variable. */
  int ncid_out, varid_out, dimid;
  int nvars_in, ngatts_in, unlimdimid_in;
  int my_rank, mpi_processes;

  /* Loop indexes, and error handling. */
  int x, y, retval ;
  int slab_size ; // is the number of entries in one slab to be read in

  size_t *p;
  size_t *start, *count;

//
//  Initialize MPI.
//
  double t1, t2, t3, dt_input, dt_solve, size_uncompr, size_compr;

  MPI_Init ( &argc, &argv );
  t1 = MPI_Wtime();   // Start of program
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



  std::string              filename(argv[1]);
  std::vector<std::string> fields(argv+2, argv+argc);
  int  Xrows, Xcols;

  ScalarType*  data = read_timeseries_matrix<ScalarType>( filename, fields, iam_in_x, iam_in_y, pes_in_x, pes_in_y, Xrows, Xcols, &start, &count, &ncid_out, &varid_out );

  // std::cout << "Creating Time Series Matrix " << Xrows << " X " << Xcols << std::endl;
#if defined( USE_EIGEN )
  Eigen::Map<MatrixXXrow> X(data,Xrows,Xcols);       // Needs to be row-major to mirror NetCDF output
#elif defined( USE_MINLIN )
  HostMatrix<ScalarType> X(Xrows,Xcols);
  for(int i=0; i<Xrows; i++)
    for(int j=0; j<Xcols; j++, data++)
       X(i,j) = *data;
#endif

  t2 = MPI_Wtime();   // After data have been read in

  //
  // want vector of length X.rows() of random values between {0,k-1}
  //

  // create a blank vector of length X.rows()

  const int Ntl = Xrows;
  const int nl =  Xcols;

  int *nl_global;
  int total_nl = 0;

  nl_global = (int*) malloc (sizeof(int)*mpi_processes);
  MPI_Allgather( &nl, 1, MPI_INT, nl_global, 1, MPI_INT, MPI_COMM_WORLD);
  for ( int rank=0; rank < mpi_processes; rank++ ) { total_nl += nl_global[rank]; } 

  size_uncompr = (double) (Ntl * total_nl);

  // std::cout << "nl sizes "; for ( int rank=0; rank < mpi_processes; rank++ ) { std::cout << nl_global[rank] << " "; } 
  // std::cout << std::endl;

  std::vector<int> gamma_ind = gamma_zero(nl_global, my_rank, KSIZE ); // Needs to be generated in a consistent way for any PE configuration

  GenericColMatrix theta(Ntl,KSIZE);                   // Time series means (one for each k), allocate outside loop
  std::vector<GenericColMatrix> TT(KSIZE,GenericColMatrix(Ntl,1) );        // Eigenvectors: 1-each for each k
  std::vector<GenericColMatrix> EOFs(KSIZE,GenericColMatrix(Ntl,MSIZE) );  // Eigenvectors: MSIZE eigenvectors for each k
  std::vector<GenericColMatrix> Xreduced(KSIZE,GenericColMatrix(MSIZE, nl) );  // Reduced representation of X
  GenericRowMatrix Xreconstructed(Ntl,nl);                                 // Reconstructed time series 
  GenericRowMatrix Diff(Ntl,nl);                                 // Reconstructed time series 

  ScalarType L_value_old = 1.0e19;   // Very big value
  ScalarType L_value_new;
  bool success;

  // initialize random seed used in lanczos algorithm in loop
  srand(RANDOM_SEED);

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
        GET_COLUMN(Xtranslated,m) = GET_COLUMN(X,Nonzeros[m]) - GET_COLUMN(theta,k) ; 
      }
      //      lanczos_correlation(Xtranslated.block(0,0,Ntl,Nonzeros.size()), 1, 1.0e-13, 50, TT[k], true);
      success = lanczos_correlation(Xtranslated, 1, 1.0e-11, 50, TT[k], true);
    }
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    if (!my_rank) std::cout << "L value after PCA " << L_value_new << std::endl;
    
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
  }

  theta_s<ScalarType>(gamma_ind, X, theta);       // Determine X column means for each active state denoted by gamma_ind
  for(int k = 0; k < KSIZE; k++) {              // Principle Component Analysis
    std::vector<int> Nonzeros = find( gamma_ind, k );
    GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
    for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
    {
      GET_COLUMN(Xtranslated,m) = GET_COLUMN(X,Nonzeros[m]) - GET_COLUMN(theta,k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
    }
    // lanczos_correlation(Xtranslated.block(0,0,Ntl,Nonzeros.size()), MSIZE, 1.0e-8, Ntl, EOFs[k], true);
    lanczos_correlation(Xtranslated, MSIZE, 1.0e-8, Ntl, EOFs[k], true);
  }
  L_value_new =  L_value( gamma_ind, EOFs, X, theta );
  if (!my_rank) std::cout << "L value final " << L_value_new << std::endl;

  // Calculated the reduced representation of X

  reduction<ScalarType>( gamma_ind, EOFs, X, theta, Xreduced );

  t3 = MPI_Wtime();   // After data have been read in

  // Calculate the reconstructed X

  reconstruction<ScalarType>( gamma_ind, EOFs, theta, Xreduced, Xreconstructed );
  ScalarType value  = 0.0;
  ScalarType output;
  for (int l = 0; l < nl; l++ ) { 
#if defined( USE_EIGEN )
    Diff.col(l) = Xreconstructed.col(l)-X.col(l);
    ScalarType colnorm = (Xreconstructed.col(l)-X.col(l)).norm(); value += colnorm*colnorm;
#elif defined( USE_MINLIN )
    // TODO: MINLIN implementation
#else
    ERROR:  must USE_EIGEN or USE_MINLIN
#endif
  }  
  MPI_Allreduce( &value, &output, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


  t3 -= t2;    // delta time solve
  t2 -= t1;    // delta time input
  
  size_compr = (double) (KSIZE * ( MSIZE + 1) *( Ntl + total_nl ) );

  MPI_Allreduce( &t3, &dt_solve, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &t2, &dt_input, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

  if (!my_rank) std::cout << "Max time for input " << dt_input << std::endl;
  if (!my_rank) std::cout << "Max time for solve " << dt_solve << std::endl;
  if (!my_rank) std::cout << "Compression ratio  " << size_uncompr / size_compr << std::endl;
  if (!my_rank) std::cout << "Root mean square error " << sqrt( output ) << std::endl;

  
// OUTPUT
  if ((retval = nc_put_vara_double(ncid_out, varid_out, start, count, Xreconstructed.data() ))) ERR(retval);
// OUTPUT FILE will be closed in main program
  if ((retval = nc_close(ncid_out))) ERR(retval);


//
//  Terminate MPI.
//

  retval =  0;
  if (!my_rank) std::cout << "retval " << retval << std::endl;

  MPI_Finalize ( );

  return 0;
}
