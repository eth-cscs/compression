#define KSIZE 10 // default number of clusters
#define MSIZE 5  // default number of eigenvectors used in final compression

#define MAX_ITER 100
#define TOL 1.0e-7
#define RANDOM_SEED 123456

#include <random>
#include <algorithm>
#include <mpi.h>
#include <iostream> 

#include <boost/program_options.hpp>

#include "usi_compression.h"
#include "read_from_netcdf.h"
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

  //
  //  Initialize MPI
  //

  MPI_Init ( &argc, &argv );
  double time_at_start = MPI_Wtime();   // used for calculating running time of each part of the algorithm

  int my_rank, mpi_processes;
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );   
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );


  //
  // Input Argument Handling
  //

  int M_size, K_size;
  std::string filename;
  std::string variable_name;
  std::vector<std::string> compressed_dims;
  std::vector<std::string> distributed_dims;

  namespace po = boost::program_options;
  po::options_description po_description("USI Compression: Options");
  po_description.add_options()
    ("help", "display this help message")
    ("version", "display the version number")
    ("compressed,c", po::value<std::vector<std::string>>(&compressed_dims)
        ->default_value({"lon", "lat"}, "lon,lat"), "list of compressed dimensions")
    ("distributed,d", po::value<std::vector<std::string>>(&distributed_dims)
        ->default_value({"mlev", "time"}, "mlev,time"), "list of distributed dimensions")
    ("clusters,K", po::value<int>(&K_size)->default_value(KSIZE),
        "the number of clusters used for PCA (K)")
    ("eigenvectors,M", po::value<int>(&M_size)->default_value(MSIZE),
        "the number of eigenvectors used for final compression (M)")
    ("file", po::value<std::string>(&filename)->required(),
        "the path to the NetCDF4 file")
    ("variable", po::value<std::string>(&variable_name)->required(),
        "the variable that is to be compressed")
    ;

  po::positional_options_description po_positional;
  po_positional.add("file",1).add("variable",1);
  po::variables_map po_vm;
  po::store(po::command_line_parser(argc, argv).options(po_description)
      .positional(po_positional).run(), po_vm);

  if (po_vm.count("help")) {
    if (!my_rank) std::cout << po_description << std::endl;
    return 1; // TODO: should this be an error or not?
  }

  if (po_vm.count("version")) {
    if (!my_rank) std::cout << "USI Compression, Version " << VERSION << std::endl;
    return 1; // TODO: should this be an error or not?
  }

  // this has to be after the help/version commands as this
  // exits with an error if the required arguments aren't
  // specified
  po::notify(po_vm);


  //
  // Read NetCDF Data
  //

  GenericColMatrix X = read_from_netcdf<ScalarType>(filename, variable_name, compressed_dims, distributed_dims);
  
  double time_after_reading_data = MPI_Wtime();


  //
  // Set Up Data Used In Algorithm
  //

  // dimensions of data are saved as Ntl and nl for convenience & clarity
  const int Ntl = X.rows();  // number of values along direction that is
                             // compressed (variables/parameters in PCA)
  const int nl =  X.cols();  // number of values along direction that is 
                             // distributed on cores (observations in PCA)

  // we need the global nl to generate the initial cluster configuration
  // in a consistent way for any PE configuration
  int *nl_global = new int[sizeof(int)*mpi_processes];
  int total_nl = 0;
  MPI_Allgather( &nl, 1, MPI_INT, nl_global, 1, MPI_INT, MPI_COMM_WORLD);
  for ( int rank=0; rank < mpi_processes; rank++ ) { total_nl += nl_global[rank]; } 
  std::vector<int> gamma_ind = gamma_zero(nl_global, my_rank, K_size);
  // print out nl sizes of all processes (for debugging)
  // std::cout << "nl sizes "; for ( int rank=0; rank < mpi_processes; rank++ ) { std::cout << nl_global[rank] << " "; } 
  // std::cout << std::endl;
  delete[] nl_global;

  // we allocate the matrices used for the algorithm outside of the loop
  GenericColMatrix theta(Ntl,K_size);                   // Time series means (one for each k), allocate outside loop
  std::vector<GenericColMatrix> TT(K_size,GenericColMatrix(Ntl,1) );        // Eigenvectors: 1-each for each k
  std::vector<GenericColMatrix> EOFs(K_size,GenericColMatrix(Ntl,M_size) );  // Eigenvectors: M_size eigenvectors for each k
  std::vector<GenericColMatrix> Xreduced(K_size,GenericColMatrix(M_size, nl) );  // Reduced representation of X
  GenericRowMatrix Xreconstructed(Ntl,nl);                                 // Reconstructed time series 
  GenericRowMatrix Diff(Ntl,nl);                                 // Reconstructed time series 

  ScalarType L_value_old = 1.0e19;   // Very big value
  ScalarType L_value_new;

#if defined( USE_EIGEN )
  // initialize random seed used in lanczos algorithm
  srand(RANDOM_SEED);
#endif


  //
  // Iterate To Find Optimal Clustering
  //

  for ( int iter = 0; iter < MAX_ITER; iter++ ) {

    // determine X column means for each active state denoted by gamma_ind
    theta_s<ScalarType>(gamma_ind, X, theta);
    
    // we calculate the L value here for output only (TODO: remove this for optimization later)
    L_value_new =  L_value( gamma_ind, TT, X, theta );
    if (!my_rank) std::cout << "L value after Theta calc " << L_value_new << std::endl;
    // Not clear if there should be monotonic decrease here:  new theta_s needs new TT, right?
    
    // Principle Component Analysis for every cluster
    for(int k = 0; k < K_size; k++) {
      std::vector<int> Nonzeros = find( gamma_ind, k );
      GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
      // if (!my_rank) std::cout << " For k = " << k << " nbr nonzeros " << Nonzeros.size() << std::endl;
      for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
      {
        GET_COLUMN(Xtranslated,m) = GET_COLUMN(X,Nonzeros[m]) - GET_COLUMN(theta,k) ; 
      }
      bool success = lanczos_correlation(Xtranslated, 1, 1.0e-11, 50, TT[k], true);
    }

    // we calculate the L value here for output only (TODO: remove this for optimization later)
    L_value_new =  L_value( gamma_ind, TT, X, theta );
    if (!my_rank) std::cout << "L value after PCA " << L_value_new << std::endl;
    
    // find new optimal clustering
    gamma_s( X, theta, TT, gamma_ind );

    // calculate new L value and decide whether to continue
    L_value_new =  L_value( gamma_ind, TT, X, theta ); 
    if (!my_rank) std::cout << "L value after gamma minimization " << L_value_new << std::endl;
    if ( (L_value_old - L_value_new) < L_value_new*TOL ) {
      if (!my_rank) std::cout << " Converged: to tolerance " << TOL << " after " << iter << " iterations " << std::endl;
      break;
    }
    else if ( L_value_new > L_value_old ) { 
      if (!my_rank) std::cout << "New L_value " << L_value_new << " larger than old: " << L_value_old << " aborting " << std::endl;
      break;
    }
    else if ( iter+1 == MAX_ITER ) {
      if (!my_rank) std::cout << " Reached maximum number of iterations " << MAX_ITER << " without convergence " << std::endl;
    }
    L_value_old = L_value_new;
  }


  //
  // Do Compression With Optimal Clustering
  //

  // Determine X column means for each active state denoted by gamma_ind
  theta_s<ScalarType>(gamma_ind, X, theta);
  
  // Principal Component Analysis for every cluster
  for(int k = 0; k < K_size; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );
    GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
    for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
    {
      GET_COLUMN(Xtranslated,m) = GET_COLUMN(X,Nonzeros[m]) - GET_COLUMN(theta,k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
    }
    lanczos_correlation(Xtranslated, M_size, 1.0e-8, Ntl, EOFs[k], true);
  }
  L_value_new =  L_value( gamma_ind, EOFs, X, theta );
  if (!my_rank) std::cout << "L value final " << L_value_new << std::endl;

  // Calculated the reduced representation of X
  reduction<ScalarType>( gamma_ind, EOFs, X, theta, Xreduced );

  // we save the time to check the runtime for each part of the algorithm
  double time_after_compression = MPI_Wtime();


  //
  // Reconstruct Data for Comparison with Original
  //

  reconstruction<ScalarType>(gamma_ind, EOFs, theta, Xreduced, Xreconstructed);
  ScalarType colnorm;
  ScalarType value  = 0.0;
  ScalarType output;
  for (int l = 0; l < nl; l++ ) { 
#if defined( USE_EIGEN )
    Diff.col(l) = Xreconstructed.col(l)-X.col(l);
    colnorm = (Xreconstructed.col(l)-X.col(l)).norm();
    value += colnorm*colnorm;
#elif defined( USE_MINLIN )
    Diff(all,l) = Xreconstructed(all,l)-X(all,l);
    colnorm = norm(Diff(all,l));
    value += colnorm*colnorm;
#else
    ERROR:  must USE_EIGEN or USE_MINLIN
#endif
  }
  MPI_Allreduce( &value, &output, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


  //
  // Statistics Output
  //

  double time_for_solve = time_after_compression -  time_after_reading_data;
  double time_for_input = time_after_reading_data - time_at_start;
  
  double max_time_for_solve, max_time_for_input;
  MPI_Allreduce( &time_for_solve, &max_time_for_solve, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &time_for_input, &max_time_for_input, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

  double size_uncompressed = (double) (Ntl * total_nl);
  double size_compressed = (double) (K_size * ( M_size + 1) *( Ntl + total_nl ) );

  if (!my_rank) std::cout << "Max time for input " << max_time_for_input << std::endl;
  if (!my_rank) std::cout << "Max time for solve " << max_time_for_solve << std::endl;
  if (!my_rank) std::cout << "Compression ratio  " << size_uncompressed / size_compressed << std::endl;
  if (!my_rank) std::cout << "Root mean square error " << sqrt( output ) << std::endl;


  //
  // Write Reconstructed Data to File
  //

  int retval;
  // TODO: this line doesn't return, fix this
  //if ((retval = nc_put_vara_double(ncid_out, varid_out, start, count, GET_POINTER(Xreconstructed) ))) ERR(retval);
  //if ((retval = nc_close(ncid_out))) ERR(retval);


  //
  //  Terminate MPI and Quit Program.
  //

  retval =  0;
  if (!my_rank) std::cout << "retval " << retval << std::endl;

  MPI_Finalize();

  return 0;
}
