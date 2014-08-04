#define VERSION 0.1
#define KSIZE 10 // default number of clusters
#define MSIZE 5  // default number of eigenvectors used in final compression

#define MAX_ITER 100
#define TOL 1.0e-7
#define RANDOM_SEED 123456

typedef double Scalar;     // feel free to change this to 'double' if supported by your hardware



#if defined(USE_EIGEN)
#include <random>
#endif

#include <algorithm>
#include <mpi.h>
#include <mkl.h>
#include <iostream> 
#include <boost/program_options.hpp>

#include "matrices.h"
#include "NetCDFInterface.h"
#include "CompressedMatrix.h"
#include "mpi_type_helper.h"


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
  std::vector<std::string> variables;
  std::vector<std::string> compressed_dims;
  std::vector<std::string> distributed_dims;
  std::vector<std::string> indexed_dims;

  std::vector<std::string> default_compressed_dims;
  default_compressed_dims.push_back("lon");
  default_compressed_dims.push_back("lat");

  namespace po = boost::program_options;
  po::options_description po_description("USI Compression: Options");
  po_description.add_options()
    ("help", "display this help message")
    ("version", "display the version number")
    ("compressed,c", po::value< std::vector<std::string> >(&compressed_dims)
        ->default_value(default_compressed_dims, "lon,lat")->multitoken(), 
        "list of compressed dimensions")
    ("indexed,i", po::value< std::vector<std::string> >(&indexed_dims)
        ->default_value(std::vector<std::string>(0), "none")->multitoken(),
        "list of indexed dimensions")
    ("clusters,K", po::value<int>(&K_size)->default_value(KSIZE),
        "the number of clusters used for PCA (K)")
    ("eigenvectors,M", po::value<int>(&M_size)->default_value(MSIZE),
        "the number of eigenvectors used for final compression (M)")
    ("horizontal-stacking,h", "stack variables in distributed (horizontal) instead of compressed (vertical) direction")
    ("file", po::value<std::string>(&filename)->required(),
        "the path to the NetCDF4 file")
    ("variables,v", po::value< std::vector<std::string> >(&variables)->required()->multitoken(),
        "the variable that is to be compressed")
    ;

  po::positional_options_description po_positional;
  po_positional.add("file",1);
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

  NetCDFInterface<Scalar>::Stacking stacking;
  if (po_vm.count("horizontal-stacking")) {
    stacking = NetCDFInterface<Scalar>::HORIZONTAL;
  } else {
    stacking = NetCDFInterface<Scalar>::VERTICAL;
  }

  // this has to be after the help/version commands as this
  // exits with an error if the required arguments aren't
  // specified
  po::notify(po_vm);


  //
  // Read NetCDF Data
  //

  NetCDFInterface<Scalar> netcdf_interface(filename, variables,
      compressed_dims, indexed_dims, stacking);
  DeviceMatrix<Scalar> X = netcdf_interface.construct_matrix();
  
  double time_after_reading_data = MPI_Wtime();


  //
  // Compress Matrix
  //

  CompressedMatrix<Scalar> X_compressed(X, K_size, M_size);
  
  double time_after_compression = MPI_Wtime();


  //
  // Reconstruct Matrix & Measure Difference
  //

  DeviceMatrix<Scalar> X_reconstructed = X_compressed.reconstruct();
  DeviceMatrix<Scalar> X_difference = X_reconstructed - X;

  Scalar column_norm;
  Scalar local_square_norm = 0.0;
  Scalar global_square_norm = 0.0;
  for (int i = 0; i < X_difference.cols(); i++ ) {
    column_norm = GET_NORM(GET_COLUMN(X_difference, i));
    local_square_norm += column_norm * column_norm;
  }
  MPI_Allreduce( &local_square_norm, &global_square_norm, 1, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );


  //
  // Statistics Output
  //

  double time_for_solve = time_after_compression -  time_after_reading_data;
  double time_for_input = time_after_reading_data - time_at_start;
  
  double max_time_for_solve, max_time_for_input;
  MPI_Allreduce( &time_for_solve, &max_time_for_solve, 1, mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &time_for_input, &max_time_for_input, 1, mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD );

  if (!my_rank) std::cout << "Max time for input " << max_time_for_input << std::endl;
  if (!my_rank) std::cout << "Max time for solve " << max_time_for_solve << std::endl;
  if (!my_rank) std::cout << "Compression ratio  " << 
      X_compressed.original_size / (double) X_compressed.compressed_size
      << std::endl;
  if (!my_rank) std::cout << "Root mean square error " << sqrt( global_square_norm ) << std::endl;


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
