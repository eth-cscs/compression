/** \file usi_compression.cpp
 *
 *  Compression of NetCDF data based on an algorithm developped by Prof. Illia
 *  Horenko at Universita della Svizzera italiana (USI).
 *
 *  Reference:
 *  Horenko, Klein, Dolaptchiev, Schuette
 *  Automated Generation of Reduced Stochastic Weather Models I:
 *  simultaneous dimension and model reduction for time series analysis
 *
 *  \copyright Copyright (c) 2014,
 *             Universita della Svizzera italiana (USI) &
 *             Centro Svizzero di Calcolo Scientifico (CSCS).
 *             All rights reserved.
 *             This software may be modified and distributed under the terms of
 *             the BSD license. See the [LICENSE file](LICENSE.md) for details.
 *
 *  \author Will Sawyer (CSCS)
 *  \author Ben Cumming (CSCS)
 *  \author Manuel Schmid (CSCS)
 */

/** \mainpage USI Compression Algorithm
 *
 * \section readme Compiling and Running the Program
 * Information on how to compile the code (including requirements) and its
 * usage can be found in the [README](README.md) file.
 *
 * \section overview Code Overview
 * The code has two main parts: The class NetCDFInterface, which provides the
 * mapping between the NetCDF file and the matrix used for compression, and
 * the class CompressedMatrix, which is responsible for compressing and
 * reconstructing the data.
 *
 * The class NetCDFInterface provides two main functions:
 * NetCDFInterface::construct_matrix() and NetCDFInterface::write_matrix().
 * In the constructor, several internal data structures are set up to describe
 * how the multi-variable, multi-dimensional data from the NetCDF file will be
 * converted to a single 2D matrix. This information is then used by
 * construct_matrix() to create the matrix that can be used for compression.
 * The reconstructed matrix can the be written back to a NetCDF file with
 * write_matrix(), which ensures that each part of the matrix is written to
 * the correct variable using the same mapping as for reading the data. Some
 * other member functions provide information about this mapping that is used
 * e.g. for calculating statistics in print_statistics().
 *
 * The class CompressedMatrix runs the compression algorithm on the data. It
 * doesn't need any information on how the data is structured within the
 * matrix. The bulk of the work is done in the constructor, which sets up the
 * compressed representation of the matrix. It then provides a function
 * CompressedMatrix::reconstruct() that reconstructs the full matrix
 * from the compressed version, apart from compression losses. The Lanczos
 * algorithm used for finding the eigenvectors is implemented in two separate
 * files, lanczos_correlation_eigen.h and lanczos_correlation_minlin.h, using
 * Eigen or minlin as linear algebra backend.
 *
 * The function print_statistics() can compare the reconstructed data to the
 * original data and prints out statistics comparing the two. As these
 * statistics are calculated for each variable, it needs information on where
 * the data for each variable is located in the matrix.
 *
 * The main() function in usi_compression.cpp is responsible for tying all the
 * parts together. It parses all the command line options, constructs the
 * matrix, compresses and decompresses the data, optionally prints out
 * statistics comparing the two matrices and writes the reconstructed data
 * back to a NetCDF file.
 *
 * The other files are small helpers where matrices.h contains type
 * definitions for matrices and vectors, mpi_type_helper.h contains a little
 * helper for using MPI types in templated code, and usi_compression.cu just
 * includes usi_compression.cpp as NVCC doesn't want to compile .cpp files.
 *
 * \see main(), NetCDFInterface, CompressedMatrix, print_statistics(),
 *      matrices.h
 *
 * \section todo Issues and Future Development
 * Issues and plans for future development are collected on the
 * [to-do list](TODO.md).
 */

#include <iostream> // std::cout, std::endl
#include <string>   // std::string
#include <vector>   // std::vector
#include <boost/program_options.hpp>
#include <mpi.h>    // MPI_Init, MPI_Comm_size, MPI_Comm_rank, MPI_Allreduce, MPI_Finalize
#include "mpi_type_helper.h"
#include "matrices.h"
#include "NetCDFInterface.h"
#include "CompressedMatrix.h"
#include "matrix_statistics.h"

#define VERSION 0.1 ///< Version number of the program.
#define KSIZE 10 ///< Default number of clusters.
#define MSIZE 5  ///< Default number of eigenvectors used for final compression.
#define RANDOM_SEED 123456  ///< Random seed for initial vector in Lanczos algorithm (Eigen version only)

/**
 * \brief The floating point number type used for all computations. Use
 * 'float' or 'double' depending on hardware support.
 */
typedef double Scalar;

/**
 * The main function of the compression algorithm. It reads the command line
 * arguments, sets up MPI, reads the NetCDF file, compresses the data,
 * reconstructs the data, prints out statistics about the compression, writes
 * the data back to a NetCDF file and closes NetCDF and MPI connections.
 *
 * \see NetCDFInterface, CompressedMatrix, print_statistics
 *
 * \param[in] argc  Number of command line arguments.
 * \param[in] argv  Command line arguments.
 */
int main(int argc, char *argv[]) {

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
  std::string filename_out;
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
    ("statistics,s", "print out statistics about the compression at the end")
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
    ("output-file,o", po::value<std::string>(&filename_out)
        ->default_value(std::string(), "FILE_reconstructed.nc4"),
        "the path to the NetCDF4 output file")
    ("append,a", "append reconstructed variables to output file instead of overwriting it")
    ("variables,v", po::value< std::vector<std::string> >(&variables)->required()->multitoken(),
        "the variable that is to be compressed")
    ;

  po::positional_options_description po_positional;
  po_positional.add("file",1);
  po::variables_map po_vm;
  po::store(po::command_line_parser(argc, argv).options(po_description)
      .positional(po_positional).run(), po_vm);

  bool append = false;
  if (po_vm.count("append")) {
    append = true;
  }

  if (po_vm.count("help")) {
    if (!my_rank) std::cout << po_description << std::endl;
    return 0;
  }

  if (po_vm.count("version")) {
    if (!my_rank) std::cout << "USI Compression, Version " << VERSION << std::endl;
    return 0;
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

  std::vector<int> col_ids = netcdf_interface.get_column_ids();
  CompressedMatrix<Scalar> X_compressed(X, K_size, M_size, col_ids);
  
  double time_after_compression = MPI_Wtime();


  //
  // Reconstruct Matrix
  //

  DeviceMatrix<Scalar> X_reconstructed = X_compressed.reconstruct();


  //
  // Statistics Output
  //

  if (po_vm.count("statistics")) {

    // Title & upper border
    if (!my_rank) std::cout << std::endl << std::setfill('-') << std::setw(80) << '-' << std::endl;
    if (!my_rank) std::cout << " Statistics" << std::endl;
    if (!my_rank) std::cout << std::setw(80) << '-' << std::setfill(' ') << std::endl << std::endl;

    // Timing information
    double time_for_solve = time_after_compression -  time_after_reading_data;
    double time_for_input = time_after_reading_data - time_at_start;
    double max_time_for_solve, max_time_for_input;
    MPI_Allreduce( &time_for_solve, &max_time_for_solve, 1, mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &time_for_input, &max_time_for_input, 1, mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD );
    if (!my_rank) std::cout << " Maximum time for input: " << max_time_for_input << std::endl;
    if (!my_rank) std::cout << " Maximum time for solve: " << max_time_for_solve << std::endl;
    if (!my_rank) std::cout << std::endl;

    // Compression ratio
    if (!my_rank) std::cout << " Compression ratio: " <<
        X_compressed.compressed_size / (double) X_compressed.original_size
        << std::endl << std::endl;

    // Matrix statistics
    std::vector<int> row_start, row_count, col_start, col_count;
    std::vector<Scalar> variable_mean, variable_max;
    netcdf_interface.get_variable_ranges(row_start, row_count, col_start, col_count);
    netcdf_interface.get_variable_transformation(variable_mean, variable_max);
    print_statistics(X, X_reconstructed, variables, variable_mean,
        variable_max, row_start, row_count, col_start, col_count);

    // Lower border
    if (!my_rank) std::cout << std::setfill('-') << std::setw(80) << '-' <<
        std::setfill(' ') << std::endl << std::endl;
  }


  //
  // Write Reconstructed Data to File
  //

  netcdf_interface.write_matrix(X_reconstructed, filename_out, append);


  //
  //  Terminate MPI and Quit Program.
  //

  netcdf_interface.close();
  MPI_Finalize();
  return 0;
}
