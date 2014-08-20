/** \file NetCDFInterface.h
 *
 *  This file defines the templated class NetCDFInterfacetype.
 *
 *  \copyright Copyright (c) 2014,
 *             Universita della Svizzera italiana (USI) &
 *             Centro Svizzero di Calcolo Scientifico (CSCS).
 *             All rights reserved.
 *             This software may be modified and distributed under the terms
 *             of the BSD license. See the LICENSE file for details.
 *
 *  \author Will Sawyer (CSCS)
 *  \author Ben Cumming (CSCS)
 *  \author Manuel Schmid (CSCS)
 */

#pragma once
#include <iostream> // std::cout, std::endl
#include <string>   // std::string
#include <vector>   // std::vector
#include <set>      // std::set
#include <map>      // std::map
#include <mpi.h>    // MPI_Comm_size, MPI_Comm_rank, MPI_Barrier
#include <netcdf_par.h>
#include <netcdf.h>
#include "matrices.h"

/// Macro to print a friendly error message for NetCDF errors.
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(1);}

/**
 * This class is responsible for the mapping of multi-dimensional, multi-
 * variable data from the NetCDF input file to a 2D matrix that is used for
 * compression. The constructor sets up a range of member variables that
 * describe the mapping. These are then used for constructing a matrix with
 * the construct_matrix() function or for writing a matrix back to a file with
 * the write_matrix() function. In addition, there are some functions for
 * getting information about the mapping that is needed by other (external)
 * functions.
 */
template<class Scalar>
class NetCDFInterface
{

public:

  /// Used for specifying the direction in which the variables are stacked.
  enum Stacking {HORIZONTAL, VERTICAL};

  /**
   * \brief Set up mapping of data from NetCDF file to matrix.
   *
   * The constructor sets up the information needed for mapping the NetCDF
   * data to a matrix. This mainly includes start_ and count_, set up in
   * select_data_ranges(), and row_map_ and col_map, set up in
   * set_up_mapping(). The other variables are mainly kept for convenience, as
   * they are used in several functions. The only member variables that aren't
   * set up here in the constructor are variable_mean_ and variable_max_, used
   * for normalizing each variable so they have comparable ranges in the
   * matrix. As these variables depend on the actual data rather than just its
   * structute, they are set up when the data is being read in
   * construct_matrix().
   *
   * It sets up the member variables #filename_ and #stacking_.
   *
   * \see initialize_data(), select_data_ranges(), set_up_mapping(),
   *      calculate_matrix_dimensions()
   *
   * \param[in] filename        The full or relative path to the input file.
   * \param[in] variables       A list of the variables that will be read.
   * \param[in] compressed_dims A list of the dimensions that will be placed
   *                            along the columns of the matrix.
   * \param[in] indexed_dims    A list of the dimensions along which only a
   *                            single entry is selected, specified in the
   *                            format DIMENSION_NAME=SELECTED_INDEX, e.g.
   *                            time=0.
   * \param[in] stacking        The direction along which variables are
   *                            stacked in the matrix if several variables are
   *                            selected. HORIZONTAL means that they are added
   *                            as columns, VERTICAL means that they are added
   *                            as rows. If a dimension is missing for some
   *                            variables (e.g. levels for 2D variables), the
   *                            variables should be stacked along the
   *                            direction where this dimension is placed, i.e.
   *                            HORIZONTAL for a distributed dimension and
   *                            VERTICAL for a compressed dimension.
   */
  NetCDFInterface(const std::string filename,
                  const std::vector<std::string> variables,
                  const std::vector<std::string> compressed_dims,
                  const std::vector<std::string> indexed_dims,
                  const Stacking stacking = VERTICAL) {

    // save internal variables
    filename_ = filename;
    stacking_ = stacking;

    initialize_data(variables, compressed_dims, indexed_dims);
    select_data_ranges();
    set_up_mapping();
    calculate_matrix_dimensions();
  }

  /**
   * \brief Create 2D matrix with data read from NetCDF file.
   *
   * This function reads the data from the NetCDF file and restructures it
   * into a 2D matrix that is returned. First, the data assigned to the
   * current process is read from the file. Then, each variable is normalized
   * by subtracting the mean and dividing by the maximum so all values are
   * between -1 and +1. Finally, the data is restructured into a 2D matrix.
   *
   * \see read_data(), normalize_data(), restructure_data()
   *
   * \return  2D matrix with all data assigned to the current process.
   */
  DeviceMatrix<Scalar> construct_matrix() {

    std::vector< HostVector<Scalar> > data = read_data();
    normalize_data(data);
    HostMatrix<Scalar> restructured_data = restructure_data(data);
    DeviceMatrix<Scalar> output = restructured_data; // copy to device

    return output;
  }

  /**
   * \brief Get vectors specifying where the data for each variable is stored
   *        within the matrix.
   *
   * If we want to access data in the matrix belonging to just one variable,
   * we have to know where it is stored. This function creates four vectors
   * with the first row/column for each variable as well as the number of
   * rows/columns. The vectors passed as arguments can be empty, as they are
   * overwritten.
   *
   * \param[out]  row_start   Index of first row for each variable.
   * \param[out]  row_count   Number of rows for each variable.
   * \param[out]  col_start   Index of first column for each variable.
   * \param[out]  col_count   Number of columns for each variable.
   */
  void get_variable_ranges(
      std::vector<int> &row_start,
      std::vector<int> &row_count,
      std::vector<int> &col_start,
      std::vector<int> &col_count) {

    // setup new vectors
    row_start = std::vector<int>(n_variables_);
    row_count = std::vector<int>(n_variables_);
    col_start = std::vector<int>(n_variables_);
    col_count = std::vector<int>(n_variables_);

    // fill in new vectors
    int row = 0;
    int col = 0;
    for (int v = 0; v < n_variables_; v++) {
      row_start[v] = row;
      row_count[v] = variable_rows_[v];
      col_start[v] = col;
      col_count[v] = variable_cols_[v];

      if (stacking_ == HORIZONTAL) {
        col += variable_cols_[v];
      } else {
        row += variable_rows_[v];
      }
    }
  }

  /**
   * \brief Get vectors with values used for transforming the data for each
   *        variable.
   *
   * This function returns vectors with the values that have been used for
   * normalizing the data for each variable. The normalization can be reversed
   * by multiplying a value with variable_max[v] and adding variable_mean[v],
   * i.e. value_original = value_transformed * variable_max[v] +
   * variable_mean[v]. The vectors that are passed as arguments can be empty
   * as they are overwritten by the function.
   *
   * \param[out]  variable_mean   The mean values of the original variables.
   * \param[out]  variable_max    The maximum absolute values of the variables
   *                              the mean has been subtracted.
   *
   * \warning The variable transformation isn't set up until the data is read
   *          from the NetCDF file. Therefore, this function shouldn't be
   *          called before construct_matrix() has been called.
   */
  void get_variable_transformation(std::vector<Scalar> &variable_mean,
                                   std::vector<Scalar> &variable_max) {
    variable_mean = variable_mean_;
    variable_max  = variable_max_;
  }

  /**
   * \brief Get a global ID for each matrix column that has been assigned to
   *        the current process, independent from the number of processes.
   *
   * This function returns a vector with an ID for each column of the data
   * matrix. The ID corresponds to the index that the column would have if
   * there was just a single process with all the data. This allows for
   * assigning initial clusters in a way that is independent from the number
   * of processes.
   *
   * \return  Vector with a global ID for each matrix column.
   */
  std::vector<int> get_column_ids() {

    // setup new vector
    std::vector<int> col_ids(n_cols_);

    int counter = 0;
    int offset = 0;

    for (int v = 0; v < n_variables_; v++) {

      int distributed_dims = distributed_dims_length_[v].size();

      // set up dimension_indices and calculate # of columns of variable
      int variable_cols = 1;
      std::vector<int> dimension_indices(distributed_dims);
      for (int d = 0; d < distributed_dims; d++) {
        dimension_indices[d] = distributed_dims_start_[v][d];
        variable_cols *= distributed_dims_length_[v][d];
      }

      // calculate all indices for the current variable
      for (int i = 0; i < variable_cols_[v]; i++) {

        int id = 0;
        int factor = 1; // TODO: better name

        for (int d = distributed_dims - 1; d >= 0; d--) { // reverse iteration
          if (dimension_indices[d] == distributed_dims_start_[v][d]
              + distributed_dims_count_[v][d] && d > 0) {
            dimension_indices[d] = distributed_dims_start_[v][d];
            dimension_indices[d-1]++;
          }
          id += dimension_indices[d] * factor;
          factor *= distributed_dims_length_[v][d];
        }

        col_ids[counter] = id + offset;
        counter++;
        dimension_indices.back()++; // increment last dimension
      }

      // increment offset for next variable
      offset += variable_cols;

      // for vertical stacking, we stop after one variable
      if (stacking_ == VERTICAL) {
        break;
      }
    }
    return col_ids;
  }

  /**
   * \brief Write matrix to NetCDF file with the same variables and dimensions
   *        as the input file.
   *
   * This function uses the same mapping used for reading the input file to
   * write a matrix back to a NetCDF file. The data is first recomposed into
   * separate vectors for each variable with the correct order of the values.
   * Then, the normalization that has been done when reading the data is
   * reversed. Finally, a new file with the same dimensions and variables is
   * created and the data is written to this file.
   *
   * \see recompose_data(), denormalize_data(), setup_output_file(),
   *      write_data()
   *
   * \param[in] input         The matrix that is to be written to the file.
   * \param[in] filename_out  The absolute or relative path to the file where
   *                          the data is written. If this is an empty string,
   *                          the data is written to
   *                          INPUT_FILE_reconstructed.nc4.
   *                          (default: "")
   * \param[in] append        Whether the data should be appended to an
   *                          existing file instead of overwriting it.
   *                          (default: false)
   */
  void write_matrix(DeviceMatrix<Scalar> input, std::string filename_out = "",
      bool append = false) {

    if (filename_out.empty()) filename_out =
        filename_.substr(0,filename_.length()-4) + "_reconstructed.nc4";
    if (!my_rank_) std::cout << "Writing data to file " << filename_out << std::endl;

    HostMatrix<Scalar> data = input; // copy to host
    std::vector< HostVector<Scalar> > recomposed_data = recompose_data(data);
    denormalize_data(recomposed_data);
    // only the first process creates the new file
    if (!my_rank_) setup_output_file(filename_out, append);
    // only continue after file has been created
    MPI_Barrier(MPI_COMM_WORLD);
    write_data(filename_out, recomposed_data);
  }

  /**
   * \brief Close the input NetCDF file.
   *
   * The input NetCDF file is opened in the constructor and kept open while
   * the object is in use as the file is accessed at various points. This
   * function should be called when the object is no longer needed and before
   * MPI_Finalize(), which produces errors otherwise.
   */
  void close() {
    if ((r_ = nc_close(netcdf_id_))) ERR(r_);
    if (!my_rank_) std::cout << "NetCDF file closed" << std::endl;
  }


private:

  int r_; ///< Used for NetCDF error handling only.

  /**
   * \name initialized in constructor:
   */
  ///@{
  /// \brief Full or relative path to input file.
  std::string filename_;
  /// \brief Direction along which variables are stacked.
  Stacking stacking_;
  ///@}

  /**
   * \name initialized in initialize_data():
   */
  ///@{
  /// \brief Used when data can't be distributed evenly between processes.
  int process_getting_more_data_;
  /// \brief Total number of MPI processes.
  int mpi_processes_;
  /// \brief Rank of current MPI process, used for assigning data and limiting
  ///        console output to one process.
  int my_rank_;
  /// \brief ID for the input NetCDF file, used by the NetCDF library.
  int netcdf_id_;
  /// \brief Number of variables.
  int n_variables_;
  /// \brief IDs for all selected variables, used by the NetCDF library.
  std::vector<int> variable_ids_;
  /// \brief The names of all dimensions that will be placed along columns of
  ///        the matrix.
  std::set<std::string> compressed_dims_;
  /// \brief The names and selected indices for all dimensions of which only a
  ///        single index is to be read from the file.
  std::map<std::string, int> indexed_dims_;
  ///@}

  /**
   * \name initialized in select_data_ranges():
   */
  ///@{
  /// \brief The number of dimensions, for each variable.
  std::vector<int> variable_dims_;
  /// \brief Whether the current process has no data, for each variable.
  std::vector<bool> variable_is_empty_;
  /// \brief The first index of the data that is read from the file, for each
  ///        variable & dimension.
  std::vector< std::vector<size_t> > start_;
  /// \brief The number of values that are read from the file, for each
  ///        variable & dimension.
  std::vector< std::vector<size_t> > count_;
  /// \brief The first index of the data that is read from the file, for each
  ///        variable & distributed dimension. This is stored separately for
  ///        convenience and is used for assigning global column IDs.
  std::vector< std::vector<size_t> > distributed_dims_start_;
  /// \brief The number of values that are read from the files, for each
  ///        variable & distributed dimension. This is stored separately for
  ///        convenience and is used for assigning global column IDs.
  std::vector< std::vector<size_t> > distributed_dims_count_;
  /// \brief The total number of values, for each variable & distributed
  ///        dimension. This is used for assigning global column IDs.
  std::vector< std::vector<size_t> > distributed_dims_length_;
  ///@}

  /**
   * \name initialized in set_up_mapping():
   */
  ///@{
  /// \brief The number of rows in the output matrix, for each variable.
  std::vector<int> variable_rows_;
  /// \brief The number fo colmns in the output matrix, for each variable.
  std::vector<int> variable_cols_;
  /// \brief The mapping for rows, for each variable & dimension. (see
  ///        set_up_mapping() and restructure_data() for details)
  std::vector< std::vector<int> > row_map_;
  /// \brief The mapping for columns, for each variable & dimension. (see
  ///        set_up_mapping() and restructure_data() for details)
  std::vector< std::vector<int> > col_map_;
  ///@}

  /**
   * \name initialized in calculate_matrix_dimensions():
   */
  ///@{
  int n_rows_;  ///< \brief Number of rows of matrix.
  int n_cols_;  ///< \brief Number of columns of matrix.
  ///@}

  /**
   * \name initialized in normalize_data():
   */
  ///@{
  /// \brief Mean value of the original data, for each variable.
  std::vector<Scalar> variable_mean_;
  /// \brief Maximum absolute value after subtracting the mean, for each
  //         variable.
  std::vector<Scalar> variable_max_;
  ///@}

  /**
   * \brief Overloaded wrapper around nc_get_vara_double.
   */
  int nc_get_vara(int ncid, int varid, const size_t start[], const size_t
      count[], double *dp) {
    int retval = nc_get_vara_double(ncid, varid, start, count, dp);
    return retval;
  }

  /**
   * \brief Overloaded wrapper around nc_get_vara_float.
   */
  int nc_get_vara(int ncid, int varid, const size_t start[], const size_t
      count[], float *fp) {
    int retval = nc_get_vara_float(ncid, varid, start, count, fp);
    return retval;
  }

  /**
   * \brief Overloaded wrapper around nc_put_vara_double.
   */
  int nc_put_vara(int ncid, int varid, const size_t start[], const size_t
      count[], const double *fp) {
    int retval = nc_put_vara_double(ncid, varid, start, count, fp);
    return retval;
  }

  /**
   * \brief Overloaded wrapper around nc_put_vara_float.
   */
  int nc_put_vara(int ncid, int varid, const size_t start[], const size_t
      count[], const float *fp) {
    int retval = nc_put_vara_float(ncid, varid, start, count, fp);
    return retval;
  }

  /**
   * Helper function to increment process_getting_more_data, restarting at
   * zero if the last process has been reached.
   */
  void increment_process_getting_more_data() {
    process_getting_more_data_++;
    if (process_getting_more_data_ == mpi_processes_) {
      process_getting_more_data_ = 0;
    }
  }

  /**
   * This function sets up some of the private variables. It reads MPI process
   * information, opens the input file, reads variable ids, and parses the
   * arguments for compressed and indexed dimensions.
   *
   * It sets up the member variables #process_getting_more_data_,
   * #mpi_processes_, #my_rank_, #netcdf_id_, #n_variables_, #variable_ids_,
   * #compressed_dims_, and #indexed_dims_.
   *
   * \param[in] variables         List of all variables that should be read.
   * \param[in] compressed_dims   List of all dimensions that should be put
   *                              along the columns of the matrix.
   * \param[in] indexed_dims      List of all dimensions for which only a
   *                              single index should be selected, specified
   *                              in the format DIMENSION=SELECTED_INDEX (e.g.
   *                              time=0).
   */
  void initialize_data(const std::vector<std::string> &variables,
      const std::vector<std::string> &compressed_dims,
      const std::vector<std::string> &indexed_dims) {

    // first process gets more data first if needed
    process_getting_more_data_ = 0;

    // initialize mpi
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

    // NetCDF: open file for read-only access
    if ((r_ = nc_open_par(filename_.c_str(), NC_NOWRITE|NC_MPIIO,
            MPI_COMM_WORLD, MPI_INFO_NULL, &netcdf_id_))) ERR(r_);

    n_variables_ = variables.size();
    variable_ids_ = std::vector<int>(n_variables_);
    for (int v = 0; v < n_variables_; v++) {
      // NetCDF: get variable ids
      if ((r_ = nc_inq_varid(netcdf_id_, variables[v].c_str(),
              &variable_ids_[v]))) ERR(r_);
    }
    
    compressed_dims_ = std::set<std::string>(compressed_dims.begin(),
        compressed_dims.end());

    for (int i = 0; i < indexed_dims.size(); i++) {
      size_t separator_index = indexed_dims[i].find("=");
      if (separator_index == std::string::npos) {
        if (!my_rank_) std::cout << "Error: Invalid separator for indexed dimension "
          << indexed_dims[i] << std::endl;
        exit(1);
      }
      std::string dim_name = indexed_dims[i].substr(0, separator_index);
      indexed_dims_[dim_name] = std::atoi(indexed_dims[i].substr(separator_index+1).c_str());
    }

  }

  /**
   * This function sets up the information about the data ranges that the
   * current process should read from the file. This mainly involves setting
   * up start_ and count_ with the first index and the number of entries that
   * will be read for each variable and dimension. The distributed dimensions
   * are only collected in this function and the entries in start_ and count_
   * are later filled in by calling calculate_distributed_dims_data_range().
   *
   * It sets up the member variables #variable_dims_, #variable_is_empty_,
   * #start_, #count_, #distributed_dims_start_, #distributed_dims_count_,
   * and #distributed_dims_length_.
   */
  void select_data_ranges() {
    /* This function sets up the internal variables start_ and count_. */
    
    variable_dims_     = std::vector<int>(n_variables_);
    variable_is_empty_ = std::vector<bool>(n_variables_, false);
    
    start_  = std::vector< std::vector<size_t> >(n_variables_);
    count_  = std::vector< std::vector<size_t> >(n_variables_);
    distributed_dims_start_  = std::vector< std::vector<size_t> >(n_variables_);
    distributed_dims_count_  = std::vector< std::vector<size_t> >(n_variables_);
    distributed_dims_length_ = std::vector< std::vector<size_t> >(n_variables_);

    for (int v = 0; v < n_variables_; v++) {

      // NetCDF: get the number of dimensions for the variable
      if ((r_ = nc_inq_varndims(netcdf_id_, variable_ids_[v],
              &variable_dims_[v]))) ERR(r_);

      start_[v] = std::vector<size_t>(variable_dims_[v]);
      count_[v] = std::vector<size_t>(variable_dims_[v]);

      std::vector<int> distributed_dims;
      distributed_dims_length_[v] = std::vector<size_t>();

      // NetCDF: get the ids of all dimensions for the variable
      // we use the dim_ids vector as an array here
      std::vector<int> dim_ids(variable_dims_[v]);
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);

      if (!my_rank_) std::cout << "Dimensions for variable " << v << ":"
          << std::endl;

      for (int d = 0; d < variable_dims_[v]; d++) {

        // NetCDF: get dimension name & length
        size_t dim_length;
        char dim_name[NC_MAX_NAME+1];
        if ((r_ = nc_inq_dim(netcdf_id_, dim_ids[d], dim_name,
                &dim_length))) ERR(r_);

        if (indexed_dims_.find(dim_name) != indexed_dims_.end()) {
          // the dimension is indexed
          start_[v][d] = indexed_dims_[dim_name];
          count_[v][d] = 1;
          if (!my_rank_) std::cout << "  " << dim_name
              << " (indexed, selecting index " << indexed_dims_[dim_name]
              << ")" << std::endl;
        }

        else if (compressed_dims_.find(dim_name) != compressed_dims_.end()) {
          // the dimension is compressed
          start_[v][d] = 0;
          count_[v][d] = dim_length;
          if (!my_rank_) std::cout << "  " << dim_name << " (compressed, "
              << dim_length << " entries)" << std::endl;
        }
        
        else {
          // the dimension is distributed
          // we only collect the distributed dimensions and their length in
          // this loop and treat them separately afterwards
          distributed_dims.push_back(d);
          distributed_dims_length_[v].push_back(dim_length);
          if (!my_rank_) std::cout << "  " << dim_name
              << " (distributed, " << dim_length << " entries)" << std::endl;
        }
      }

      // this adds in the correct data ranges for the distributed dimensions
      if (distributed_dims.size()) {
        calculate_distributed_dims_data_range(v, distributed_dims);
      } else {
        // for variables that have no dimensions in the distributed direction
        // (this should only happen if they are stacked horizontally),
        // all processes get the data (there is no count in the distributed
        // direction as the variable doesn't have a dimension in this
        // direction). we therefore have to assign it to one of the processes
        // here. we can then skip this variable when we load the data into the
        // output matrix in restructure_data().
        assert(stacking_ == HORIZONTAL);
        if (process_getting_more_data_ != my_rank_) variable_is_empty_[v] = true;
        increment_process_getting_more_data();
      }
    }
  }

  /**
   * This function sets up the start and count for the distributed dimensions
   * of a variable. It assumes that the length of each dimension has already
   * been stored in distributed_dims_length_[variable][i] for all dimensions
   * distributed_dims[i]. This is done in the calling function
   * select_data_ranges(). We use get_process_distribution() to decide how
   * many processes there are along each distributed dimension.
   *
   * \param[in] variable          The ID of the variable that will be treated.
   * \param[in] distributed_dims  The indices of the dimensions (e.g. in
   *                              start_ and count_) that are distributed.
   */
  void calculate_distributed_dims_data_range(const int variable,
      const std::vector<int> &distributed_dims) {

    std::vector<int> process_distribution =
        get_process_distribution(distributed_dims.size());

    int n_dims = distributed_dims.size();
    distributed_dims_start_[variable] = std::vector<size_t>(n_dims);
    distributed_dims_count_[variable] = std::vector<size_t>(n_dims);

    int r = my_rank_;        // needed for calculating index along dimension
    int p = mpi_processes_;  // needed for calculating index along dimension

    for (int i = 0; i < n_dims; i++) {

      // In order to split the data between the processes, each process needs
      // to be assigned a unique index along each distributed dimension. This
      // index is then used to calculate the start and end index for the data.
      //
      // We assign an index along each dimension based on the rank of the
      // current process. For three distributed dimensions, the first process
      // would get 0-0-0, the second gets 0-0-1 and so on. This is similar to
      // the conversion of a number to a different base. In fact, if there are
      // two processes along each dimension, it corresponds to the binary
      // representation of the rank.
      //
      // To calculate this, we start with the first 'digit'. For decimal
      // numbers (corresponding to 10 processes along each dimension),
      // this would be the rank divided by 10^n, where n is the number of
      // following dimensions. In our case, this is the rank divided by the
      // product of the number of processes along all following dimensions.
      // For the second 'digit', we proceed with the remainder of this
      // division.
      //
      // In the following code, we use p as the product of the number of
      // processes along all following dimensions. As the product of for all
      // dimensions is given by the overall number of processes, we start with
      // this and continue dividing by the number of processes along the
      // current dimension. This leaves us with the product for all remaining
      // dimensions.
      p /= process_distribution[i];
      int dim_index = r / p;
      r %= p;

      int dim_length = distributed_dims_length_[variable][i];
      int size_along_dim = dim_length / process_distribution[i];

      start_[variable][distributed_dims[i]] = dim_index * size_along_dim;
      count_[variable][distributed_dims[i]] = size_along_dim;

      // if we are the last process along a dimension, add the remainder
      if (dim_index == process_distribution[i] - 1) {
        count_[variable][distributed_dims[i]] += dim_length
          % process_distribution[i];
      }

      // we keep separate lists with the start/count of the distributed
      // dimensions only
      distributed_dims_start_[variable][i] = start_[variable][distributed_dims[i]];
      distributed_dims_count_[variable][i] = count_[variable][distributed_dims[i]];

    }
  }

  /**
   * This function builds a list with the number of processes along each
   * distributed dimension. Note: We assume the number of MPI processes to be
   * a power of 2.
   *
   * \param[in] n_distributed_dims  The number of distributed dimensions.
   * \return    Vector with the number of processes along each distributed
   *            dimension.
   */
  std::vector<int> get_process_distribution(const int n_distributed_dims) {

    int p = mpi_processes_;
    int i = 0;
    std::vector<int> process_distribution(n_distributed_dims, 1);

    while (p > 1) {
      if (p % 2 != 0) {
        std::cout << "Error: The number of processes must be a power of 2" << std::endl;
        exit(1);
      }
      p /= 2;
      process_distribution[i] *= 2;
      // restart at the beginning if the last dimension is reached
      i = (i + 1) % process_distribution.size();
    }
    return process_distribution;
  }

  /**
   * This function goes through all variables and dimensions to build row_map_
   * and col_map_. These contain the number of rows and columns that will be
   * between two consecutive elements along a dimension when they are written
   * to the matrix. This makes it fairly easy to calculate the row and column
   * for each value when the data is written to a matrix in the function
   * restructure_data().
   *
   * It sets up the member variables #row_map_, #col_map_, #variable_rows_,
   * and #variable_cols_.
   *
   * \remark As variables can have a different number of dimensions (e.g. 2D
   *         and 3D variables), we do this for every variable. We go through
   *         all dimensions starting with the last one as this is the fastest
   *         changing dimension in the NetCDF data. For an indexed dimension,
   *         the values don't matter as they will be multiplied by 0 in
   *         restructure_data(). For the first compressed dimension, the
   *         elements will be directly next to each other along the columns
   *         (inter-element distance = 1) and they will be placed in the same
   *         row (inter-element distance = 0). We then multiply the column
   *         inter-element distance by the length of the dimension as this is
   *         how far apart the elements of the next compressed dimension will
   *         be placed in the matrix. For the distributed dimensions, this is
   *         identical, except that rows and columns are switched.
   */
  void set_up_mapping() {

    row_map_ = std::vector< std::vector<int> >(n_variables_);
    col_map_ = std::vector< std::vector<int> >(n_variables_);

    variable_rows_ = std::vector<int>(n_variables_);
    variable_cols_ = std::vector<int>(n_variables_);

    for (int v = 0; v < n_variables_; v++) {

      row_map_[v] = std::vector<int>(variable_dims_[v]);
      col_map_[v] = std::vector<int>(variable_dims_[v]);

      int row_interelement_distance = 1;
      int col_interelement_distance = 1;

      // NetCDF: get the ids of all dimensions for the variable
      // we use the dim_ids vector as an array here
      std::vector<int> dim_ids(variable_dims_[v]);
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);

      // last dimensions are the fastest changing
      for (int d = variable_dims_[v] - 1; d >= 0; d--) {

        // NetCDF: get dimension name & length
        char dim_name[NC_MAX_NAME+1];
        if ((r_ = nc_inq_dimname(netcdf_id_, dim_ids[d], dim_name))) ERR(r_);

        if (indexed_dims_.find(dim_name) != indexed_dims_.end()) {
          // the dimension is indexed
          // in this case, the values don't matter as they are multiplied by 0
          row_map_[v][d] = 0;
          col_map_[v][d] = 0;
        }

        else if (compressed_dims_.find(dim_name) != compressed_dims_.end()) {
          // the dimension is compressed
          row_map_[v][d] = row_interelement_distance;
          col_map_[v][d] = 0; // all values are in the same column
          row_interelement_distance *= count_[v][d];
        }
        
        else {
          // the dimension is distributed
          row_map_[v][d] = 0; // all values are in the same row
          col_map_[v][d] = col_interelement_distance;
          col_interelement_distance *= count_[v][d];
        }
      }
      variable_rows_[v] = row_interelement_distance;
      variable_cols_[v] = col_interelement_distance;
    }
  }

  /**
   * This function sums up the rows or columns of all variables to calculate
   * the dimensions of the matrix, taking the stacking direction into account.
   *
   * It sets up the member variables #n_rows_ and #n_cols_.
   */
  void calculate_matrix_dimensions() {
    n_rows_ = variable_rows_[0];
    n_cols_ = variable_cols_[0];
    for (int v = 1; v < n_variables_; v++) {
      if (stacking_ == HORIZONTAL) {
        assert(n_rows_ == variable_rows_[v]);
        if (!variable_is_empty_[v]) n_cols_ += variable_cols_[v];
      } else {
        assert(n_cols_ == variable_cols_[v]);
        n_rows_ += variable_rows_[v];
      }
    }
  }

  /**
   * This function reads the data for all variables in the order it appears in
   * the NetCDF file. It uses #start_ and #count_ to select the data assigned
   * to the current process.
   *
   * \returns   Vector with one data array (HostVector) per variable.
   */
  std::vector< HostVector<Scalar> > read_data() {

    std::vector< HostVector<Scalar> > output(n_variables_);

    for (int v = 0; v < n_variables_; v++) {
      output[v] = HostVector<Scalar>(variable_rows_[v] * variable_cols_[v]);
      // NetCDF: get data array for variable
      if ((r_ = nc_get_vara(netcdf_id_, variable_ids_[v], &(start_[v][0]),
              &(count_[v][0]), GET_POINTER(output[v])))) ERR(r_);
    }

    if (!my_rank_) std::cout << "Data read from NetCDF file" << std::endl;
    return output;
  }

  /**
   * This function normalizes the data for each variable to a value between -1
   * and +1. It first calculates the global mean value and subtracts it from
   * all values. It then finds the maximum absolute value of the resulting
   * data and divides all values by it.
   *
   * It initializes the member variables #variable_mean_ and #variable_max_.
   *
   * \param[in,out] data  Vector with data for each variable as returned by
   *                      read_data().
   */
  void normalize_data(std::vector< HostVector<Scalar> > &data) {

    // initialize vectors
    variable_mean_ = std::vector<Scalar>(n_variables_);
    variable_max_  = std::vector<Scalar>(n_variables_);

    for (int v = 0; v < n_variables_; v++) {

      // find mean
      int local_count = data[v].size();
      int global_count;
      MPI_Allreduce(&local_count, &global_count, 1,
          MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#if defined(USE_EIGEN)
      Scalar local_sum = data[v].sum();
#elif defined(USE_MINLIN)
      Scalar local_sum = sum(data[v]);
#endif
      MPI_Allreduce(&local_sum, &variable_mean_[v], 1,
          mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
      variable_mean_[v] /= global_count;

      // subtract mean
#if defined(USE_EIGEN)
      data[v] = data[v].array() - variable_mean_[v];
#elif defined(USE_MINLIN)
      data[v] -= variable_mean_[v];
#endif

      // find max abs value
#if defined(USE_EIGEN)
      Scalar local_max = data[v].cwiseAbs().maxCoeff();
#elif defined(USE_MINLIN)
      Scalar local_max = max(abs(data[v]));
#endif
      MPI_Allreduce(&local_max, &variable_max_[v], 1,
          mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);

      // divide by max abs value
      data[v] /= variable_max_[v];
    }

  }

  /**
   * This function reverses the transformation done by normalize_data().
   *
   * \param[in,out] data  Vector with data for each variable as read by
   *                      write_data().
   */
  void denormalize_data(std::vector< HostVector<Scalar> > &data) {
    for (int v = 0; v < n_variables_; v++) {
#if defined(USE_EIGEN)
      data[v] = variable_mean_[v] + data[v].array() * variable_max_[v];
#elif defined(USE_MINLIN)
      data[v] = variable_mean_[v] + data[v] * variable_max_[v];
#endif
    }
  }

  /**
   * This function takes the values as they have been read from the NetCDF
   * file and writes them to the output matrix in the correct order. It uses
   * the variables #row_map_ and #col_map_ that have been set up in
   * set_up_mapping().
   *
   * \param[in] data  Vector with data for each variable as returned by
   *                  read_data().
   * \return          Matrix with restructured data.
   *
   * \remark  We use the vector dim_indicies to keep track of the index along
   *          each dimension of the current value. As we have saved the inter-
   *          element distance in #row_map_ and #col_map_, we can simply
   *          multiply these with the indices along the dimension to get the
   *          row and column index of the position in the output matrix. In
   *          addition, we use an offset to account for other variables that
   *          have already been written to the matrix.
   */
  HostMatrix<Scalar> restructure_data(std::vector< HostVector<Scalar> > &data) {

    HostMatrix<Scalar> output(n_rows_, n_cols_);

    int row_offset = 0;
    int col_offset = 0;
    for (int v = 0; v < n_variables_; v++) {

      // variables with no dimensions in distributed direction are detected
      // and set at the end of select_data_ranges()
      if (variable_is_empty_[v]) {
        continue;
      }

      std::vector<int> dim_indicies(variable_dims_[v], 0);
      for (int i = 0; i < data[v].size(); i++) {

        int output_row = row_offset;
        int output_col = col_offset;
        for (int d = variable_dims_[v] - 1; d >= 0; d--) {
          // increase indices of next dimension and reset current index
          // if a dimension has reached the maximum
          if (dim_indicies[d] == count_[v][d]) {
            dim_indicies[d] = 0;
            dim_indicies[d-1] += 1;
          }
          output_row += dim_indicies[d] * row_map_[v][d];
          output_col += dim_indicies[d] * col_map_[v][d];
        }

        // copy the data to the matrix, normalizing with the mean & max values
        output(output_row, output_col) = data[v](i);

        // increase index along last dimension for next iteration
        dim_indicies[variable_dims_[v]-1]++;
      }

      if (stacking_ == HORIZONTAL) {
        col_offset += variable_cols_[v];
      } else {
        row_offset += variable_rows_[v];
      }
    }
    if (!my_rank_) std::cout << "Data restructured into a 2D matrix" << std::endl;
    return output;
  }

  /**
   * This function restructures the values from a matrix to separate arrays
   * for each variable that have the correct order to be written to a NetCDF
   * file. It basically reverses the restructuring done in restructure_data().
   *
   * \param[in] data  Matrix with restructured data.
   * \return          Vector with data for each variable as read by
   *                  write_data().
   */
  std::vector< HostVector<Scalar> > recompose_data(HostMatrix<Scalar> data) {

    std::vector< HostVector<Scalar> > output(n_variables_);
    
    int row_offset = 0;
    int col_offset = 0;
    for (int v = 0; v < n_variables_; v++) {

      // variables with no dimensions in distributed direction are detected
      // and set at the end of select_data_ranges()
      if (variable_is_empty_[v]) {
        continue;
      }

      output[v] = HostVector<Scalar>(variable_rows_[v] * variable_cols_[v]);

      std::vector<int> dim_indicies(variable_dims_[v], 0);
      for (int i = 0; i < output[v].size(); i++) {

        int data_row = row_offset;
        int data_col = col_offset;
        for (int d = variable_dims_[v] - 1; d >= 0; d--) {
          // increase indices of next dimension and reset current index
          // if a dimension has reached the maximum
          if (dim_indicies[d] == count_[v][d]) {
            dim_indicies[d] = 0;
            dim_indicies[d-1] += 1;
          }
          data_row += dim_indicies[d] * row_map_[v][d];
          data_col += dim_indicies[d] * col_map_[v][d];
        }
        output[v](i) = data(data_row, data_col);

        // increase index along last dimension for next iteration
        dim_indicies[variable_dims_[v]-1]++;
      }

      if (stacking_ == HORIZONTAL) {
        col_offset += variable_cols_[v];
      } else {
        row_offset += variable_rows_[v];
      }
    }

    return output;
  }

  /**
   * This function prepares the output file to make sure it has the dimensions
   * and variable definitions for all variables that will be written to it. It
   * either creates a new file or appends to an existing file. The dimensions
   * and variables are created by the functions create_all_dimensions() and
   * create_all_variables() respectively.
   *
   * \param[in] filename_out  Full or relative path to the output NetCDF file.
   * \param[in] append        Whether the data should be appended to the file
   *                          instead of overwriting it.
   *
   * \warning This function doesn't use parallel file access and should only
   *          be called by one process.
   */
  void setup_output_file(std::string filename_out, bool append) {
    /* This creates a new file 'filename' with all the dimensions and
     * variables that have been read from the original input file. */

    // NetCDF: create new file, overwriting if it already exists
    int netcdf_id_out;
    if (append) {
      // NetCDF: open file in write mode
      if ((r_ = nc_open(filename_out.c_str(), NC_WRITE,
              &netcdf_id_out))) ERR(r_);
      // NetCDF: place file in define mode
      if ((r_ = nc_redef(netcdf_id_out))) ERR(r_);
    } else {
      // NetCDF: create new file, overwriting if it already exists
      if ((r_ = nc_create(filename_out.c_str(), NC_CLOBBER|NC_NETCDF4,
              &netcdf_id_out))) ERR(r_);
    }

    // create dimensions & variables
    std::map<int, int> dimension_map = create_all_dimensions(netcdf_id_out);
    create_all_variables(netcdf_id_out, dimension_map);

    // NetCDF: end define mode and close output file
    if ((r_ = nc_enddef(netcdf_id_out))) ERR(r_);
    if ((r_ = nc_close(netcdf_id_out))) ERR(r_);

    if (append) {
      std::cout << "Existing NetCDF file opened for output" << std::endl;
    } else {
      std::cout << "NetCDF output file created" << std::endl;
    }
  }

  /**
   * This function collects all dimensions that are used bysome of the
   * variables and creates them inside the file with the ID passed as an
   * argument. It collects the IDs of the newly created dimensions. Note that
   * the NetCDF file has to be in define mode.
   *
   * \param[in] netcdf_id_out The ID of the output file as used by the NetCDF
   *                          library.
   * \return                  Map from the old dimension IDs to the new ones.
   */
  std::map<int, int> create_all_dimensions(int netcdf_id_out) {
    
    // we need to collect all dimension ids that are used
    std::set<int> all_used_dim_ids;
    for (int v = 0; v < n_variables_; v++) {
      std::vector<int> dim_ids(variable_dims_[v]);
      // NetCDF: get the ids of all dimensions for the variable
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);
      for (int d = 0; d < variable_dims_[v]; d++) {
        all_used_dim_ids.insert(dim_ids[d]);
      }
    }
    
    // create all dimensions
    std::map<int,int> dim_map;
    for (std::set<int>::iterator dim = all_used_dim_ids.begin();
        dim != all_used_dim_ids.end(); dim++) {
      char dim_name[NC_MAX_NAME+1];
      size_t dim_length;
      // NetCDF: get name and length for each dimension
      if ((r_ = nc_inq_dim(netcdf_id_, *dim, dim_name, &dim_length))) ERR(r_);

      int dim_id;
      // check whether dimension exists already (i.e. when we are appending)
      int nc_return = nc_inq_dimid(netcdf_id_out, dim_name, &dim_id);
      if (nc_return) {
        if (nc_return == NC_EBADDIM) { // the dimension doesn't exist yet
          // NetCDF: write name and length for each dimension
          if ((r_ = nc_def_dim(netcdf_id_out, dim_name, dim_length, &dim_id))) ERR(r_);
        } else {
          ERR(nc_return);
        }
      }

      dim_map[*dim] = dim_id;
    }

    return dim_map;
  }

  /**
   * This function goes through all variables that have originally been read
   * from the input NetCDF file and creates them in the NetCDF file with the
   * ID passed as an argument. The variables are created with the same
   * dimensions in the same order as in the original file. Note that the
   * NetCDF file has to be in define mode.
   *
   * \param[in] netcdf_id_out The ID of the output file as used by the NetCDF
   *                          library.
   * \param[in] dim_map       Map from the old dimension IDs to the new ones.
   */
  void create_all_variables(int netcdf_id_out, const std::map<int, int> dim_map) {

    for (int v = 0; v < n_variables_; v++) {

      char var_name[NC_MAX_NAME+1];
      // NetCDF: get the name of the original variable
      if ((r_ = nc_inq_varname(netcdf_id_, variable_ids_[v],
              var_name))) ERR(r_);

      std::vector<int> dim_ids(variable_dims_[v]);
      // NetCDF: get the original ids of all dimensions for the original variable
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);

      // translate dimension ids for variable to ids for output file
      std::vector<int> dim_ids_out(variable_dims_[v]);
      for (int d = 0; d < variable_dims_[v]; d++) {
        dim_ids_out[d] = dim_map[dim_ids[d]];
      }

      int var_id;
      // NetCDF: define new variable
      if ((r_ = nc_def_var(netcdf_id_out, var_name, NC_DOUBLE,
              variable_dims_[v], &dim_ids_out[0], &var_id))) ERR(r_);
    }
  }

  /**
   * This function writes data back to a NetCDF file.
   *
   * \param[in] filename_out  Full or relative path to the output file.
   * \param[in] data          Vector of data arrays as returned by
   *                          recompose_data().
   */
  void write_data(std::string filename_out, std::vector< HostVector<Scalar> > data) {

    // NetCDF: open file for read/write access
    int netcdf_id_out;
    if ((r_ = nc_open_par(filename_out.c_str(), NC_MPIIO|NC_WRITE,
            MPI_COMM_WORLD, MPI_INFO_NULL, &netcdf_id_out))) ERR(r_);

    std::vector<int> new_variable_ids = get_new_variable_ids(netcdf_id_out);

    for (int v = 0; v < n_variables_; v++) {
      if (variable_is_empty_[v]) continue;

      // NetCDF: write data array for variable
      if ((r_ = nc_put_vara(netcdf_id_out, new_variable_ids[v], &(start_[v][0]),
              &(count_[v][0]), GET_POINTER(data[v])))) ERR(r_);
    }

    if ((r_ = nc_close(netcdf_id_out))) ERR(r_);
    if (!my_rank_) std::cout << "Data written to NetCDF file" << std::endl;
  }

  /**
   * This function finds the IDs used by the NetCDF library for the newly
   * created variables and returns them in the same order as #variable_ids_.
   *
   * \param[in] netcdf_id_out The ID of the output file as used by the NetCDF
   *                          library.
   * \return                  IDs for all selected variables, used by the
   *                          NetCDF library.
   */
  std::vector<int> get_new_variable_ids(int netcdf_id_out) {

    std::vector<int> new_variable_ids(n_variables_);

    for (int v = 0; v < n_variables_; v++) {

      char var_name[NC_MAX_NAME+1];
      // NetCDF: get variable name from input file
      if ((r_ = nc_inq_varname(netcdf_id_, variable_ids_[v],
              var_name))) ERR(r_);

      int new_var_id;
      // NetCDF: get variable id from output file
      if ((r_ = nc_inq_varid(netcdf_id_out, var_name,
              &new_var_id))) ERR(r_);

      new_variable_ids[v] = new_var_id;
    }

    return new_variable_ids;

  }

};
