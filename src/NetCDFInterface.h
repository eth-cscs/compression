#include <vector>
#include <netcdf_par.h>
#include <netcdf.h>
#include "mpi.h"

#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(1);}

template<class Scalar>
class NetCDFInterface
{

public:

  enum Stacking {HORIZONTAL, VERTICAL};

  NetCDFInterface(const std::string filename,
                  const std::vector<std::string> variables,
                  const std::vector<std::string> compressed_dims,
                  const std::vector<std::string> indexed_dims,
                  const Stacking stacking = VERTICAL) {

    // save internal variables
    filename_ = filename;
    stacking_ = stacking;
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

    initialize_data(variables);
    select_data_ranges();
    set_up_mapping();
  }


  DeviceMatrix<Scalar> construct_matrix() {

    std::vector< HostVector<Scalar> > data = read_data();
    HostMatrix<Scalar> restructured_data = restructure_data(data);
    DeviceMatrix<Scalar> output = restructured_data; // copy to device

    return output;
  }


private:

  // initialized in constructor:
  std::string filename_;
  Stacking stacking_;
  std::set<std::string> compressed_dims_;
  std::map<std::string, int> indexed_dims_;

  // initialized in initialize_data:
  int mpi_processes_;
  int my_rank_;
  int netcdf_id_;
  int n_variables_;
  std::vector<int> variable_ids_;             // for each variable

  // initialized in select_data_ranges:
  std::vector<int> variable_dims_;            // for each variable
  std::vector< std::vector<size_t> > start_;  // for each variable & dimension
  std::vector< std::vector<size_t> > count_;  // for each variable & dimension

  // initialized in set_up_mapping:
  std::vector<int> variable_rows_;            // for each variable
  std::vector<int> variable_cols_;            // for each variable
  std::vector< std::vector<int> > row_map_;   // for each variable & dimension
  std::vector< std::vector<int> > col_map_;   // for each variable & dimension



  int N_c_;
  int N_d_;
  int r_;

  // variables for reordering the data
  std::vector<int> row_offset_;             // for each variable
  std::vector<int> col_offset_;             // for each variable



  std::vector< HostVector<Scalar> > read_data() {

    std::vector< HostVector<Scalar> > output(n_variables_);

    for (int v = 0; v < n_variables_; v++) {
      output[v] = HostVector<Scalar>(variable_rows_[v] * variable_cols_[v]);
      // NetCDF: get data array for variable
      if ((r_ = nc_get_vara(netcdf_id_, variable_ids_[v], &(start_[v][0]),
              &(count_[v][0]), GET_POINTER(output[v])))) ERR(r_);
    }

    return output;
  }


  HostMatrix<Scalar> restructure_data(std::vector< HostVector<Scalar> > data) {

    HostMatrix<Scalar> output(N_c_, N_d_);

    for (int v = 0; v < n_variables_; v++) {

      std::vector<int> dim_indicies(variable_dims_[v], 0);
      for (int i = 0; i < data[v].size(); i++) {

        int output_row = row_offset_[v];
        int output_col = col_offset_[v];
        // we leave out d=0 here as this would involve updating at index -1
        // which doesn't exist. this situation shouldn't occur anyway.
        for (int d = variable_dims_[v] - 1; d > 0; d--) {
          // increase indices of next dimension and reset current index
          // if a dimension has reached the maximum
          if (dim_indicies[d] == count_[v][d]) {
            dim_indicies[d] = 0;
            dim_indicies[d-1] += 1;
          }
          output_row += dim_indicies[d] * row_map_[v][d];
          output_col += dim_indicies[d] * col_map_[v][d];
        }
        output(output_row, output_col) = data[v](i);

        // increase index along last dimension for next iteration
        dim_indicies[variable_dims_[v]-1]++;
      }
    }
    return output;
  }


  void initialize_data(const std::vector<std::string> variables) {

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

  }


  void select_data_ranges() {
    /* This function sets up the internal variables start_ and count_. */
    
    start_  = std::vector< std::vector<size_t> >(n_variables_);
    count_  = std::vector< std::vector<size_t> >(n_variables_);

    for (int v = 0; v < n_variables_; v++) {

      // NetCDF: get the number of dimensions for the variable
      if ((r_ = nc_inq_varndims(netcdf_id_, variable_ids_[v],
              &variable_dims_[v]))) ERR(r_);

      start_[v] = std::vector<size_t>(variable_dims_[v]);
      count_[v] = std::vector<size_t>(variable_dims_[v]);

      std::vector<int> distributed_dims;

      // NetCDF: get the ids of all dimensions for the variable
      // we use the dim_ids vector as an array here
      std::vector<int> dim_ids(variable_dims_[v]);
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);

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
        }

        else if (compressed_dims_.find(dim_name) != compressed_dims_.end()) {
          // the dimension is compressed
          start_[v][d] = 0;
          count_[v][d] = dim_length;
        }
        
        else {
          // the dimension is distributed
          // we only collect the distributed dimensions and their length in
          // this loop and treat them separately afterwards
          distributed_dims.push_back(d);
          count_[v][d] = dim_length; // will be changed afterwards
        }
      }

      // this adds in the correct data ranges for the distributed dimensions
      calculate_distributed_dims_data_range(v, distributed_dims);
    }
  }


  void calculate_distributed_dims_data_range(int variable,
      std::vector<int> distributed_dims) {
    /* This sets up the start and count for the distributed dimensions of
     * 'variable'. The indices of the distributed dimenions among all the
     * dimensions of the variable are saved in 'distributed_dims'. The length
     * of each dimension is found in count_[var][distributed_dims[i]]. */

    std::vector<int> process_distribution =
        get_process_distribution(distributed_dims.size());

    int r = my_rank_;        // needed for calculating index along dimension
    int p = mpi_processes_;  // needed for calculating index along dimension

    for (int i = 0; i < distributed_dims.size(); i++) {

      // find index along dimension
      // note: this is not immediately obvious, change with care!
      // TODO: explanation
      p /= process_distribution[i];
      int dim_index = r / p;
      r %= p;

      int dim_length = count_[variable][distributed_dims[i]];
      int size_along_dim = dim_length / process_distribution[i];

      start_[variable][distributed_dims[i]] = dim_index * size_along_dim;
      count_[variable][distributed_dims[i]] = size_along_dim;

      // if we are the last process along a dimension, add the remainder
      if (dim_index == process_distribution[i] - 1) {
        count_[variable][distributed_dims[i]] += dim_length
          % process_distribution[i];
      }

    }
  }

  int nc_get_vara(int ncid, int varid, size_t start[], size_t
      count[], double *dp) {
    int retval = nc_get_vara_double(ncid, varid, start, count, dp);
    return retval;
  }

  int nc_get_vara(int ncid, int varid, size_t start[], size_t
      count[], float *fp) {
    int retval = nc_get_vara_float(ncid, varid, start, count, fp);
    return retval;
  }


  void set_up_mapping() {
      
    variable_rows_ = std::vector<int>(n_variables_);
    variable_cols_ = std::vector<int>(n_variables_);

    row_offset_ = std::vector<int>(n_variables_, 0);
    col_offset_ = std::vector<int>(n_variables_, 0);

    for (int v = 0; v < n_variables_; v++) {

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

      if (stacking_ == HORIZONTAL) {
        col_offset_[v] += variable_cols_[v];
      } else {
        row_offset_[v] += variable_rows_[v];
      }
    }

  }

  std::vector<int> get_process_distribution(int n_distributed_dims) {
    /* This builds a list with the number of processes along each distributed
     * dimension. Note: We assume mpi_processes to be a power of 2. */

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



};