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

    initialize_data(variables, compressed_dims, indexed_dims);
    select_data_ranges();
    set_up_mapping();
  }


  DeviceMatrix<Scalar> construct_matrix() {

    std::vector< HostVector<Scalar> > data = read_data();
    HostMatrix<Scalar> restructured_data = restructure_data(data);
    DeviceMatrix<Scalar> output = restructured_data; // copy to device

    return output;
  }


  void write_matrix(DeviceMatrix<Scalar> input) {

    HostMatrix<Scalar> data = input; // copy to host
    std::vector< HostVector<Scalar> > recomposed_data = recompose_data(data);
    // only the first process creates the new file
    if (!my_rank_) setup_output_file("output.nc4");
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "continuing after barrier" << std::endl;
    write_data("output.nc4", recomposed_data);

  }


private:

  int r_; // used for NetCDF error handling only
  int process_getting_more_data_ = 0; // used when data can't be distributed
                                      // evenly between processes

  // initialized in constructor:
  std::string filename_;
  Stacking stacking_;

  // initialized in initialize_data:
  int mpi_processes_;
  int my_rank_;
  int netcdf_id_;
  int n_variables_;
  std::vector<int> variable_ids_;             // for each variable
  std::set<std::string> compressed_dims_;
  std::map<std::string, int> indexed_dims_;

  // initialized in select_data_ranges:
  std::vector<int> variable_dims_;            // for each variable
  std::vector<bool> variable_is_empty_;       // for each variable
  std::vector< std::vector<size_t> > start_;  // for each variable & dimension
  std::vector< std::vector<size_t> > count_;  // for each variable & dimension

  // initialized in set_up_mapping:
  std::vector<int> variable_rows_;            // for each variable
  std::vector<int> variable_cols_;            // for each variable
  std::vector< std::vector<int> > row_map_;   // for each variable & dimension
  std::vector< std::vector<int> > col_map_;   // for each variable & dimension


  int nc_get_vara(int ncid, int varid, const size_t start[], const size_t
      count[], double *dp) {
    int retval = nc_get_vara_double(ncid, varid, start, count, dp);
    return retval;
  }


  int nc_get_vara(int ncid, int varid, const size_t start[], const size_t
      count[], float *fp) {
    int retval = nc_get_vara_float(ncid, varid, start, count, fp);
    return retval;
  }


  int nc_put_vara(int ncid, int varid, const size_t start[], const size_t
      count[], const double *fp) {
    int retval = nc_put_vara_double(ncid, varid, start, count, fp);
    return retval;
  }


  int nc_put_vara(int ncid, int varid, const size_t start[], const size_t
      count[], const float *fp) {
    int retval = nc_put_vara_float(ncid, varid, start, count, fp);
    return retval;
  }


  void increment_process_getting_more_data() {
    process_getting_more_data_++;
    if (process_getting_more_data_ == mpi_processes_) {
      process_getting_more_data_ = 0;
    }
  }


  void initialize_data(const std::vector<std::string> &variables,
      const std::vector<std::string> &compressed_dims,
      const std::vector<std::string> &indexed_dims) {

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


  void select_data_ranges() {
    /* This function sets up the internal variables start_ and count_. */
    
    variable_dims_     = std::vector<int>(n_variables_);
    variable_is_empty_ = std::vector<bool>(n_variables_, false);
    
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


  void calculate_distributed_dims_data_range(const int variable,
      const std::vector<int> &distributed_dims) {
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


  std::vector<int> get_process_distribution(const int n_distributed_dims) {
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

      if (!my_rank_) std::cout << "Dimensions for variable " << v << ":"
          << std::endl;

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
          if (!my_rank_) std::cout << "  " << dim_name
              << " (indexed, selecting index " << indexed_dims_[dim_name]
              << "), map: " << row_map_[v][d] << " " << col_map_[v][d] << std::endl;
        }

        else if (compressed_dims_.find(dim_name) != compressed_dims_.end()) {
          // the dimension is compressed
          row_map_[v][d] = row_interelement_distance;
          col_map_[v][d] = 0; // all values are in the same column
          row_interelement_distance *= count_[v][d];
          if (!my_rank_) std::cout << "  " << dim_name << " (compressed, "
              << count_[v][d] << " entries), map: " << row_map_[v][d] << " " << col_map_[v][d] << std::endl;
        }
        
        else {
          // the dimension is distributed
          row_map_[v][d] = 0; // all values are in the same row
          col_map_[v][d] = col_interelement_distance;
          col_interelement_distance *= count_[v][d];
          if (!my_rank_) std::cout << "  " << dim_name
              << " (distributed, selecting entries " << start_[v][d]
              << " to " << start_[v][d] + count_[v][d] << "), map: " << row_map_[v][d] << " " << col_map_[v][d] << std::endl;
        }
      }
      variable_rows_[v] = row_interelement_distance;
      variable_cols_[v] = col_interelement_distance;
    }
  }


  std::vector< HostVector<Scalar> > read_data() {

    std::vector< HostVector<Scalar> > output(n_variables_);

    for (int v = 0; v < n_variables_; v++) {
      output[v] = HostVector<Scalar>(variable_rows_[v] * variable_cols_[v]);
      // NetCDF: get data array for variable
      if ((r_ = nc_get_vara(netcdf_id_, variable_ids_[v], &(start_[v][0]),
              &(count_[v][0]), GET_POINTER(output[v])))) ERR(r_);
    }

    if ((r_ = nc_close(netcdf_id_))) ERR(r_);
    if (!my_rank_) std::cout << "Data successfully read from NetCDF file" << std::endl;
    return output;
  }


  HostMatrix<Scalar> restructure_data(std::vector< HostVector<Scalar> > &data) {

    int output_row_size = variable_rows_[0];
    int output_col_size = variable_cols_[0];
    for (int v = 1; v < n_variables_; v++) {
      if (stacking_ == HORIZONTAL) {
        assert(output_row_size == variable_rows_[v]);
        if (!variable_is_empty_[v]) output_col_size += variable_cols_[v];
      } else {
        assert(output_col_size == variable_cols_[v]);
        output_row_size += variable_rows_[v];
      }
    }

    HostMatrix<Scalar> output(output_row_size, output_col_size);

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
        //if (!my_rank_) std::cout << "dim_indicies: ";
        for (int d = variable_dims_[v] - 1; d >= 0; d--) {
          // increase indices of next dimension and reset current index
          // if a dimension has reached the maximum
          if (dim_indicies[d] == count_[v][d]) {
            dim_indicies[d] = 0;
            dim_indicies[d-1] += 1;
          }
          //if (!my_rank_) std::cout << dim_indicies[d] << " ";
          output_row += dim_indicies[d] * row_map_[v][d];
          output_col += dim_indicies[d] * col_map_[v][d];
        }
        //if (!my_rank_) std::cout << "writing value " << data[v](i) << " to (" << output_row << ", " << output_col << ")" << std::endl;
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
    if (!my_rank_) std::cout << "Data successfully restructured into a 2D matrix" << std::endl;
    return output;
  }


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
        //if (!my_rank_) std::cout << "dim_indicies: ";
        for (int d = variable_dims_[v] - 1; d >= 0; d--) {
          // increase indices of next dimension and reset current index
          // if a dimension has reached the maximum
          if (dim_indicies[d] == count_[v][d]) {
            dim_indicies[d] = 0;
            dim_indicies[d-1] += 1;
          }
          //if (!my_rank_) std::cout << dim_indicies[d] << " ";
          data_row += dim_indicies[d] * row_map_[v][d];
          data_col += dim_indicies[d] * col_map_[v][d];
        }
        //if (!my_rank_) std::cout << "writing value " << data[v](i) << " to (" << output_row << ", " << output_col << ")" << std::endl;
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


  void setup_output_file(std::string filename_out) {
    /* This creates a new file 'filename' with all the dimensions and
     * variables that have been read from the original input file. */

    // NetCDF: create new file, overwriting if it already exists
    int netcdf_id_out;
    std::cout << "creating file" << std::endl;
    if ((r_ = nc_create_par(filename_out.c_str(), NC_CLOBBER|NC_NETCDF4|NC_MPIIO,
            MPI_COMM_WORLD, MPI_INFO_NULL, &netcdf_id_out))) ERR(r_);

    // NetCDF: put output file into define mode
    std::cout << "putting file in define mode" << std::endl;
    //if ((r_ = nc_redef(netcdf_id_out))) ERR(r_);

    // create dimensions & variables
    std::cout << "creating dimensions" << std::endl;
    std::map<int, int> dimension_map = create_all_dimensions(netcdf_id_out);
    std::cout << "creating variables" << std::endl;
    create_all_variables(netcdf_id_out, dimension_map);
    std::cout << "file created" << std::endl;

    // NetCDF: end define mode and close output file
    if ((r_ = nc_enddef(netcdf_id_out))) ERR(r_);
    if ((r_ = nc_close(netcdf_id_out))) ERR(r_);

  }

  std::map<int, int> create_all_dimensions(int netcdf_id_out) {
    /* collects all dimensions that are used by some of the variables
     * and creates them inside the file specified by netcdf_id_out.
     * returns a map from the old dimension ids to the new ones.
     * the output file has to be in define mode. */
    
    // we need to collect all dimension ids that are used
    std::set<int> all_used_dim_ids;
    for (int v = 0; v < n_variables_; v++) {
      std::vector<int> dim_ids(variable_dims_[v]);
      std::cout << "searching variable " << variable_ids_[v] << std::endl;
      // NetCDF: get the ids of all dimensions for the variable
      if ((r_ = nc_inq_vardimid(netcdf_id_, variable_ids_[v],
              &dim_ids[0]))) ERR(r_);
      for (int d = 0; d < variable_dims_[v]; d++) {
        all_used_dim_ids.insert(dim_ids[d]);
      }
    }
    
    // create all dimensions
    std::map<int,int> dim_map;
    for (std::set<int>::iterator dim = all_used_dim_ids.begin(); dim != all_used_dim_ids.end(); dim++) {
      char dim_name[NC_MAX_NAME+1];
      size_t dim_length;
      // NetCDF: get name and length for each dimension
      if ((r_ = nc_inq_dim(netcdf_id_, *dim, dim_name, &dim_length))) ERR(r_);
      int dim_id;
      // NetCDF: write name and length for each dimension
      if ((r_ = nc_def_dim(netcdf_id_out, dim_name, dim_length, &dim_id))) ERR(r_);
      dim_map[*dim] = dim_id;
    }

    return dim_map;
  }

  void create_all_variables(int netcdf_id_out, std::map<int, int> dim_map) {

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


  void write_data(std::string filename_out, std::vector< HostVector<Scalar> > data) {

    // NetCDF: open file for read/write access
    int netcdf_id_out;
    if ((r_ = nc_open_par(filename_out.c_str(), NC_MPIIO, MPI_COMM_WORLD,
            MPI_INFO_NULL, &netcdf_id_out))) ERR(r_);


    std::vector<int> new_variable_ids = get_new_variable_ids(netcdf_id_out);


    for (int v = 0; v < n_variables_; v++) {
      if (variable_is_empty_[v]) continue;

      // NetCDF: write data array for variable (TODO: change variable_ids_)
      if ((r_ = nc_put_vara(netcdf_id_out, new_variable_ids[v], &(start_[v][0]),
              &(count_[v][0]), GET_POINTER(data[v])))) ERR(r_);
    }

    if ((r_ = nc_close(netcdf_id_))) ERR(r_);
    if (!my_rank_) std::cout << "Data successfully written to NetCDF file" << std::endl;
  }

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

  std::vector<int> get_new_dimension_ids(int netcdf_id_out, int v) {
    /* this returns a map of the old dimension ids to the new ones for a given
     * variable. */

    std::vector<int> new_dimension_ids(variable_dims_[v]);

    std::vector<int> dim_ids(variable_dims_[v]);
    // NetCDF: get dimension ids for variable
    if ((r_ = nc_inq_varndims(netcdf_id_, variable_ids_[v],
            &dim_ids[0]))) ERR(r_);

    for (int d = 0; d < variable_dims_[v]; d++) {
      char dim_name[NC_MAX_NAME+1];
      // NetCDF: get dimension name from id
      if ((r_ = nc_inq_dimname(netcdf_id_, dim_ids[d],
              dim_name))) ERR(r_);
      int new_dim_id;
      // NetCDF: get dimension id from output file
      if ((r_ = nc_inq_dimid(netcdf_id_out, dim_name,
              &new_dim_id))) ERR(r_);

      new_dimension_ids[d] = new_dim_id;
    }

    return new_dimension_ids;

  }

};
