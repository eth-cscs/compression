#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace std; // we dump the namespace because we use it extensively



template<class Scalar>
void print_statistics(const DeviceMatrix<Scalar> &X_original,
                      const DeviceMatrix<Scalar> &X_reconstructed,
                      const std::vector<std::string> variable_names,
                      const std::vector<int> row_start,
                      const std::vector<int> row_count,
                      const std::vector<int> col_start,
                      const std::vector<int> col_count) {

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  for (int v = 0; v < variable_names.size(); v++) {

    // collect local information
    int rs = row_start[v];
    int rc = row_count[v];
    int re = rs + rc - 1;
    int cs = col_start[v];
    int cc = col_count[v];
    int ce = cs + cc - 1;

    int local_count = row_count[v] * col_count[v];

    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_min_original = std::numeric_limits<Scalar>::max();
    Scalar local_max_original = - std::numeric_limits<Scalar>::max();
    Scalar local_sum_original = 0.0;
    Scalar local_min_reconstructed = std::numeric_limits<Scalar>::max();
    Scalar local_max_reconstructed = - std::numeric_limits<Scalar>::max();
    Scalar local_sum_reconstructed = 0.0;
    Scalar local_max_absolute_error = 0.0;
    Scalar local_square_error_sum = 0.0;

    if (local_count) {
#if defined(USE_EIGEN)
      local_min_original = X_original.block(rs,cs,rc,cc).minCoeff();
      local_max_original = X_original.block(rs,cs,rc,cc).maxCoeff();
      local_sum_original = X_original.block(rs,cs,rc,cc).sum();
      local_min_reconstructed = X_reconstructed.block(rs,cs,rc,cc).minCoeff();
      local_max_reconstructed = X_reconstructed.block(rs,cs,rc,cc).maxCoeff();
      local_sum_reconstructed = X_reconstructed.block(rs,cs,rc,cc).sum();
      local_max_absolute_error = ( X_reconstructed.block(rs,cs,rc,cc)
          - X_original.block(rs,cs,rc,cc) ).cwiseAbs().maxCoeff();
      local_square_error_sum = ( X_reconstructed.block(rs,cs,rc,cc)
          - X_original.block(rs,cs,rc,cc) ).squaredNorm();
#elif defined(USE_MINLIN)
      local_min_original = min(X_original(rs, re, cs, ce));
      local_max_original = max(X_original(rs, re, cs, ce));
      local_sum_original = sum(X_original(rs, re, cs, ce));
      local_min_reconstructed = min(X_reconstructed(rs, re, cs, ce));
      local_max_reconstructed = max(X_reconstructed(rs, re, cs, ce));
      local_sum_reconstructed = sum(X_reconstructed(rs, re, cs, ce));
      local_max_absolute_error = max(abs(X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce)));
      local_square_error_sum = sum(mul((X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce)), (X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce))));
#endif
    }

    // initialize global variables
    int count;
    Scalar min_original;
    Scalar max_original;
    Scalar sum_original;
    Scalar min_reconstructed;
    Scalar max_reconstructed;
    Scalar sum_reconstructed;
    Scalar max_absolute_error;
    Scalar square_error_sum;


    // MPI calls
    MPI_Allreduce(&local_count, &count, 1,
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&local_min_original, &min_original, 1,
        mpi_type_helper<Scalar>::value, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_original, &max_original, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum_original, &sum_original, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&local_min_reconstructed, &min_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_reconstructed, &max_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum_reconstructed, &sum_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&local_max_absolute_error, &max_absolute_error, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_square_error_sum, &square_error_sum, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);


    // calculate means and original range
    Scalar mean_original = sum_original / count;
    Scalar mean_reconstructed = sum_reconstructed / count;
    Scalar rms_error = sqrt(square_error_sum / count);
    Scalar original_range = max_original - min_original;


    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_square_deviations_original = 0.0;
    Scalar local_square_deviations_reconstructed = 0.0;
    Scalar local_multiplied_deviations = 0.0;


    // calculate values depending on mean
    // (x-x_mean)^2, (y-y_mean)^2, (x-x_mean)*(y-y_mean)
    if (local_count) {
#if defined(USE_EIGEN)
      local_square_deviations_original =
        ( (X_original.block(rs,cs,rc,cc).array() - mean_original)
        * (X_original.block(rs,cs,rc,cc).array() - mean_original) ).sum();
      local_square_deviations_reconstructed =
        ( (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed)
        * (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed) ).sum();
      local_multiplied_deviations =
        ( (X_original.block(rs,cs,rc,cc).array() - mean_original)
        * (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed) ).sum();
#elif defined(USE_MINLIN)
      local_square_deviations_original = sum(mul(
          (X_original(rs, re, cs, ce) - mean_original),
          (X_original(rs, re, cs, ce) - mean_original)));
      local_square_deviations_reconstructed = sum(mul(
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed),
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed)));
      local_multiplied_deviations = sum(mul(
          (X_original(rs, re, cs, ce) - mean_original),
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed)));
#endif
    }


    Scalar square_deviations_original;
    Scalar square_deviations_reconstructed;
    Scalar multiplied_deviations;
    MPI_Allreduce(&local_square_deviations_original, &square_deviations_original, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_square_deviations_reconstructed, &square_deviations_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_multiplied_deviations, &multiplied_deviations, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    Scalar std_original = sqrt(square_deviations_original / count);
    Scalar std_reconstructed = sqrt(square_deviations_reconstructed / count);
    Scalar correlation = multiplied_deviations /
        (sqrt(square_deviations_original) * sqrt(square_deviations_reconstructed));


    // signal-to-residual ratio (SSR) & PrecisionBits
    Scalar std_residual = sqrt((square_deviations_original +
          square_deviations_reconstructed - 2 * multiplied_deviations) / count);
    Scalar ssr = log2(std_original / std_residual);
    Scalar precision_bits = log2(original_range / (2 * max_absolute_error));


    // normalize errors by original range
    max_absolute_error /= original_range;
    rms_error /= original_range;


    // Output
    int col_width = 14;
    if (!my_rank) {
      cout << " Variable " << variable_names[v] << ":" << endl << endl << setfill(' ')
          << setw(col_width) << "min" << setw(col_width) << "max"
          << setw(col_width) << "mean" << setw(col_width) << "std" << endl
          << setw(col_width) << min_original << setw(col_width) << max_original
          << setw(col_width) << mean_original << setw(col_width)
          << std_original << "   (original data)" << endl
          << setw(col_width) << min_reconstructed << setw(col_width)
          << max_reconstructed << setw(col_width) << mean_reconstructed
          << setw(col_width) << std_reconstructed << "   (reconstructed data)" << endl
          << endl
          << "   maximum error:  " << max_absolute_error << " (normalized with range)" << endl
          << "   RMS error:      " << rms_error << " (normalized with range)" << endl
          << "   correlation:    " << correlation << endl
          << "   SRR:            " << ssr << endl
          << "   PrecisionBits:  " << precision_bits << endl
          << endl;
    }
  }
}
