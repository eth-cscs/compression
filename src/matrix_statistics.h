/** \file matrix_statistics.h
 *
 *  This file defines a function to print out statistics about two matrices
 *  containing data from multiple variables.
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
                      const std::vector<std::string> &variable_names,
                      const std::vector<Scalar> &variable_mean,
                      const std::vector<Scalar> &variable_max,
                      const std::vector<int> &row_start,
                      const std::vector<int> &row_count,
                      const std::vector<int> &col_start,
                      const std::vector<int> &col_count) {

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  for (int v = 0; v < variable_names.size(); v++) {

    // set up data ranges
    int rs = row_start[v];
    int rc = row_count[v];
    int re = rs + rc - 1;
    int cs = col_start[v];
    int cc = col_count[v];
    int ce = cs + cc - 1;

    // collect variable count
    int local_count = row_count[v] * col_count[v];
    int count;
    MPI_Allreduce(&local_count, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    // MINIMUM & MAXIMUM
    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_min_original = std::numeric_limits<Scalar>::max();
    Scalar local_max_original = - std::numeric_limits<Scalar>::max();
    Scalar local_min_reconstructed = std::numeric_limits<Scalar>::max();
    Scalar local_max_reconstructed = - std::numeric_limits<Scalar>::max();

    if (local_count) {
#if defined(USE_EIGEN)
      local_min_original = X_original.block(rs,cs,rc,cc).minCoeff();
      local_max_original = X_original.block(rs,cs,rc,cc).maxCoeff();
      local_min_reconstructed = X_reconstructed.block(rs,cs,rc,cc).minCoeff();
      local_max_reconstructed = X_reconstructed.block(rs,cs,rc,cc).maxCoeff();
#elif defined(USE_MINLIN)
      local_min_original = min(X_original(rs, re, cs, ce));
      local_max_original = max(X_original(rs, re, cs, ce));
      local_min_reconstructed = min(X_reconstructed(rs, re, cs, ce));
      local_max_reconstructed = max(X_reconstructed(rs, re, cs, ce));
#endif
    }

    Scalar min_original;
    MPI_Allreduce(&local_min_original, &min_original, 1,
        mpi_type_helper<Scalar>::value, MPI_MIN, MPI_COMM_WORLD);
    Scalar max_original;
    MPI_Allreduce(&local_max_original, &max_original, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);
    Scalar min_reconstructed;
    MPI_Allreduce(&local_min_reconstructed, &min_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_MIN, MPI_COMM_WORLD);
    Scalar max_reconstructed;
    MPI_Allreduce(&local_max_reconstructed, &max_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);

    // reverse transformation & calculate range
    min_original = variable_mean[v] + min_original * variable_max[v];
    max_original = variable_mean[v] + max_original * variable_max[v];
    min_reconstructed = variable_mean[v] + min_reconstructed * variable_max[v];
    max_reconstructed = variable_mean[v] + max_reconstructed * variable_max[v];
    Scalar original_range = max_original - min_original;


    // MEANS
    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_sum_original = 0.0;
    Scalar local_sum_reconstructed = 0.0;

    if (local_count) {
#if defined(USE_EIGEN)
      local_sum_original = X_original.block(rs,cs,rc,cc).sum();
      local_sum_reconstructed = X_reconstructed.block(rs,cs,rc,cc).sum();
#elif defined(USE_MINLIN)
      local_sum_original = sum(X_original(rs, re, cs, ce));
      local_sum_reconstructed = sum(X_reconstructed(rs, re, cs, ce));
#endif
    }

    Scalar sum_original;
    MPI_Allreduce(&local_sum_original, &sum_original, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    Scalar sum_reconstructed;
    MPI_Allreduce(&local_sum_reconstructed, &sum_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    // calculate mean & reverse transformation
    Scalar mean_original_untransformed = sum_original / count;
    Scalar mean_reconstructed_untransformed = sum_reconstructed / count;
    Scalar mean_original = variable_mean[v] + mean_original_untransformed * variable_max[v];
    Scalar mean_reconstructed = variable_mean[v] + mean_reconstructed_untransformed * variable_max[v];


    // MAX ABS ERROR, RMS ERROR
    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_max_absolute_error = 0.0;
    Scalar local_square_error_sum = 0.0;

    if (local_count) {
#if defined(USE_EIGEN)
      local_max_absolute_error = ( X_reconstructed.block(rs,cs,rc,cc)
          - X_original.block(rs,cs,rc,cc) ).cwiseAbs().maxCoeff();
      local_square_error_sum = ( X_reconstructed.block(rs,cs,rc,cc)
          - X_original.block(rs,cs,rc,cc) ).squaredNorm();
#elif defined(USE_MINLIN)
      local_max_absolute_error = max(abs(X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce)));
      local_square_error_sum = sum(mul((X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce)), (X_reconstructed(rs, re, cs, ce)
            - X_original(rs, re, cs, ce))));
#endif
    }

    Scalar max_absolute_error;
    MPI_Allreduce(&local_max_absolute_error, &max_absolute_error, 1,
        mpi_type_helper<Scalar>::value, MPI_MAX, MPI_COMM_WORLD);
    Scalar square_error_sum;
    MPI_Allreduce(&local_square_error_sum, &square_error_sum, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    // calculate rms error & reverse transformation
    max_absolute_error *= variable_max[v];
    Scalar rms_error = sqrt(square_error_sum / count) * variable_max[v];


    // STANDARD DEVIATIONS
    // these values depend on the mean calculated earlier
    // (x-x_mean)^2, (y-y_mean)^2, (x-x_mean)*(y-y_mean)
    // initialize local variables to values that do not contribute
    // to global values if the current process has no data for a
    // variable
    Scalar local_square_deviations_original = 0.0;
    Scalar local_square_deviations_reconstructed = 0.0;
    Scalar local_multiplied_deviations = 0.0;

    if (local_count) {
#if defined(USE_EIGEN)
      local_square_deviations_original =
        ( (X_original.block(rs,cs,rc,cc).array() - mean_original_untransformed)
        * (X_original.block(rs,cs,rc,cc).array() - mean_original_untransformed) ).sum();
      local_square_deviations_reconstructed =
        ( (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed_untransformed)
        * (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed_untransformed) ).sum();
      local_multiplied_deviations =
        ( (X_original.block(rs,cs,rc,cc).array() - mean_original_untransformed)
        * (X_reconstructed.block(rs,cs,rc,cc).array() - mean_reconstructed_untransformed) ).sum();
#elif defined(USE_MINLIN)
      local_square_deviations_original = sum(mul(
          (X_original(rs, re, cs, ce) - mean_original_untransformed),
          (X_original(rs, re, cs, ce) - mean_original_untransformed)));
      local_square_deviations_reconstructed = sum(mul(
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed_untransformed),
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed_untransformed)));
      local_multiplied_deviations = sum(mul(
          (X_original(rs, re, cs, ce) - mean_original_untransformed),
          (X_reconstructed(rs, re, cs, ce) - mean_reconstructed_untransformed)));
#endif
    }

    Scalar square_deviations_original;
    MPI_Allreduce(&local_square_deviations_original, &square_deviations_original, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    Scalar square_deviations_reconstructed;
    MPI_Allreduce(&local_square_deviations_reconstructed, &square_deviations_reconstructed, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    Scalar multiplied_deviations;
    MPI_Allreduce(&local_multiplied_deviations, &multiplied_deviations, 1,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);

    // calculate standard deviations & correlation with reverse transformation
    Scalar std_original = sqrt(square_deviations_original / count) * variable_max[v];
    Scalar std_reconstructed = sqrt(square_deviations_reconstructed / count) * variable_max[v];
    Scalar correlation = multiplied_deviations /
        (sqrt(square_deviations_original) * sqrt(square_deviations_reconstructed));


    // SIGNAL-TO-RESIDUAL RATIO (SSR) & PRECISIONBITS
    // with reverse transformation
    Scalar std_residual = sqrt((square_deviations_original +
          square_deviations_reconstructed - 2 * multiplied_deviations) / count) * variable_max[v];
    Scalar ssr = log2(std_original / std_residual);
    Scalar precision_bits = log2(original_range / (2 * max_absolute_error));


    // normalize errors with original range
    // we cannot do this earlier because the PrecisionBits metric uses the max
    // absolute error before normalization
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
