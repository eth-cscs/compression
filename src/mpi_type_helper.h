/** \file mpi_type_helper.h
 *
 *  This file contains a type overloaded helper struct that is used for
 *  selecting the right variable type for MPI calls in templated
 *  classes and functions.
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

#include <mpi.h>

template <typename T>
struct mpi_type_helper {};

template <>
struct mpi_type_helper<float> {
  static const int value = MPI_FLOAT;
};

template <>
struct mpi_type_helper<double> {
  static const int value = MPI_DOUBLE;
};

template <>
struct mpi_type_helper<int> {
  static const int value = MPI_INT;
};
