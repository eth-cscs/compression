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
 *             This software may be modified and distributed under the terms of
 *             the BSD license. See the [LICENSE file](LICENSE.md) for details.
 *
 *  \author Will Sawyer (CSCS)
 *  \author Ben Cumming (CSCS)
 *  \author Manuel Schmid (CSCS)
 */

#pragma once

#include <mpi.h>

/**
 * A simple templated struct that has a variable 'value'. This is overloaded
 * so 'value' always contains the MPI type corresponding to the typename T.
 * The standard template doesn't define 'value' so a compiler error is raised
 * when the helper is used with an unsupported type.
 *
 * \see mpi_type_helper<float>, mpi_type_helper<double>, mpi_type_helper<int>
 */
template <typename T>
struct mpi_type_helper {};

/**
 * \brief Overloaded template (for floats), see mpi_type_helper.
 */
template <>
struct mpi_type_helper<float> {
  static const int value = MPI_FLOAT; ///< Variable holding the MPI type.
};

/**
 * \brief Overloaded template (for doubles), see mpi_type_helper.
 */
template <>
struct mpi_type_helper<double> {
  static const int value = MPI_DOUBLE; ///< Variable holding the MPI type.
};

/**
 * \brief Overloaded template (for integers), see mpi_type_helper.
 */
template <>
struct mpi_type_helper<int> {
  static const int value = MPI_INT; ///< Variable holding the MPI type.
};
