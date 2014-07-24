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
