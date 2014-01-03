#include <Eigen/Dense>
using namespace Eigen;

typedef Array<int, Dynamic, 1> ArrayX1i;

ArrayX1i gamma_zero(const int *nl_global, const int my_rank, const int K );

#include "gamma_zero.cxx"
