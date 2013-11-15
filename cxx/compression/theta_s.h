#include <Eigen/Dense>
using namespace Eigen;

template <typename ScalarType>
Matrix<ScalarType, Dynamic, Dynamic> theta_s(const ArrayX1i gamma_ind, const Matrix<ScalarType, Dynamic, Dynamic> X, const int K);

#include "theta_s.txx"
