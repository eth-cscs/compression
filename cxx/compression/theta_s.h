#include <Eigen/Dense>
using namespace Eigen;

template <typename ScalarType>
void theta_s(const ArrayX1i gamma_ind, const Matrix<ScalarType, Dynamic, Dynamic> X, Matrix<ScalarType, Dynamic, Dynamic> *theta);

#include "theta_s.txx"
