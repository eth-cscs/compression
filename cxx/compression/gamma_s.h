#include <Eigen/Dense>
using namespace Eigen;

template <typename ScalarType>
void gamma_s( const MatrixXXrow &X, const MatrixXX &theta, const std::vector<MatrixXX> TT, ArrayX1i &gamma_ind );

#include "gamma_s.txx"
