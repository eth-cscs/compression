#include <Eigen/Dense>
using namespace Eigen;

template <typename ScalarType>
ScalarType L_value( const ArrayX1i &gamma_ind, const MatrixXX *TT, const MatrixXXrow &X, const MatrixXX &theta );

#include "L_value.txx"
