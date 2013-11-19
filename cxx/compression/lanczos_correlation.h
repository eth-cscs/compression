#include <Eigen/Dense>
using namespace Eigen;

template <typename ScalarType>
Matrix<ScalarType, Dynamic, Dynamic> lanczos_correlation(const Matrix<ScalarType, Dynamic, Dynamic> Xtranslated, const int nbr_eig, const ScalarType tol );

#include "lanczos_correlation.txx"
