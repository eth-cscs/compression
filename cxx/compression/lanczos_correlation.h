#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

template <typename ScalarType>
void lanczos_correlation(const MatrixXX &Xtranslated, const int ne, const ScalarType tol, const int max_iter, MatrixXX &EV, bool reorthogonalize = false );

#include "lanczos_correlation.txx"
