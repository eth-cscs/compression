#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

template <typename ScalarType>
void lanczos_correlation(const MatrixXX &Xtranslated, const int nbr_eig, const ScalarType tol, const int max_iter, MatrixXX &EV );

#include "lanczos_correlation.txx"
