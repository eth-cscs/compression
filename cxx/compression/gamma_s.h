#include <limits>

template <typename ScalarType>
void gamma_s( const GenericRowMatrix &X, const GenericColMatrix &theta, const std::vector<GenericColMatrix> TT, std::vector<int> &gamma_ind );

#include "gamma_s.txx"
