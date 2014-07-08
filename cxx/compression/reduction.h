template <typename ScalarType>
void reduction( const std::vector<int> &gamma_ind, const std::vector<GenericColMatrix> &EOFs, const GenericRowMatrix &X, const GenericColMatrix &theta, std::vector<GenericColMatrix> &Xreduced );

#include "reduction.txx"
