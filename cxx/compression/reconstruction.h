template <typename ScalarType>
void reconstruction( const std::vector<int> &gamma_ind, const std::vector<GenericColMatrix> &EOFs, const GenericColMatrix &theta, const std::vector<GenericColMatrix> &Xreduced, GenericRowMatrix &Xreconstructed );

#include "reconstruction.txx"
