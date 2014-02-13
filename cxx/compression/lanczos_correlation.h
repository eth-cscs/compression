using namespace std;

#include "generic_wrappers.h"

template <typename ScalarType>
bool lanczos_correlation(const GenericColMatrix &Xtranslated, const int ne, const ScalarType tol, const int max_iter, GenericColMatrix &EV, bool reorthogonalize = false );

#include "lanczos_correlation.txx"
