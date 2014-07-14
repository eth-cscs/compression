#include "generic_wrappers.h"

template <typename ScalarType>
bool lanczos_correlation(const GenericColMatrix &Xtranslated, const int ne, const ScalarType tol, const int max_iter, GenericColMatrix &EV, bool reorthogonalize = false );

#if defined( USE_EIGEN )
#include "lanczos_correlation_eigen.txx"
#else
#include "lanczos_correlation_minlin.txx"
#endif
