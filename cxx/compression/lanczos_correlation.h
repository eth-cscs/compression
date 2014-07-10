#include "generic_wrappers.h"
#if defined( USE_MINLIN )
#include "minlin_wrappers.h"
#endif

template <typename ScalarType>
bool lanczos_correlation(const GenericColMatrix &Xtranslated, const int ne, const ScalarType tol, const int max_iter, GenericColMatrix &EV, bool reorthogonalize = false );

#include "lanczos_correlation.txx"
