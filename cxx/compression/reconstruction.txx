/**
	   Calculate the reduced field (in distributed form ) for each k
           
           Note: The matrix Xreduced is not changed in this function but we cannot pass 
           it as const as minlin cannot do y = A*B(all,j) if B is const.
	 
	   @param[in]     gamma_ind       placement of state probabilities
	   @param[in]     EOFs            Final EOFs (Array of GenericColMatrix)
	   @param[in]     theta           List of the fields to be extracted (GenericColMatrix)
	   @param[in]     Xreduced        Reduced representation of time series (std::vector<GenericColMatrix>)
	   @param[out]    Xreconstructed  Reconstructed time series (GenericRowMatrix)
	 */

template <typename ScalarType>
void reconstruction( const std::vector<int> &gamma_ind, const std::vector<GenericColMatrix> &EOFs, const GenericColMatrix &theta, std::vector<GenericColMatrix> &Xreduced, GenericRowMatrix &Xreconstructed )
{
  const int Ntl = theta.rows();
  const int K   = theta.cols();

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );
    for (int m = 0; m < Nonzeros.size() ; m++ ) {       // Translate X columns with mean value at new origin
#if defined( USE_EIGEN )
      Xreconstructed.col(Nonzeros[m]) = EOFs[k] * Xreduced[k].col(Nonzeros[m]) + theta.col(k);          // Translated back to theta_k 
#elif defined( USE_MINLIN )
      Xreconstructed(all,Nonzeros[m]) = EOFs[k] * Xreduced[k](all,Nonzeros[m]);
      Xreconstructed(all,Nonzeros[m]) += theta(all,k);
 #else
      ERROR:  must USE_EIGEN or USE_MINLIN
#endif
    }
  }
}
