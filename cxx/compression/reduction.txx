

/**
	   Calculate the reduced field (in distributed form ) for each k
	 
	   @param[in]     gamma_ind   placement of state probabilities
	   @param[in]     X           Time series (GenericRowMatrix)
	   @param[in]     theta       List of the fields to be extracted (GenericColMatrix)
	   @param[in]     EOFs        Final EOFs (Array of GenericColMatrix)
	   @return                    reduced X (std::vector<GenericColMatrix>)
	 */

template <typename ScalarType>
void reduction( const std::vector<int> &gamma_ind, const std::vector<GenericColMatrix> &EOFs, const GenericRowMatrix &X, const GenericColMatrix &theta, std::vector<GenericColMatrix> &Xreduced)
{
  const int Ntl = theta.rows();
  const int K   = theta.cols();
  const int nl  = gamma_ind.size();

  GenericColMatrix Xtranslated( Ntl, nl);    // For all the partition subsets (better here than in loop)
  GenericVector  colnorm( nl ) ;
  GenericVector  XminusTheta( Ntl );

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );

    for (int m = 0; m < Nonzeros.size() ; m++ ) {       // Translate X columns with mean value at new origin
#if defined( USE_EIGEN )      
      Xtranslated.col(Nonzeros[m]) = X.col(Nonzeros[m]) - theta.col(k);                     // Translated to theta_k 
      Xreduced[k].col(Nonzeros[m]) =  EOFs[k].transpose() * Xtranslated.col(Nonzeros[m]);
#elif defined( USE_MINLIN )
      Xtranslated(all,Nonzeros[m]) = X(all,Nonzeros[m]) - theta(all,k);                     // Translated to theta_k 
      Xreduced[k](all,Nonzeros[m]) = transpose(EOFs[k]) * Xtranslated(all,Nonzeros[m]);
#else
      ERROR:  must USE_EIGEN or USE_MINLIN
#endif
    }
  }
}

