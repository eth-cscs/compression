/**
	   Update the gamma index array, using the optimization proposed by Horenko 
	 
	   @param[in]     X           Time series (MatrixXX)
	   @param[in]     theta       List of the fields to be extracted (MatrixXX)
	   @param[in]     TT          Eigenvectors (MatrixXX)
	   @param[out]    gamma_ind   new placement of state probabilities
	   @return                    void
	 */

void gamma_s( const GenericRowMatrix &X, const GenericColMatrix &theta, const std::vector<GenericColMatrix> TT, std::vector<int> &gamma_ind )
{
  const int K   = theta.cols();
  const int Ntl = X.rows();
  const int nl  = X.cols();

  std::vector<ScalarType> gamma_min(nl, std::numeric_limits<ScalarType>::max());

  GenericColMatrix Xtranslated( Ntl, nl ) ;
  GenericColMatrix colnorm( nl, K ) ;

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
// This loop can be multithreaded!
    for(int l = 0; l < nl; l++) { GET_COLUMN(Xtranslated,l) = GET_COLUMN(X,l) - GET_COLUMN(theta,k); }  // translate each column of X
#if defined( USE_EIGEN )
    Xtranslated -= TT[k] * ( TT[k].transpose() * Xtranslated );
#elif defined( USE_MINLIN )
    {
      GenericVector tmp_nl(nl);
      tmp_nl(all) = transpose(Xtranslated) * TT[k];
      geru_wrapper( Xtranslated, TT[k].pointer(), tmp_nl.pointer(), -1.);
    }
#else
    ERROR:  -DUSE_EIGEN or -DUSE_MINLIN
#endif
    for(int l = 0; l < nl; l++) {
      ScalarType this_min = NORM( GET_COLUMN(Xtranslated,l) );  // translate each column of X
      if(this_min<gamma_min[l])
      {
        gamma_min[l] = this_min;
        gamma_ind[l] = k;
      }
    }
  }
}
