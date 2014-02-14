

/**
	   Update the gamma index array, using the optimization proposed by Horenko 
	 
	   @param[in]     gamma_ind   placement of state probabilities
	   @param[in]     X           Time series (GenericRowMatrix)
	   @param[in]     theta       List of the fields to be extracted (GenericColMatrix)
	   @param[in]     EOFs        Final EOFs (Array of GenericColMatrix)
	   @return                    Norm (ScalarType)
	 */

ScalarType L_value( const std::vector<int> &gamma_ind, const std::vector<GenericColMatrix> EOFs, const GenericRowMatrix &X, const GenericColMatrix &theta )
{
  ScalarType value  = 0.0;
  ScalarType output = 0.0;

  const int Ntl = X.rows();
  const int nl  = X.cols();
  const int K   = theta.cols();

  GenericColMatrix Xtranslated( Ntl, nl);    // For all the partition subsets (better here than in loop)
  GenericColMatrix Junk( Ntl, 1);            // For all the partition subsets (better here than in loop)
  GenericVector  colnorm( nl ) ;
  GenericVector  XminusTheta( Ntl );

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );

    for (int m = 0; m < Nonzeros.size() ; m++ ) {       // Translate X columns with mean value at new origin
#if defined( USE_EIGEN )      
      Xtranslated.col(Nonzeros[m])  = X.col(Nonzeros[m]) - theta.col(k);                     // Translated to theta_k 
      Xtranslated.col(Nonzeros[m]) -=  EOFs[k] * ( EOFs[k].transpose() * Xtranslated.col(Nonzeros[m]) );
#elif defined( USE_MINLIN )
      Xtranslated(all,Nonzeros[m])  = X(all,Nonzeros[m]) - theta(all,k);                     // Translated to theta_k 
// Quick hack to use CUBLAS; this functionality should be integrated into MINLIN     Should be:
//     Xtranslated(all,Nonzeros[m]) -=  EOFs[k] * ( transpose(EOFs[k]) * Xtranslated(all,Nonzeros[m]) ); 
      GenericVector tmp( EOFs[k].cols() );
      gemv_wrapper(tmp.pointer(), Xtranslated.pointer() + Ntl*Nonzeros[m], EOFs[k], 1., 0., 'T');
      gemv_wrapper(Xtranslated.pointer() + Ntl*Nonzeros[m], tmp.pointer(), EOFs[k], -1., 1., 'N');
#else
      ERROR:  must USE_EIGEN or USE_MINLIN
#endif
    }
  }
  // Now Xtranslated contains the column vectors for all l in 0..nl-1; the square norms just need to be summed
  for (int l = 0; l < nl; l++ ) { 
#if defined( USE_EIGEN )
    ScalarType colnorm = Xtranslated.col(l).norm(); value += colnorm*colnorm;
#elif defined( USE_MINLIN )
    ScalarType colnorm = norm(Xtranslated(all,l)); value += colnorm*colnorm;
#else
    ERROR:  must USE_EIGEN or USE_MINLIN
#endif
 }  
  MPI_Allreduce( &value, &output, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return output;
}

