/**
	   Update the gamma index array, using the optimization proposed by Horenko 
	 
	   @param[in]     gamma_ind   placement of state probabilities
	   @param[in]     X           Time series (MatrixXX)
	   @param[in]     theta       List of the fields to be extracted (MatrixXX)
	   @param[in]     EOFs        Final EOFs (Array of MatrixXX)
	   @return                    Norm (ScalarType)
	 */

ScalarType L_value( const ArrayX1i &gamma_ind, const MatrixXX *EOFs, const MatrixXXrow &X, const MatrixXX &theta )
{
  ScalarType value  = 0.0;
  ScalarType output = 0.0;

  const int Ntl = X.rows();
  const int nl  = X.cols();
  const int K   = theta.cols();

  MatrixXX Xtranslated( Ntl, nl);    // For all the partition subsets (better here than in loop)
  VectorX  colnorm( nl ) ;
  VectorX  XminusTheta( Ntl );

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );

    for (int m = 0; m < Nonzeros.size() ; m++ ) {       // Translate X columns with mean value at new origin
      
      Xtranslated.col(Nonzeros[m])  = X.col(Nonzeros[m]) - theta.col(k);                     // Translated to theta_k 
      Xtranslated.col(Nonzeros[m]) -=  EOFs[k] * ( EOFs[k].transpose() * Xtranslated.col(Nonzeros[m]) ); 
    }
  }
  // Now Xtranslated contains the column vectors for all l in 0..nl-1; the square norms just need to be summed
  for (int l = 0; l < nl; l++ ) { ScalarType colnorm = Xtranslated.col(l).norm(); value += colnorm*colnorm; }  
  MPI_Allreduce( &value, &output, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return output;
}

