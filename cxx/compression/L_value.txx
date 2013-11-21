/**
	   Update the gamma index array, using the optimization proposed by Horenko 
	 
	   @param[in]     gamma_ind   placement of state probabilities
	   @param[in]     X           Time series (MatrixXX)
	   @param[in]     theta       List of the fields to be extracted (MatrixXX)
	   @param[in]     TT          Eigenvectors (MatrixXX)
	   @return                    Norm (ScalarType)
	 */

ScalarType L_value( const ArrayX1i &gamma_ind, const MatrixXX &TT, const MatrixXX &X, const MatrixXX &theta )
{
  ScalarType output = 0.0;

  const int Ntl = X.rows();
  const int nl  = X.cols();
  const int K   = theta.cols();

  VectorX  colnorm( nl ) ;

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
    std::vector<int> Nonzeros = find( gamma_ind, k );

    MatrixXX Xtranslated(X.rows(),Nonzeros.size());    // Partition subset for this K
    for (int m = 0; m < Nonzeros.size() ; m++ ) {       // Translate X columns with mean value at new origin
      Xtranslated.col(m) = X.col(Nonzeros[m]) - theta.col(k);  // bsxfun(@minus,X(:,Nonzeros),Theta(:,k))
    }
    Xtranslated -= TT.col(k) * ( TT.col(k).transpose() * Xtranslated );
    for (int m = 0; m < Nonzeros.size(); m++) { output += Xtranslated.col(m).norm(); }  // translate each column of X
  }

  return output;
}

