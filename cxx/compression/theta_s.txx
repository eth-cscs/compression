/**
	   Write description of function here.
	   The function should follow these comments.
	   Use of "brief" tag is optional. (no point to it)
	   
	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     filename    Filename (string)
	   @param[in]     fields      List of the fields to be extracted (vector of strings0
	   @return                    vector of concatenated field values
	 */

template <typename ScalarType>
Matrix<ScalarType, Dynamic, Dynamic> theta_s(const ArrayX1i gamma_ind, const Matrix<ScalarType, Dynamic, Dynamic> X, const int K)
{
  const int nj = X.cols();

  ArrayX1i sum_gamma(K);   // Number of entries containing each index
  for(int k = 0; k < K; k++) 
  { sum_gamma[k] = (gamma_ind == k).count();   
    std::cout << "sum_gamma[ " << k << "] = " << sum_gamma[k] << std::endl;
    }

  Matrix<ScalarType,Dynamic,Dynamic> theta_update( X.rows(), K) ;

/*  
  MatrixX colnorm( X.cols(), K ) ;

  for(int i = 0; i < K; i++)
    {
// This loop can be multithreaded!
      for(int j = 0; j < nj; j++) { Xtranslated.col(j) = X.col(j) - TT.col(i); }  // translate each column of X
      Xtranslated -= TT.col(i) * ( TT.transpose().row(i) * Xtranslated );
      for(int j = 0; j < nj; j++) { colnorm.col(j) = Xtr.col(j).norm(); }  // translate each column of X
    }
  // [val GammaInd]=min(colnorm,[],2);
*/

  return theta_update;
}
