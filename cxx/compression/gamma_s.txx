/**
	   Update the gamma index array, using the optimization proposed by Horenko 
	 
	   @param[in]     X           Time series (MatrixXX)
	   @param[in]     theta       List of the fields to be extracted (MatrixXX)
	   @param[in]     TT          Eigenvectors (MatrixXX)
	   @param[out]    gamma_ind   new placement of state probabilities
	   @return                    void
	 */

void gamma_s( const MatrixXXrow &X, const MatrixXX &theta, const std::vector<MatrixXX> TT, std::vector<int> &gamma_ind )
{
  const int K  = theta.cols();
  const int Ntl = X.rows();
  const int nl = X.cols();

  MatrixXX Xtranslated( Ntl, nl ) ;
  MatrixXX colnorm( nl, K ) ;

// This loop can be multithreaded, if all threads have a separate copy of Xtranslated
  for(int k = 0; k < K; k++) {
// This loop can be multithreaded!
    for(int l = 0; l < nl; l++) { Xtranslated.col(l) = X.col(l) - theta.col(k); }  // translate each column of X
    Xtranslated -= TT[k] * ( TT[k].transpose() * Xtranslated );
// This loop can be multithreaded!
    for(int l = 0; l < nl; l++) { colnorm(l,k) = Xtranslated.col(l).norm(); }  // translate each column of X
  }

//   std::cout << "row 1 of colnorm is " << colnorm.row(1) << " rows " << colnorm.rows() << " cols " << colnorm.cols() << std::endl;
// This loop can be multithreaded!
  for(int l = 0; l < nl; l++) {   // Pity that we have to loop through all rows
    std::ptrdiff_t i;
    ScalarType value = colnorm.row(l).minCoeff(&i);
    gamma_ind[l] = i;   // Only interested in position, not value
  }
}
