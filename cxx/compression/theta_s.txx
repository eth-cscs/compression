/**
	   Calculate the matrix of time series means for each K
	   as described in algorithm documentation.  
	   The algorithm first finds the number of columns associated with K.
	   
	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     gamma_ind   Indices where all gamma state probability is placed (0..K-1)
	   @param[in]     X           Time series
	   @param[in]     K           Number of possible states
	   @param[out]    theta	      Matrix of means for each K   
	   @return                    void
	 */

template <typename ScalarType>
void theta_s(const ArrayX1i &gamma_ind, const MatrixXX &X, MatrixXX &theta )
{
  const int K = theta.cols();

  ScalarType sum_gamma;   // Number of entries containing each index
  // This loop is parallel: No dependencies between the columns
  for(int k = 0; k < K; k++) { 
    std::vector<int> Nonzeros = find( gamma_ind, k );
    sum_gamma = static_cast<ScalarType> (Nonzeros.size());
    std::cout << "k " << k << " nonzeros " << sum_gamma << std::endl;
    if ( sum_gamma > 0 ) {
      theta.col(k) = X.col(Nonzeros[0]); 
      for (int m = 1; m < Nonzeros.size() ; m++ ) {
        theta.col(k) += X.col(Nonzeros[m]); 
      }
      theta.col(k) /= sum_gamma;
      std::cout << "Norm of theta column(" << k << ")= " << theta.col(k).norm() << std::endl;
    }
  }
}
