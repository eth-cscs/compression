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
void theta_s(const ArrayX1i gamma_ind, const MatrixXX X, MatrixXX theta )
{
  const int nl = X.cols();

  ScalarType sum_gamma;   // Number of entries containing each index
  // This loop is parallel: No dependencies between the columns
  for(int k = 0; k < (theta).cols(); k++)
  { 
    theta.setZero();  // Sums should start from zero
    sum_gamma = static_cast<ScalarType> ((gamma_ind == k).count());
    std::vector<int> found_items = find( gamma_ind, k );
    if ( sum_gamma > 0 )
    {
      for (int m = 0; m < found_items.size() ; m++ ) 
      {
         theta.col(k) += X.col(found_items[m]); 
      }
      theta.col(k) /= sum_gamma;
    }
  }
}
