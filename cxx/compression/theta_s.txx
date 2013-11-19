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
  const int nl = X.cols();

  Matrix<ScalarType, Dynamic, Dynamic> output = Matrix<ScalarType, Dynamic, Dynamic>::Zero(X.rows(),K); 
  ScalarType sum_gamma;   // Number of entries containing each index
  // This loop is parallel: No dependencies between the columns
  for(int k = 0; k < K; k++)
  { 
    sum_gamma = static_cast<ScalarType> ((gamma_ind == k).count());
    std::vector<int> found_items = find( gamma_ind, k );
    if ( sum_gamma > 0 )
    {
      for (int m = 0; m < found_items.size() ; m++ ) 
      {
         output.col(k) += X.col(found_items[m]); 
      }
      output.col(k) /= sum_gamma;
    }
  }

  return output;
}
