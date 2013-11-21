/**
	   This function roughly corresponds to the MATLAB find.  
	   It searches through int_vector to find the indices where
	   the entries are equal to input k.   It returns an index set as 
	   a std::vector<int>.

	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     int_vector    Vector of integer values (Eigen Array)
	   @param[in]     k             Integer of interest in vector
	   @return                      Index set of positions with value k (std::vector)
	 */

//Array<int, Dynamic, 1> find(const Array<int, Dynamic, 1> int_vector, const int k)
std::vector<int> find(const Array<int, Dynamic, 1> int_vector, const int k)
{
  std::vector<int> found_items;
  for (int i=0; i<int_vector.rows(); i++) {
    if (int_vector[i] == k) { found_items.push_back(i); }
  }
  return found_items;
/*
  // John wants to clean up this version sometime.
  std::for_each(gamma_ind.data(), gamma_ind.data()+gamma_ind.rows(),

               [&match, &found_items, &gamma_ind](int *entry) mutable
               {
                 if( *entry == match)
                 { 
                   found_items.push_back(static_cast<int>(entry-gamma_ind.data())); 
                 }
               } );
*/

//  Map<Array<int, Dynamic, 1>> output( found_items.data(), found_items.size(), 1);
//  return output;
}
