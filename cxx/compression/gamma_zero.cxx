/**
	   Initialize gamma.  In this case there is a non-random step pattern:
           
	   @param[in]     nl          Number of local points in horizontal (int)
	   @param[in]     K           Subspace size
	   @return                    Array containing indices of probability 1
	 */

ArrayX1i gamma_zero(const int nl, const int K )
{
  ArrayX1i random_vector(nl);

  for ( int l=0; l < nl ; l++ ) {
    random_vector(l) = l%K;
  }

  //  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");
  //  std::cout << "First 25 random numbers are " << std::endl 
  //  << random_vector.block(0,0,25,1).format(CommaInitFmt) << std::endl;

  return random_vector;
}
