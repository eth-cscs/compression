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

ArrayX1i gamma_zero(const int nl, const int K )
{
  ArrayX1i random_vector(nl);

  std::uniform_int_distribution<int> distribution(0,K-1);
  //
  std::default_random_engine engine;
  auto generator = std::bind(distribution, engine);
  std::generate_n(random_vector.data(), nl, generator); 

  //  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");
  //  std::cout << "First 25 random numbers are " << std::endl 
  //  << random_vector.block(0,0,25,1).format(CommaInitFmt) << std::endl;

  return random_vector;
}
