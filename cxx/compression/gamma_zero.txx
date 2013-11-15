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

ArrayX1i gamma_zero(const int Ntl, const int K )
{
  ArrayX1i random_vector(Ntl);

  std::uniform_int_distribution<int> distribution(0,K-1);
  //
  std::default_random_engine engine;
  auto generator = std::bind(distribution, engine);
  std::generate_n(random_vector.data(), Ntl, generator); 

  return random_vector;
}
