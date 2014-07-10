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
void theta_s(const std::vector<int> &gamma_ind, const GenericRowMatrix &X, GenericColMatrix &theta )
{
  const int Ntl = theta.rows();
  const int K   = theta.cols();
  GenericColMatrix local_theta(Ntl,K);
  std::vector<int> local_nbr_nonzeros( K ), global_nbr_nonzeros( K );

  // This loop is parallel: No dependencies between the columns  (local_vector needs to be private)
  for(int k = 0; k < K; k++) { 
    std::vector<int> Nonzeros = find( gamma_ind, k );
    local_nbr_nonzeros[k] = Nonzeros.size();
  }   
  MPI_Allreduce( local_nbr_nonzeros.data(), global_nbr_nonzeros.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

  // This loop is parallel: No dependencies between the columns  (local_vector needs to be private)
  for(int k = 0; k < K; k++) { 
    std::vector<int> Nonzeros = find( gamma_ind, k );   // Could use a matrix for this to avoid 2nd load;
    ScalarType sum_gamma;   // Number of entries containing each index
    sum_gamma = static_cast<ScalarType> (global_nbr_nonzeros[k]);
    // std::cout << " k " << k << " nonzeros " << sum_gamma << std::endl;
    if ( sum_gamma > 0 ) {
      //  std::cout << "Norm of first X column " << X.col(Nonzeros[0]).norm() << std::endl;
#if defined( USE_EIGEN )
      local_theta.col(k) =  Eigen::MatrixXd::Zero(Ntl, 1);
      for (int m = 0; m < Nonzeros.size() ; m++ ) {
        local_theta.col(k) += X.col(Nonzeros[m]);  
      }
      local_theta.col(k) /= sum_gamma;
#elif defined( USE_MINLIN )
      local_theta(all,k) =  0.;
      for (int m = 0; m < Nonzeros.size() ; m++ ) {
        local_theta(all,k) += X(all,Nonzeros[m]);  
      }
      local_theta(all,k) /= sum_gamma;
#else
      ERROR:  must USE_EIGEN or USE_MINLIN
#endif
    }
  }
#if defined( USE_EIGEN )
  MPI_Allreduce( local_theta.data(), theta.data(), Ntl*K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
#elif defined( USE_MINLIN )
  MPI_Allreduce( local_theta.pointer(), theta.pointer(), Ntl*K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
#else
  ERROR:  must USE_EIGEN or USE_MINLIN
#endif
}
