/**
	   This template performs the Lanczos algorithm on a correlation
	   matrix defined through the "Xtranslated" set of observations.
           In this case only the eigenvectors are needed

	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     filename    Filename (string)
	   @param[in]     fields      List of the fields to be extracted (vector of strings0
	   @return                    vector of concatenated field values
	 */

template <typename ScalarType>
void lanczos_correlation(const MatrixXX &Xtranslated, const int nbr_eig, const ScalarType tol, const int max_iter, MatrixXX &EV)
{
  int nrows = Xtranslated.rows();
  ScalarType gamma, delta;
  
  MatrixXX V  = MatrixXX::Zero(nrows,max_iter);  // transformation
  MatrixXX Trid  = MatrixXX::Zero(max_iter,max_iter);  // Tridiagonal

  V.col(0).setOnes();     // Simple initial vector; no apparent side effects
/*  
  V.col(0).setRandom();   // Warning:  all nodes must generate same random vector in parallel (utilize the seed; but how is this done in Eigen?)
*/
  V.col(0) /= V.col(0).norm();    // Unit vector

  VectorX AV = Xtranslated*(Xtranslated.transpose()*V.col(0));  // order important! evaluate right to left to save calculation!

  ScalarType convergence_error;
  int iter;

  for(iter=1; iter < max_iter; iter++)   // main loop (sequential), will terminate earlier if tolerance is reached
  {
    VectorX global_vector( nrows );

    // The Allreduce here implies that AV.data will not be reproducible over all PE configurations.  
    // a reproducible variant should be provided
    MPI_Allreduce( AV.data(), global_vector.data(), nrows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    if ( iter == 1 ) { V.col(iter) = global_vector; }
    else {  V.col(iter) = global_vector - gamma*V.col(iter-2); }

    delta = V.col(iter).transpose() * V.col(iter-1);
    V.col(iter) -= delta*V.col(iter-1);

    gamma  = V.col(iter).norm();
    V.col(iter) /= gamma;
    // cout << "gamma " << gamma << " delta " << delta << " should be the same on all PEs" << endl;

    // reorthogonalize  -- this can be an argument-driven option   OpenMP: no dependencies
    for( int sub_iter = 0; sub_iter < iter; sub_iter++ )  {
      ScalarType crap = V.col(iter).transpose()*V.col(sub_iter);
      V.col(iter) -= crap*V.col(sub_iter);
      //      V.col(iter) -= (V.col(iter).transpose()*V.col(sub_iter))*V.col(sub_iter);  // run-time crash... Why?
    }

    AV = Xtranslated*(Xtranslated.transpose()*V.col(iter));  // order important! evaluate right to left to save calculation!
    Trid(iter-1,iter-1) = delta;  Trid(iter-1,iter) = gamma;  Trid(iter,iter-1) = gamma;
    SelfAdjointEigenSolver<MatrixXX> eigensolver(Trid.block(0,0,iter+1,iter+1));
    if (eigensolver.info() != Success) abort();

    VectorX  eigs = eigensolver.eigenvalues();  // Ritz values, sorted (largest eigenvalues last)
    MatrixXX UT = eigensolver.eigenvectors();   // Ritz vectors

    if ( iter >= nbr_eig ) {
      ScalarType max_err = 0.0;
      VectorX local_vector( nrows );
      EV = V.block(0,0,nrows,iter+1)*UT.block(0,iter-nbr_eig+1,iter+1,nbr_eig);  // Eigenvector approximations
      for (int count = 0; count < nbr_eig; count++) {      // Go through the Ritz values of interest
        ScalarType this_eig = eigs(count+iter-nbr_eig+1);  // Now determine the associated relative error
        local_vector = Xtranslated*(Xtranslated.transpose()*EV.col(count));
        // This communication is unfortunate; we could avoid this by doing max_iter iterations...
        // The Allreduce here implies that global_vector will not be reproducible over all PE configurations.  
        // a reproducible variant should be provided
        MPI_Allreduce( local_vector.data(), global_vector.data(), nrows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
	max_err = max( abs(((global_vector - EV.col(count)*this_eig).norm())/this_eig), max_err);
      }
      convergence_error = max_err;
      if ( max_err < tol ) {
	break;
      }
    }
  }
  cout << "The maximum error is : " << convergence_error << " after " << iter << " iterations" << endl;

}
