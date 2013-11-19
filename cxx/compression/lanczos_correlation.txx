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
Matrix<ScalarType, Dynamic, Dynamic> lanczos_correlation(const Matrix<ScalarType, Dynamic, Dynamic> Xtranslated, const int nbr_eig, const ScalarType tol, const int max_iter)
{
  typedef Matrix<ScalarType, Dynamic, Dynamic> MatrixXX;
  typedef Matrix<ScalarType, Dynamic, 1> VectorX;

  int nrows = Xtranslated.rows();
  ScalarType gamma, delta;
  
  MatrixXX EV = MatrixXX::Zero(nrows,nbr_eig);   // output
  MatrixXX V  = MatrixXX::Zero(nrows,max_iter);  // transformation
  MatrixXX Trid  = MatrixXX::Zero(max_iter,max_iter);  // Tridiagonal
  
  V.col(0).setRandom();   // Warning:  all nodes must generate same random vector in parallel (utilize the seed; but how is this done in Eigen?)
  V.col(0) /= V.col(0).norm();    // Unit vector

  MatrixXX AV = Xtranslated*(Xtranslated.transpose()*V.col(0));  // order important! evaluate right to left to save calculation!
  
  for(int iter=1; iter < max_iter; iter++)   // main loop, will terminate earlier if tolerance is reached
  {
    if ( iter == 1 ) { V.col(1) = AV; }
    else {  V.col(iter) = AV - gamma*V.col(iter-2); }

    delta = V.col(iter).transpose() * V.col(iter-1);
    V.col(iter) -= delta*V.col(iter-1);

    gamma  = V.col(iter).norm();
    V.col(iter) /= gamma;

    // reorthogonalize  -- this can be an argument-driven option
    for( int sub_iter = 0; sub_iter < iter; sub_iter++ )  {
      crap = V.col(iter).transpose()*V.col(sub_iter);
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
      EV = V.block(0,0,nrows,iter+1)*UT.block(0,iter-nbr_eig+1,iter+1,nbr_eig);  // Eigenvector approximations
      ScalarType max_err = 0.0;
      for (int count = 0; count < nbr_eig; count++) {      // Go through the Ritz values of interest
        ScalarType this_eig = eigs(count+iter-nbr_eig+1);  // Now determine the associated relative error
	max_err = max( abs(((Xtranslated*(Xtranslated.transpose()*EV.col(count)) - EV.col(count)*this_eig).norm())/this_eig), max_err);
	cout << " Ritz value of interest " << this_eig << endl; 
      }
      cout << "The maximum error is : " << max_err << endl;
      if ( max_err < tol ) break; 
    }

  }

  return EV;
}
