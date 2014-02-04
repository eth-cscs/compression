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
void lanczos_correlation(const MatrixXX &Xtranslated, const int ne, const ScalarType tol, const int max_iter, MatrixXX &EV, bool reorthogonalize=false)
{
  int N = Xtranslated.rows();
  ScalarType gamma, delta;

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  MatrixXX V  = MatrixXX::Zero(N,max_iter);  // transformation
  MatrixXX Trid  = MatrixXX::Zero(max_iter,max_iter);  // Tridiagonal

  V.col(0).setOnes();     // Simple initial vector; no apparent side effects
/*  
  V.col(0).setRandom();   // Warning:  all nodes must generate same random vector in parallel (utilize the seed; but how is this done in Eigen?)
*/
  V.col(0) /= V.col(0).norm();    // Unit vector

  // find product w=A*V(:,0)
  VectorX w(N);
  VectorX tmp_vector = Xtranslated*(Xtranslated.transpose()*V.col(0));  // order important! evaluate right to left to save calculation!
  MPI_Allreduce( tmp_vector.data(), w.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


  delta = w.transpose() * V.col(0);
  Trid(0,0) = delta;  // store in tridiagonal matrix

  // preallocate storage vectors
  VectorX r(N);   // residual, temporary

  ScalarType convergence_error;
  int iter;

/*
  for(iter=1; iter < max_iter; iter++)   // main loop (sequential), will terminate earlier if tolerance is reached
  {
    VectorX global_vector( N );

    // The Allreduce here implies that w.data will not be reproducible over all PE configurations.  
    // a reproducible variant should be provided
    MPI_Allreduce( w.data(), global_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

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

    w = Xtranslated*(Xtranslated.transpose()*V.col(iter));  // order important! evaluate right to left to save calculation!
    Trid(iter-1,iter-1) = delta;  Trid(iter-1,iter) = gamma;  Trid(iter,iter-1) = gamma;
    SelfAdjointEigenSolver<MatrixXX> eigensolver(Trid.block(0,0,iter+1,iter+1));
    if (eigensolver.info() != Success) abort();

    VectorX  eigs = eigensolver.eigenvalues();  // Ritz values, sorted (largest eigenvalues last)
    MatrixXX UT = eigensolver.eigenvectors();   // Ritz vectors

    if ( iter >= ne ) {
      ScalarType max_err = 0.0;
      VectorX local_vector( N );
      EV = V.block(0,0,N,iter+1)*UT.block(0,iter-ne+1,iter+1,ne);  // Eigenvector approximations
      for (int count = 0; count < ne; count++) {      // Go through the Ritz values of interest
        ScalarType this_eig = eigs(count+iter-ne+1);  // Now determine the associated relative error
        local_vector = Xtranslated*(Xtranslated.transpose()*EV.col(count));
        // This communication is unfortunate; we could avoid this by doing max_iter iterations...
        // The Allreduce here implies that global_vector will not be reproducible over all PE configurations.  
        // a reproducible variant should be provided
        MPI_Allreduce( local_vector.data(), global_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
	max_err = max( abs(((global_vector - EV.col(count)*this_eig).norm())/this_eig), max_err);
      }
      convergence_error = max_err;
      if ( max_err < tol ) {
	break;
      }
    }
  }
  cout << "The maximum error is : " << convergence_error << " after " << iter << " iterations" << endl;
*/

    // main loop, will terminate earlier if tolerance is reached
    bool converged = false;
    for(int j=1; j<max_iter && !converged; ++j) {
        ////// timing logic //////
        //double time = -omp_get_wtime();
        //////////////////////////
        // std::cout << "================= ITERATION " << j << "    " << std::endl;

	// The Allreduce here implies that w.data will not be reproducible over all PE configurations.  
	// a reproducible variant should be provided

        if ( j == 1 )
            w -= delta*V.col(j-1);
        else
            w -= delta*V.col(j-1) + gamma*V.col(j-2);

        gamma = w.norm();
        V.col(j) = (1./gamma)*w;

        // reorthogonalize
        if( reorthogonalize ) {
            for( int jj = 0; jj < j; ++jj )  {
	      ScalarType alpha =  V.col(jj).transpose() * V.col(j) ;
                V.col(j) -= alpha * V.col(jj);
            }
        }

        // write off-diagonal values in tri-diagonal matrix
        Trid(j-1,j  ) = gamma;
        Trid(j  ,j-1) = gamma;

        // find matrix-vector product for next iteration
        r = Xtranslated*(Xtranslated.transpose()*V.col(j));
        MPI_Allreduce( r.data(), w.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

        // update diagonal of tridiagonal system
        delta = w.transpose() * V.col(j);
        Trid(j, j) = delta;
        if ( j >= ne ) {
            // find eigenvectors/eigenvalues for the reduced triangular system
#ifdef EIGEN_EIGENSOLVE
	    SelfAdjointEigenSolver<MatrixXX> eigensolver(Trid.block(0,0,j+1,j+1));
	    if (eigensolver.info() != Success) abort();
	    VectorX  eigs = eigensolver.eigenvalues().block(j+1-ne,0,ne,1);  // ne largest Ritz values, sorted ascending
	    MatrixXX UT = eigensolver.eigenvectors();   // Ritz vectors
            // std::cout << "iteration : " << j << ", Tblock : " << Trid.block(0,0,j+1,j+1) << std::endl;
            // std::cout << "iteration : " << j << ", ritz values " << eigs << std::endl;
            // std::cout << "iteration : " << j << ", ritz vectors " << UT << std::endl;
            // j or j+1 ??
	    EV = V.block(0,0,N,j+1)*UT.block(0,j+1-ne,j+1,ne);  // Eigenvector approximations for largest ne eigenvalues
#else
            MatrixXX Tsub = Trid.block(0,0,j+1,j+1);
	    VectorX  eigs(j+1);
            MatrixXX UT(j+1,ne);
            assert( steigs( Tsub.data(), UT.data(), eigs.data(), j+1, ne) );
            EV = V.block(0,0,N,j+1)*UT.block(0,0,j+1,ne);
#endif


            // copy eigenvectors for reduced system to the device
            ////////////////////////////////////////////////////////////////////
            // TODO : can we find a way to allocate memory for UV outside the
            //        inner loop? this memory allocation is probably killing us
            //        particularly if we go to large subspace sizes
            //
            ////////////////////////////////////////////////////////////////////

            ScalarType max_err = 0.;
            const int boundary = j+1-ne;
            for(int count=ne-1; count>=0 && !converged; count--){
                ScalarType this_eig = eigs(count);
                // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;

                // find the residual
                // r = Xtranslated*( Xtranslated.transpose() * EV.col(count) ) - this_eig*EV.col(count);
		r = Xtranslated*( Xtranslated.transpose() * EV.col(count) );

		// Global summation or matrix product
	        MPI_Allreduce( r.data(), tmp_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
                // compute the relative error from the residual
                r = tmp_vector - EV.col(count)*this_eig;   //residual
                ScalarType this_err = std::abs( (r.norm()) / this_eig );
		max_err = std::max(max_err, this_err);
                // terminate early if the current error exceeds the tolerance
                if(max_err > tol)
                    break;
            } // end-for error estimation
            // std::cout << "iteration : " << j << ", max_err : " << max_err << std::endl;
            // test for convergence
            if(max_err < tol) {
	      converged = true;
	    }
        } // end-if estimate eigenvalues
        ////// timing logic //////
        //time += omp_get_wtime();
        //std::cout << "took " << time*1000. << " miliseconds" << std::endl;
        //////////////////////////
    } // end-for main

    // return failure if no convergence

/*
    if(!converged)
      return false;
    else
      return true;
*/

}
