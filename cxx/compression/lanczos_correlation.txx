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
bool lanczos_correlation(const GenericColMatrix &Xtranslated, const int ne, const ScalarType tol, const int max_iter, GenericColMatrix &EV, bool reorthogonalize=false)
{
  int N = Xtranslated.rows();
  ScalarType gamma, delta;
  bool converged = false;

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  GenericColMatrix V(N,max_iter);  // transformation

  GenericVector w(N);

  // preallocate storage vectors
  GenericVector r(N);   // residual, temporary
  GenericVector tmp_vector(N);   // temporary

#if defined( USE_EIGEN )      
  //V.col(0).setOnes();     // Simple initial vector; no apparent side effects
  V.col(0).setRandom();     // Random initial vector, all nodes must generate same vector so use srand(RANDOM_SEED) in caller
  V.col(0) /= V.col(0).norm();    // Unit vector
  GenericColMatrix Trid = GenericColMatrix::Zero(max_iter,max_iter);  // Tridiagonal
  tmp_vector = Xtranslated*(Xtranslated.transpose()*V.col(0));  // order important! evaluate right to left to save calculation!
  MPI_Allreduce( tmp_vector.data(), w.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  delta = w.transpose() * V.col(0);
#elif defined( USE_MINLIN )
  GenericVector er(max_iter);        // ScalarType components of eigenvalues
  GenericVector ei(max_iter);        // imaginary components of eigenvalues
  V(all,0) = ScalarType(1.);
  V(all,0) /= norm(V(all,0));    // Unit vector
  GenericColMatrix Trid(max_iter,max_iter);  // Tridiagonal
  Trid(all) = 0.;                            // Matrix must be zeroed out
  {
    GenericVector tmp_ne(Xtranslated.cols());
    gemv_wrapper( tmp_ne.pointer(), V.pointer(), Xtranslated, 1., 0., 'T' );
    gemv_wrapper( tmp_vector.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
  }
  MPI_Allreduce( tmp_vector.pointer(), w.pointer(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  delta = dot(w, V(all,0));
#endif

  Trid(0,0) = delta;  // store in tridiagonal matrix

  ScalarType convergence_error;
  int iter;

    // main loop, will terminate earlier if tolerance is reached

  for(int j=1; j<max_iter && !converged; ++j) {
    ////// timing logic //////
    //double time = -omp_get_wtime();
    //////////////////////////
    // std::cout << "================= ITERATION " << j << "    " << std::endl;

    // The Allreduce here implies that w.data will not be reproducible over all PE configurations.  
    // a reproducible variant should be provided

    if ( j == 1 )
      w -= delta * GET_COLUMN(V,j-1) ;
    else
      w -= delta * GET_COLUMN(V,j-1) + gamma * GET_COLUMN(V,j-2);

    gamma = GET_NORM(w);
    GET_COLUMN(V,j) = (1./gamma)*w;

    // reorthogonalize
    if( reorthogonalize ) {
      for( int jj = 0; jj < j; ++jj )  {
        // ScalarType alpha =  V.col(jj).transpose() * V.col(j) ;
        ScalarType alpha =  DOT_PRODUCT( GET_COLUMN(V,jj), GET_COLUMN(V,j) ) ;
        GET_COLUMN(V,j) -= alpha * GET_COLUMN(V,jj);
      }
    }

    
    // write off-diagonal values in tri-diagonal matrix
    Trid(j-1,j  ) = gamma;
    Trid(j  ,j-1) = gamma;

    // find matrix-vector product for next iteration
#if defined( USE_EIGEN )
    r = Xtranslated*(Xtranslated.transpose()*V.col(j));
#elif defined( USE_MINLIN )
  {
    GenericVector tmp_ne(Xtranslated.cols());
    gemv_wrapper( tmp_ne.pointer(), V.pointer()+j*N, Xtranslated, 1., 0., 'T' );
    gemv_wrapper( r.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
  }    
#else
    ERROR:  -DUSE_EIGEN or -DUSE_MINLIN
#endif
    MPI_Allreduce( GET_POINTER(r), GET_POINTER(w), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    // update diagonal of tridiagonal system
    delta = DOT_PRODUCT( w, GET_COLUMN(V,j) );
    Trid(j, j) = delta;
    if ( j >= ne ) {
      // find eigenvectors/eigenvalues for the reduced triangular system
#if defined( USE_EIGEN )

#if defined( EIGEN_EIGENSOLVE )
      Eigen::SelfAdjointEigenSolver<GenericColMatrix> eigensolver(Trid.block(0,0,j+1,j+1));
      if (eigensolver.info() != Eigen::Success) abort();
      GenericVector  eigs = eigensolver.eigenvalues().block(j+1-ne,0,ne,1);  // ne largest Ritz values, sorted ascending
      GenericColMatrix UT = eigensolver.eigenvectors();   // Ritz vectors
      // std::cout << "iteration : " << j << ", Tblock : " << Trid.block(0,0,j+1,j+1) << std::endl;
      // std::cout << "iteration : " << j << ", ritz values " << eigs << std::endl;
      // std::cout << "iteration : " << j << ", ritz vectors " << UT << std::endl;
      // j or j+1 ??
      EV = V.block(0,0,N,j+1)*UT.block(0,j+1-ne,j+1,ne);  // Eigenvector approximations for largest ne eigenvalues
#else
      GenericColMatrix Tsub = Trid.block(0,0,j+1,j+1);
      GenericColMatrix UT(j+1,ne);
      GenericVector  eigs(j+1);
      // TODO:  ensure that eigenvalues have ascending order
      assert( steigs( Tsub.data(), UT.data(), eigs.data(), j+1, ne) );
      EV = V.block(0,0,N,j+1)*UT.block(0,0,j+1,ne);
#endif

#elif defined( USE_MINLIN )

      GenericVector  eigs(ne);
      {
        HostMatrix<ScalarType> Tsub = Trid(0,j,0,j);
        HostMatrix<ScalarType> UVhost(j+1,ne);
#ifdef FULL_EIGENSOLVE
        assert( geev(Tsub.pointer(), UVhost.pointer(), er.pointer(), ei.pointer(), ne) );
#else
        std::cout << "j+1 " << j+1 << " ne " << ne << std::endl;
        assert( steigs( Tsub.pointer(), UVhost.pointer(), er.pointer(), j+1, ne) );
#endif
        // copy eigenvectors for reduced system to the device
        GenericColMatrix UV = UVhost;

        // find approximate eigenvectors of full system
        EV = V(all,0,j)*UV;
      }

#else
      ERROR:  -DUSE_EIGEN or -DUSE_MINLIN
#endif


      // copy eigenvectors for reduced system to the device
      ////////////////////////////////////////////////////////////////////
      // TODO : can we find a way to allocate memory for UV outside the
      //        inner loop? this memory allocation is probably killing us
      //        particularly if we go to large subspace sizes
      //
      ////////////////////////////////////////////////////////////////////

      ScalarType max_err = 0.;

#if defined( USE_EIGEN )
        // TODO: sadly Eigen only returns these in ascending order; fix this
      for(int count=ne-1; count>=0 && !converged; count--){
        ScalarType this_eig = eigs(count);
        // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;
        tmp_vector = Xtranslated*( Xtranslated.transpose() * EV.col(count) );  // TODO: MINLIN
#elif defined( USE_MINLIN )
      for(int count=0; count<ne && !converged; count++){
        ScalarType this_eig = er(count);
        // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;
        {
          GenericVector tmp_ne(Xtranslated.cols());
          gemv_wrapper( tmp_ne.pointer(), EV.pointer()+count*N, Xtranslated, 1., 0., 'T' );
          gemv_wrapper( tmp_vector.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
        }    
#else
        ERROR:  -DUSE_EIGEN or -DUSE_MINLIN
#endif        

        // find the residual
        // r = Xtranslated*( Xtranslated.transpose() * EV.col(count) ) - this_eig*EV.col(count);
        // Global summation or matrix product
        MPI_Allreduce( GET_POINTER(tmp_vector), GET_POINTER(r), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        // compute the relative error from the residual
        r -= GET_COLUMN(EV,count) * this_eig;   //residual
        ScalarType this_err = std::abs( NORM(r) / this_eig );
        max_err = std::max(max_err, this_err);
        // terminate early if the current error exceeds the tolerance
        // std::cout << "iteration : " << j << " count " << count << ", this_eig : " << this_eig << "max_err" << max_err << std::endl;

        if(max_err > tol)
          break;
      } // end-for error estimation
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

  return (!converged);

}
