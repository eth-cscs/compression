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

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  GenericColMatrix V(N,max_iter);  // transformation

  GenericVector w(N);

  // preallocate storage vectors
  GenericVector r(N);   // residual, temporary
  GenericVector tmp_vector(N);   // temporary

  GenericVector er(max_iter);        // ScalarType components of eigenvalues
  GenericVector ei(max_iter);        // imaginary components of eigenvalues
  V(all,0) = ScalarType(1.); //TODO: make this a random vector
  V(all,0) /= norm(V(all,0));    // Unit vector
  GenericColMatrix Trid(max_iter,max_iter);  // Tridiagonal
  Trid(all) = 0.;                            // Matrix must be zeroed out
  {
    GenericVector tmp_ne(Xtranslated.cols());
    // calculate Xtranslated.T*V0 (TODO: find better way to do this)
    gemv_wrapper( tmp_ne.pointer(), V.pointer(), Xtranslated, 1., 0., 'T' );
    //for(int i=0; i<tmp_ne.size(); i++) {
    //  tmp_ne(i) = dot(Xtranslated(all,i),V(all,0));
    //}
    // calculate Xtranslated*(Xtranslated.T*V0)
    gemv_wrapper( tmp_vector.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
    //tmp_vector = Xtranslated * tmp_ne;
    //std::cout << "doing initial A*x0 calculation" << std::endl;
  }
  MPI_Allreduce( tmp_vector.pointer(), w.pointer(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  delta = dot(w, V(all,0));

  Trid(0,0) = delta;  // store in tridiagonal matrix

  // main loop, will terminate earlier if tolerance is reached
  bool converged = false;
  for(int j=1; j<max_iter && !converged; ++j) {
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
  {
    GenericVector tmp_ne(Xtranslated.cols());
    gemv_wrapper( tmp_ne.pointer(), V.pointer()+j*N, Xtranslated, 1., 0., 'T' );
    gemv_wrapper( r.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
  }    
    MPI_Allreduce( GET_POINTER(r), GET_POINTER(w), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    // update diagonal of tridiagonal system
    delta = DOT_PRODUCT( w, GET_COLUMN(V,j) );
    Trid(j, j) = delta;
    if ( j >= ne ) {
      // find eigenvectors/eigenvalues for the reduced triangular system
      GenericVector eigs(j+1); //TODO: this is not used, use either eigs or er
      {
        HostMatrix<ScalarType> Tsub = Trid(0,j,0,j);
        HostMatrix<ScalarType> UVhost(j+1,ne);
#ifdef FULL_EIGENSOLVE
        //TODO: this is not working
        // The geev routine calculates all j+1 vectors & eigenvalues
        // whereas the UV matrix only has ne columns.
        // If this should be made to work, geev needs to return only the
        // ne eigenvectors with the largest eigenvalues
        assert( geev(Tsub.pointer(), UVhost.pointer(), er.pointer(), ei.pointer(), j+1) );
        //std::cout << "er: " << er(0,j) << std::endl;
#else
        assert( steigs( Tsub.pointer(), UVhost.pointer(), er.pointer(), j+1, ne) );
        //std::cout << "er: " << er(0,j) << std::endl;
#endif
        //TODO: make sure eigenvalues have ascending/descending order
        // copy eigenvectors for reduced system to the device
        GenericColMatrix UV = UVhost;

        // find approximate eigenvectors of full system
        EV = V(all,0,j)*UV;
      }

      // copy eigenvectors for reduced system to the device
      ////////////////////////////////////////////////////////////////////
      // TODO : can we find a way to allocate memory for UV outside the
      //        inner loop? this memory allocation is probably killing us
      //        particularly if we go to large subspace sizes
      //
      ////////////////////////////////////////////////////////////////////

      ScalarType max_err = 0.;

      for(int count=0; count<ne && !converged; count++){
        ScalarType this_eig = er(count);
        // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;
        {
          GenericVector tmp_ne(Xtranslated.cols());
          gemv_wrapper( tmp_ne.pointer(), EV.pointer()+count*N, Xtranslated, 1., 0., 'T' );
          gemv_wrapper( tmp_vector.pointer(), tmp_ne.pointer(), Xtranslated, 1., 0., 'N' );
        }    

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
  } // end-for main

  // return failure if no convergence
  return (!converged);

}
