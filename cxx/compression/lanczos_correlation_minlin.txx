/**
	   This template performs the Lanczos algorithm on a correlation
	   matrix defined through the "Xtranslated" set of observations.
           In this case only the eigenvectors are needed

	   The function arguments listed with "param" will be compared
	   to the declaration and verified.
	 
	   @param[in]     Xtranslated Matrix with column vectors that are used for PCA    
	   @param[in]     ne          Number of eigenvectors that are to be calculated
	   @param[in]     tol         Tolerance for eigenvector residuals
	   @param[in]     max_iter    Maximum number of iterations for the Lanczos algorithm
	   @param[out]    EV          Matrix where the calculated eigenvectors are written
	   @param[in]     reorthogonalize Whether the Lanczos algorithm should use reorthogonalization (default:false)
	   @return                    Returns true if the algorithm fails
	 */


template <typename ScalarType>
bool lanczos_correlation(const GenericMatrix &Xtranslated, const int ne, const ScalarType tol, const int max_iter, GenericMatrix &EV, bool reorthogonalize=false)
{
  int N = Xtranslated.rows(); // this corresponds to Ntl in usi_compression.cpp
  ScalarType gamma, delta;

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  // set up matrices for Arnoldi decomposition
  GenericMatrix V(N,max_iter);  // transformation
  V(all,0) = ScalarType(1.); //TODO: make this a random vector
  V(all,0) /= norm(V(all,0));    // Unit vector
  GenericMatrix Trid(max_iter,max_iter);  // Tridiagonal
  Trid(all) = 0.;                            // Matrix must be zeroed out
  
  // preallocate storage vectors
  GenericVector r(N);   // residual, temporary
  GenericVector w(N);
  GenericVector tmp_vector(N);   // temporary
  GenericVector tmp_ne(Xtranslated.cols()); // used for storing intermediate result because
                                            // minlin cannot do A*(A.T*v) efficiently

  // calculate the first entry of the tridiagonal matrix
  if (Xtranslated.cols() > 0) {
    // we have to do this in two separate operations, otherwise we cannot
    // force minlin to do the multiplications in the optimal order
    tmp_ne = transpose(Xtranslated) * GET_COLUMN(V,0);
    tmp_vector = Xtranslated * tmp_ne;
  } else {
    // if there are no data columns assigned for the current cluster and process,
    // do not attempt to participate in the calculation
    tmp_vector(all) = 0;
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
        ScalarType alpha =  DOT_PRODUCT( GET_COLUMN(V,jj), GET_COLUMN(V,j) ) ;
        GET_COLUMN(V,j) -= alpha * GET_COLUMN(V,jj);
      }
    }
    
    // write off-diagonal values in tri-diagonal matrix
    Trid(j-1,j  ) = gamma;
    Trid(j  ,j-1) = gamma;

    // find matrix-vector product for next iteration
    if (Xtranslated.cols() > 0) {
      // we have to do this in two separate operations, otherwise we cannot
      // force minlin to do the multiplications in the optimal order
      tmp_ne = transpose(Xtranslated) * GET_COLUMN(V,j);
      r = Xtranslated * tmp_ne;
    } else {
      // if there are no data columns assigned for the current cluster and process,
      // do not attempt to participate in the calculation
      r(all) = 0;
    }
    MPI_Allreduce( GET_POINTER(r), GET_POINTER(w), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    // update diagonal of tridiagonal system
    delta = DOT_PRODUCT( w, GET_COLUMN(V,j) );
    Trid(j, j) = delta;

    // we only calculate the eigenvalues & vectors if we have done enough iterations
    // to calculate as many as we need
    // the eigenvalues/-vectors are always calculated on the host
    if ( j >= ne ) {

      // find eigenvectors/eigenvalues for the reduced triangular system
      GenericVector eigs(ne);
      
      HostMatrix<ScalarType> Tsub = Trid(0,j,0,j);
      HostMatrix<ScalarType> UVhost(j+1,ne);

      // we calculate the eigenvalues with a MKL routine as minlin doesn't
      // have an eigensolver
      assert( steigs( Tsub.pointer(), UVhost.pointer(), eigs.pointer(), j+1, ne) );
      
      // copy eigenvectors for reduced system to the device
      GenericMatrix UV = UVhost;

      // find approximate eigenvectors of full system
      EV = V(all,0,j)*UV;

      ////////////////////////////////////////////////////////////////////
      // TODO : can we find a way to allocate memory for UV outside the
      //        inner loop? this memory allocation is probably killing us
      //        particularly if we go to large subspace sizes
      //
      ////////////////////////////////////////////////////////////////////

      // check whether we have converged to the tolerated error
      ScalarType max_err = 0.;
      for(int count=0; count<ne && !converged; count++){
        ScalarType this_eig = eigs(count);
        
        if (Xtranslated.cols() > 0) {
          // we have to do this in two separate operations, otherwise we cannot
          // force minlin to do the multiplications in the optimal order
          tmp_ne = transpose(Xtranslated) * GET_COLUMN(EV,count);
          tmp_vector = Xtranslated * tmp_ne;
        } else {
          // if there are no data columns assigned for the current cluster and process,
          // do not attempt to participate in the calculation
          tmp_vector(all) = 0;
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
