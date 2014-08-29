/** \file lanczos_correlation_minlin.h
 *
 *  This file contains the minlin implementation of the Lanczos algorithm for
 *  a correlation matrix as well as a wrapper around the LAPACK eigensolver.
 *
 *  \copyright Copyright (c) 2014,
 *             Universita della Svizzera italiana (USI) &
 *             Centro Svizzero di Calcolo Scientifico (CSCS).
 *             All rights reserved.
 *             This software may be modified and distributed under the terms of
 *             the BSD license. See the [LICENSE file](LICENSE.md) for details.
 *
 *  \author Will Sawyer (CSCS)
 *  \author Ben Cumming (CSCS)
 *  \author Manuel Schmid (CSCS)
 */

#pragma once
#include <cstdlib>   // std::abs
#include <algorithm> // std::copy, std::max
#include <mkl.h>     // ssteqr, dsteqr
#include <mpi.h>     // MPI_Allreduce
#include "mpi_type_helper.h"
#include "matrices.h"

/**
 * \brief Overloaded wrapper around ssteqr.
 */
lapack_int steqr(const lapack_int n, float* d, float* e, float* z, float* work) {
    int info;
    char compz = 'I';
    ssteqr(&compz, &n, d, e, z, &n, work, &info);
    return info;
}

/**
 * \brief Overloaded wrapper around dsteqr.
 */
lapack_int steqr(const lapack_int n, double* d, double* e, double* z, double* work) {
    int info;
    char compz = 'I';
    dsteqr(&compz, &n, d, e, z, &n, work, &info);
    return info;
}

/**
 * This function is a wrapper around the LAPACK eigensolver ?steqr that we use
 * for finding the eigenvectors of the Krylov subspace constructed in the
 * Lanczos solver. It resorts the eigenvalues and -vectors in descending order
 * and only returns the N largest eigenvalues and their -vectors.
 *
 * \param[in]  T        The tridiagonal matrix for which we want to find the
 *                      eigenvalues and -vectors (n*n array).
 * \param[out] V        The output matrix for the eigenvectors (n*num_eigs
 *                      array).
 * \param[out] eigs     The output vector for the eigenvalues (1*num_eigs array)
 * \param[in]  n        The dimensions of the matrix.
 * \param[in]  num_eigs The number of eigenvalues/-vectors we want to calculate
                        (optional, default: n).
 * \return              Returns 'true' if successful, 'false' otherwise.
 */
template <typename Scalar>
bool steigs(const Scalar *T, Scalar *V, Scalar *eigs, const int n, int num_eigs=-1) {

  // set num_eigs if it is omitted and avoid trying to return more than n values
  num_eigs = num_eigs<0 ? n : num_eigs;
  num_eigs = num_eigs>n ? n : num_eigs;

  // allocate memory for arrays used by LAPACK
  Scalar *e = new Scalar[n-1];   // superdiagonal
  Scalar *z = new Scalar[n*n];   // eigenvectors returned by LAPACK
  Scalar *d = new Scalar[n];       // diagonal, used by ?steqr for storing eigenvalues
  Scalar *work = new Scalar[2*n];  // working array for LAPACK

  // pack the diagonal and super diagonal of T
  int pos=0;
  for(int i=0; i<n-1; i++) {
      d[i] = T[pos];       // diagonal at T(i,i)
      e[i] = T[pos+1];     // off diagonal at T(i,i+1)
      pos += (n+1);
  }
  d[n-1] = T[pos];

  // compute eigenvalues
  lapack_int result = steqr(n, d, e, z, work);
  if(result)
      return false;

  // copy the eigenvalues/-vectors to the output arrays
  // and reverse the order as ?steqr returns them in
  // ascending order
  for(int i=0; i<num_eigs; i++) {
      Scalar* ptr_to   = V + i*n;
      Scalar* ptr_from = z + (n-i-1)*n;
      std::copy(ptr_from,  ptr_from+n,  ptr_to);
      std::copy(d + (n-i-1), d + (n-i), eigs + i);
  }

  // free working arrays
  delete[] e;
  delete[] z;
  delete[] d;
  delete[] work;

  return true;
}

/**
 * This function calculates the N largest eigenvectors for the covariance
 * matrix Xtranslated*Xtranslated.T using the Lanczos algorithm. A Krylov
 * subspace is created, in which the eigenvalues/-vectors are calculated. The
 * Ritz vectors are then used to find corresponding eigenvectors of the
 * original matrix. The algorithm iteratively increases the size of the Krylov
 * subspace until the N largest eigenvectors are found to a sufficient
 * accuracy.
 *
 * This is the minlin version of this algorithm.
 *
 * \see steigs
 *
 * \param[in]  Xtranslated      The matrix of which we want to calculate the
 *                              eigenvectors of the data's covariance matrix.
 * \param[in]  ne               The number of eigenvectors we want to find.
 * \param[in]  tol              The tolerance for the largest relative error.
 * \param[in]  max_iter         The maximum number of iterations.
 * \param[out] EV               The output matrix, the columns of which will be
 *                              set to the eigenvectors that are found.
 * \param[in]  reorthogonalize  Whether the basis of the Krylov subspace should
 *                              be reorthogonalized at every iteration step.
 *                              (default: false)
 * \return                      Returns 'false' if successful, 'true' otherwise,
 *                              similar to a POSIX exit status.
 */
template <typename Scalar>
bool lanczos_correlation(const DeviceMatrix<Scalar> &Xtranslated, const int ne, const Scalar tol, int max_iter, DeviceMatrix<Scalar> &EV, bool reorthogonalize = false)
{
  int N = Xtranslated.rows(); // this corresponds to Ntl in usi_compression.cpp
  Scalar gamma, delta;

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  // check whether the cluster is empty
  int local_vectors = Xtranslated.cols();
  int vectors_in_cluster;
  MPI_Allreduce(&local_vectors, &vectors_in_cluster, 1, MPI_INT, MPI_SUM,
      MPI_COMM_WORLD);
  if (!vectors_in_cluster) {
      std::cout << "WARNING: empty cluster skipped" << std::endl;
      return false; // corresponds to success
  }

  // set up matrices for Arnoldi decomposition
  DeviceMatrix<Scalar> V(N,max_iter);  // transformation
  V(all,0) = Scalar(1.); //TODO: make this a random vector
  V(all,0) /= norm(V(all,0));    // Unit vector
  HostMatrix<Scalar> Trid(max_iter,max_iter);  // Tridiagonal
  Trid(all) = 0.;                            // Matrix must be zeroed out
  
  // preallocate storage vectors
  DeviceVector<Scalar> r(N);   // residual, temporary
  DeviceVector<Scalar> w(N);
  DeviceVector<Scalar> tmp_vector(N);   // temporary
  DeviceVector<Scalar> tmp_ne(Xtranslated.cols()); // used for storing intermediate result because
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
#if defined(USE_GPU)
  HostVector<Scalar> tmp_local(N);
  HostVector<Scalar> tmp_global(N);
  tmp_local = tmp_vector;
  MPI_Allreduce(tmp_local.pointer(), tmp_global.pointer(), N,
      mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
  w = tmp_global;
#else
  MPI_Allreduce( tmp_vector.pointer(), w.pointer(), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );
#endif
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
        Scalar alpha =  DOT_PRODUCT( GET_COLUMN(V,jj), GET_COLUMN(V,j) ) ;
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
#if defined(USE_GPU)
    tmp_local = r;
    MPI_Allreduce(tmp_local.pointer(), tmp_global.pointer(), N,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    w = tmp_global;
#else
    MPI_Allreduce( GET_POINTER(r), GET_POINTER(w), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );
#endif

    // update diagonal of tridiagonal system
    delta = DOT_PRODUCT( w, GET_COLUMN(V,j) );
    Trid(j, j) = delta;

    // we only calculate the eigenvalues & vectors if we have done enough iterations
    // to calculate as many as we need
    // the eigenvalues/-vectors are always calculated on the host
    if ( j >= ne ) {

      // find eigenvectors/eigenvalues for the reduced triangular system
      HostVector<Scalar> eigs(ne);
      
      HostMatrix<Scalar> Tsub = Trid(0,j,0,j);
      HostMatrix<Scalar> UVhost(j+1,ne);

      // we calculate the eigenvalues with a MKL routine as minlin doesn't
      // have an eigensolver
      assert( steigs( Tsub.pointer(), UVhost.pointer(), eigs.pointer(), j+1, ne) );
      
      // copy eigenvectors for reduced system to the device
      DeviceMatrix<Scalar> UV = UVhost;

      // find approximate eigenvectors of full system
      EV = V(all,0,j)*UV;

      ////////////////////////////////////////////////////////////////////
      // TODO : can we find a way to allocate memory for UV outside the
      //        inner loop? this memory allocation is probably killing us
      //        particularly if we go to large subspace sizes
      //
      ////////////////////////////////////////////////////////////////////

      // check whether we have converged to the tolerated error
      Scalar max_err = 0.0;
      for(int count=0; count<ne && !converged; count++){
        Scalar this_eig = eigs(count);
        
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
#if defined(USE_GPU)
        tmp_local = tmp_vector;
        MPI_Allreduce(tmp_local.pointer(), tmp_global.pointer(), N,
            mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
        r = tmp_global;
#else
        MPI_Allreduce( GET_POINTER(tmp_vector), GET_POINTER(r), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );
#endif
        // compute the relative error from the residual
        r -= GET_COLUMN(EV,count) * this_eig;   //residual
        Scalar this_err = std::abs( GET_NORM(r) / this_eig );
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
