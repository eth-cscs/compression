#pragma once
#include <cstdlib>   // std::abs
#include <algorithm> // std::max
#include <mpi.h>     // MPI_Allreduce
#include "mpi_type_helper.h"
#include "matrices.h"

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


template <typename Scalar>
bool lanczos_correlation(const DeviceMatrix<Scalar> &Xtranslated, const int ne, const Scalar tol, const int max_iter, DeviceMatrix<Scalar> &EV, bool reorthogonalize = false)
{
  int N = Xtranslated.rows();
  Scalar gamma, delta;
  bool converged = false;

  // check that output matrix has correct dimensions
  assert(EV.rows() == N);
  assert(EV.cols() == ne);
  assert(N         >= max_iter);

  DeviceMatrix<Scalar> V(N,max_iter);  // transformation

  DeviceVector<Scalar> w(N);

  // preallocate storage vectors
  DeviceVector<Scalar> r(N);   // residual, temporary
  DeviceVector<Scalar> tmp_vector(N);   // temporary

  //V.col(0).setOnes();     // Simple initial vector; no apparent side effects
  V.col(0).setRandom();     // Random initial vector, all nodes must generate same vector so use srand(RANDOM_SEED) in caller
  V.col(0) /= V.col(0).norm();    // Unit vector
  DeviceMatrix<Scalar> Trid = DeviceMatrix<Scalar>::Zero(max_iter,max_iter);  // Tridiagonal
  if (Xtranslated.cols() > 0) {
    tmp_vector = Xtranslated*(Xtranslated.transpose()*V.col(0));  // order important! evaluate right to left to save calculation!
  } else {
    // if there are no data columns assigned for the current cluster and process,
    // do not attempt to participate in the calculation
    tmp_vector.setZero();
  }
  MPI_Allreduce( tmp_vector.data(), w.data(), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );
  delta = w.transpose() * V.col(0);

  Trid(0,0) = delta;  // store in tridiagonal matrix

  Scalar convergence_error;
  int iter;

  // main loop, will terminate earlier if tolerance is reached
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
        // Scalar alpha =  V.col(jj).transpose() * V.col(j) ;
        Scalar alpha =  DOT_PRODUCT( GET_COLUMN(V,jj), GET_COLUMN(V,j) ) ;
        GET_COLUMN(V,j) -= alpha * GET_COLUMN(V,jj);
      }
    }

    
    // write off-diagonal values in tri-diagonal matrix
    Trid(j-1,j  ) = gamma;
    Trid(j  ,j-1) = gamma;

    // find matrix-vector product for next iteration
    if (Xtranslated.cols() > 0) {
      r = Xtranslated*(Xtranslated.transpose()*V.col(j));
    } else {
      // if there are no data columns assigned for the current cluster and process,
      // do not attempt to participate in the calculation
      r.setZero();
    }
    MPI_Allreduce( GET_POINTER(r), GET_POINTER(w), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );

    // update diagonal of tridiagonal system
    delta = DOT_PRODUCT( w, GET_COLUMN(V,j) );
    Trid(j, j) = delta;
    if ( j >= ne ) {
      // find eigenvectors/eigenvalues for the reduced triangular system
      Eigen::SelfAdjointEigenSolver<DeviceMatrix<Scalar>> eigensolver(Trid.block(0,0,j+1,j+1));
      if (eigensolver.info() != Eigen::Success) abort();
      // Eigen returns eigenvalues and -vectors in ascending order
      DeviceVector<Scalar>  eigs = eigensolver.eigenvalues().block(j+1-ne,0,ne,1);  // ne largest Ritz values, sorted ascending
      DeviceMatrix<Scalar> UT = eigensolver.eigenvectors();   // Ritz vectors
      // std::cout << "iteration : " << j << ", Tblock : " << Trid.block(0,0,j+1,j+1) << std::endl;
      // std::cout << "iteration : " << j << ", ritz values " << eigs << std::endl;
      // std::cout << "iteration : " << j << ", ritz vectors " << UT << std::endl;
      // j or j+1 ??
      EV = V.block(0,0,N,j+1)*UT.block(0,j+1-ne,j+1,ne);  // Eigenvector approximations for largest ne eigenvalues

      // copy eigenvectors for reduced system to the device
      ////////////////////////////////////////////////////////////////////
      // TODO : can we find a way to allocate memory for UV outside the
      //        inner loop? this memory allocation is probably killing us
      //        particularly if we go to large subspace sizes
      //
      ////////////////////////////////////////////////////////////////////


      // error estimation
      Scalar max_err = 0.0;
      for (int count = 0; count < ne; count++) {
        Scalar this_eig = eigs(count);
        // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;
        if (Xtranslated.cols() > 0) {
          tmp_vector = Xtranslated*( Xtranslated.transpose() * EV.col(count) );  // TODO: MINLIN
        } else {
          // if there are no data columns assigned for the current cluster and process,
          // do not attempt to participate in the calculation
          tmp_vector.setZero();
        }

        // find the residual
        // r = Xtranslated*( Xtranslated.transpose() * EV.col(count) ) - this_eig*EV.col(count);
        // Global summation or matrix product
        MPI_Allreduce( GET_POINTER(tmp_vector), GET_POINTER(r), N, mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD );
        // compute the relative error from the residual
        r -= GET_COLUMN(EV,count) * this_eig;   //residual
        Scalar this_err = std::abs( GET_NORM(r) / this_eig );
        max_err = std::max(max_err, this_err);

        // terminate early if the current error exceeds the tolerance
        if(max_err > tol) break;
      } // end-for error estimation

      // test for convergence
      if(max_err < tol) converged = true;

    } // end-if estimate eigenvalues
  } // end-for main
  // return failure if no convergence

  return (!converged);

}
