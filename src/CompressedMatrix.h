/** \file CompressedMatrix.h
 *
 *  This file defines the templated class CompressedMatrix.
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
#include <cstdlib>  // std::srand (Eigen only)
#include <iostream> // std::cout, std::endl
#include <vector>   // std::vector
#include <limits>   // std::numeric_limits
#include <algorithm>// std::max
#include <mpi.h>    // MPI_Comm_size, MPI_Comm_rank, MPI_Allreduce, MPI_Allgather
#include "mpi_type_helper.h"
#include "matrices.h"

#if defined(USE_EIGEN)
#include "lanczos_correlation_eigen.h"
#elif defined(USE_MINLIN)
#include "lanczos_correlation_minlin.h"
#endif

#define MAX_ITER 100 ///< Maximum number of iterations for clustering.
#define TOL 1.0e-7   ///< Tolerance level for cluster convergence.
#define RANDOM_SEED 123456  ///< Random seed for initial vector in Lanczos algorithm (Eigen version only)

/**
 * The class CompressedMatrix stores all the information needed to represent
 * the data in compressed form. The compression is done in the constructor
 * and there is a member function for reconstructing the full matrix. In
 * addition, there are two public variables 'original_size' and
 * 'compressed_size' with the number of elements stored for the full or
 * compressed matrix respectively.
 */
template<class Scalar>
class CompressedMatrix
{

public:

  int original_size;    ///< \brief The number of elements (scalar numbers) in
                        ///< the original (uncompressed) matrix.
  int compressed_size;  ///< \brief The number of elements (scalar numbers)
                        ///< used for the compressed representation of the
                        ///< matrix.

  /**
   * \brief Construct a compressed representation of a matrix.
   *
   * The constructor of CompressedMatrix does all the compression work. We
   * set up the data structures and call member functions to find a good
   * clustering and compress the data.
   *
   * \see initialize_data(), do_iterative_clustering(), do_final_pca(),
   *      calculate_reduced_form()
   *
   * \param[in] X           The matrix that is to be compressed.
   * \param[in] K           The number of clusters used for compression.
   * \param[in] M           The number of eigenvectors used per cluster.
   * \param[in] column_ids  A vector of unique IDs for the column of the
   *                        matrix, independent of the number of processes,
   *                        used for assigning the initial clusters.
   */
  CompressedMatrix(const DeviceMatrix<Scalar> &X, const int K, const int M,
      std::vector<int> column_ids) {

    Nc_ = X.rows();   // number of entries along compressed direction
    Nd_ = X.cols();   // number of entries along distributed direction
    K_  = K;          // number of clusters
    M_  = M;          // number of eigenvectors per cluster

    initialize_data(column_ids);
    do_iterative_clustering(X);
    do_final_pca(X);
    calculate_reduced_form(X);

  }

  /**
   * \brief Reconstruct the full matrix.
   *
   * This function reconstructs the full matrix using the compressed data.
   * This matrix should be similar to the original matrix, except for the
   * compression losses.
   *
   * \return      The reconstructed matrix.
   */
  DeviceMatrix<Scalar> reconstruct() {

    DeviceMatrix<Scalar> X_reconstructed(Nc_, Nd_);

    // TODO: This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    for (int i = 0; i < Nd_; i++) {
      // minlin needs this to be done in two steps
      GET_COLUMN(X_reconstructed, i) = eigenvectors_[cluster_indices_[i]]
        * GET_COLUMN(X_reduced_, i);
      GET_COLUMN(X_reconstructed, i) += GET_COLUMN(cluster_means_, cluster_indices_[i]);
    }

    return X_reconstructed;
  }


private:

  /// \brief The reduced representation of the data. For each column of the
  /// original matrix, this contains the coefficients for the eigenvectors of
  /// its cluster. This corresponds to the projection of the original column
  /// vector into the subspace spanned by the eigenvectors.
  DeviceMatrix<Scalar> X_reduced_;
  /// The mean vector for each cluster.
  DeviceMatrix<Scalar> cluster_means_;
  /// The M largest eigenvectors for each cluster.
  std::vector< DeviceMatrix<Scalar> > eigenvectors_;
  /// The index of the cluster each column of the original matrix belongs to.
  std::vector<int> cluster_indices_;

  // data sizes
  int K_;         ///< The number of clusters.
  int M_;         ///< The number of eigenvectors calculated fo each cluster.
  int Nc_;        ///< The number of rows of the original matrix.
  int Nd_;        ///< \brief The number of columns of the original matrix
                  ///< that are assigned to the current process.
  int Nd_total_;  ///< The total number of columns of the original matrix.

  // information about MPI
  int my_rank_;   ///< \brief The rank (ID) of the current MPI process. This
                  ///< is mainly used to limit the console output to one
                  ///< process.

  /**
   * This function is called by the constructor and sets up the member
   * variables with the correct sizes. It initializes the cluster indicies
   * based on the column ids that are passed to it and it calculates the
   * original and compressed matrix sizes based on the matrix dimensions
   * and parameters.
   *
   * \param[in] column_ids  A vector with an ID for each column of the data
   *                        matrix that is passed to the current process.
   */
  void initialize_data(std::vector<int> column_ids) {
    
    // collect number of columns each process has
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
    MPI_Allreduce(&Nd_, &Nd_total_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // set up data
    cluster_indices_ = std::vector<int>(Nd_, 0);
    cluster_means_ = DeviceMatrix<Scalar>(Nc_, K_);
    eigenvectors_ = std::vector< DeviceMatrix<Scalar> >(K_, DeviceMatrix<Scalar>(Nc_, M_));
    X_reduced_ = DeviceMatrix<Scalar>(M_, Nd_);
    
    // initialize clusters independent from number of processes
    for (int i=0; i<Nd_; i++) cluster_indices_[i] = column_ids[i]%K_;

    // set up sizes
    original_size = Nc_ * Nd_total_;
    compressed_size = Nd_total_ + Nc_ * K_ + K_ * Nc_ * M_ + M_ * Nd_total_;
  }

  /**
   * This function iterates until a near optimal clustering is found. For each
   * iteration, it does a PCA for each cluster finding the largest eigenvector
   * only. It then reassigns all columns to the cluster where they are best
   * represented by the eigenvector. This is repeated until the norm measuring
   * the difference between the columns and the eigenvectors doesn't change
   * noticeably anymore.
   *
   * \param[in] X   The matrix that is to be compressed.
   */
  void do_iterative_clustering(const DeviceMatrix<Scalar> &X) {

    // eigenvectors: 1 for each k
    std::vector< DeviceMatrix<Scalar> > TT(K_, DeviceMatrix<Scalar>(Nc_, 1));

    Scalar L_value_old = std::numeric_limits<Scalar>::max();
    Scalar L_value_new;

#if defined( USE_EIGEN )
    // initialize random seed used in lanczos algorithm
    std::srand(RANDOM_SEED);
#endif

    for (int iter = 0; iter < MAX_ITER; iter++) {

      // determine X column means for each active state denoted by gamma_ind
      update_cluster_means(X);
      
#if defined(DEBUG)
      // we calculate the L value here for output only
      L_value_new =  L_norm(X, TT);
      if (!my_rank_) std::cout << "L value after Theta calc " << L_value_new << std::endl;
      // Not clear if there should be monotonic decrease here
      // new theta_s needs new TT, right?
#endif
      
      // Principle Component Analysis for every cluster
      for(int k = 0; k < K_; k++) {
        std::vector<int> current_cluster_indices = indices_for_cluster(k);
        DeviceMatrix<Scalar> X_translated(Nc_, current_cluster_indices.size()) ;
        for (int m = 0; m < current_cluster_indices.size() ; m++ ) {
          // Translate X columns with mean value at new origin
          GET_COLUMN(X_translated, m) = GET_COLUMN(X, current_cluster_indices[m])
              - GET_COLUMN(cluster_means_, k);
        }
        bool success = lanczos_correlation(X_translated, 1, (Scalar) 1.0e-11,
            50, TT[k], true);
        assert(success == 0);
      }

#if defined(DEBUG)
      // we calculate the L value here for output only
      L_value_new = L_norm(X, TT);
      if (!my_rank_) std::cout << "L value after PCA " << L_value_new << std::endl;
#endif
      
      // find new optimal clustering
      update_clustering(X, TT);

      // calculate new L value and decide whether to continue
      L_value_new = L_norm(X, TT);
      if (!my_rank_) std::cout << "L value after gamma minimization " << L_value_new << std::endl;
      if ( (L_value_old - L_value_new) < L_value_new*TOL ) {
        if (!my_rank_) std::cout << " Converged: to tolerance " << TOL
            << " after " << iter << " iterations " << std::endl;
        break;
      }
      else if ( L_value_new > L_value_old ) { 
        if (!my_rank_) std::cout << "New L_value " << L_value_new
            << " larger than old: " << L_value_old << " aborting" << std::endl;
        break;
      }
      else if ( iter+1 == MAX_ITER ) {
        if (!my_rank_) std::cout << " Reached maximum number of iterations "
            << MAX_ITER << " without convergence " << std::endl;
      }
      L_value_old = L_value_new;
    }
  }

  /**
   * This function does the final PCA that is used for the actual compression
   * of the matrix. It uses the clusters found during the iterative search
   * performed in do_iterative_clustering(). Based on these, it calculates the
   * M largest eigenvectors for each cluster where M is the parameter passed
   * to the constructor.
   *
   * \param[in] X   The matrix that is to be compressed.
   */
  void do_final_pca(const DeviceMatrix<Scalar> &X) {

    update_cluster_means(X);
    
    // Principal Component Analysis for every cluster
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);
      DeviceMatrix<Scalar> X_translated(Nc_, current_cluster_indices.size());
      for (int i = 0; i < current_cluster_indices.size(); i++) {
        // Translate X columns with mean value at new origin
        GET_COLUMN(X_translated, i) = GET_COLUMN(X, current_cluster_indices[i])
            - GET_COLUMN(cluster_means_, k);
      }
      bool success = lanczos_correlation(X_translated, M_, (Scalar) 1.0e-8,
          std::max(50,3*M_), eigenvectors_[k], true);
      assert(success == 0);
    }
    Scalar L_value_final = L_norm(X, eigenvectors_);
    if (!my_rank_) std::cout << "L value final " << L_value_final << std::endl;
  }

  /**
   * This function uses the eigenvectors found in do_final_pca() to calculate
   * a reduced representation of the original matrix. Each column is
   * represented as a linear combination of the eigenvectors for the cluster
   * it belongs to. The coefficients of this linear combination are stored in
   * X_reduced_. This corresponds to a projection of the column vector into
   * the subspace spanned by the eigenvectors.
   *
   * \param[in] X   The matrix that is to be compressed.
   */
  void calculate_reduced_form(const DeviceMatrix<Scalar> &X) {

    // TODO: This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
#if defined(USE_EIGEN)
    for (int i = 0; i < Nd_; i++) {
      X_reduced_.col(i) = eigenvectors_[cluster_indices_[i]].transpose()
          * (X.col(i) - cluster_means_.col(cluster_indices_[i]));
    }
#elif defined(USE_MINLIN)
    DeviceVector<Scalar> tmp_vector(Nc_);
    for (int i = 0; i < Nd_; i++) {
      // minlin needs to do this computation in two steps
      tmp_vector(all) = X(all, i) - cluster_means_(all, cluster_indices_[i]);
      X_reduced_(all, i) = transpose(eigenvectors_[cluster_indices_[i]])
          * tmp_vector;
    }
#endif
  }

  /**
   * This function goes through all clusters and calculates the new mean
   * vector based on the current cluster assignment.
   *
   * \param[in] X   The matrix that is to be compressed.
   */
  void update_cluster_means(const DeviceMatrix<Scalar> &X) {

    // TODO: This loop is parallel: No dependencies between the columns
    // (local_vector needs to be private)
    std::vector<int> local_cluster_sizes(K_);
    std::vector<int> global_cluster_sizes(K_);
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);
      local_cluster_sizes[k] = current_cluster_indices.size();
    }   
    MPI_Allreduce(local_cluster_sizes.data(), global_cluster_sizes.data(), K_,
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // TODO: This loop is parallel: No dependencies between the columns
    // (local_vector needs to be private)
    DeviceMatrix<Scalar> local_means(Nc_,K_);
    for(int k = 0; k < K_; k++) {
      
      // TODO: Could use a matrix for this to avoid 2nd load
      std::vector<int> current_cluster_indices = indices_for_cluster(k);
      
      // Number of entries containing each index
      Scalar sum_gamma = static_cast<Scalar> (global_cluster_sizes[k]);
      
      if (sum_gamma > 0) {
#if defined( USE_EIGEN )
        local_means.col(k) =  Eigen::MatrixXd::Zero(Nc_, 1);
        for (int i = 0; i < current_cluster_indices.size(); i++ ) {
          local_means.col(k) += X.col(current_cluster_indices[i]);
        }
        local_means.col(k) /= sum_gamma;
#elif defined( USE_MINLIN )
        local_means(all,k) =  0.;
        for (int i = 0; i < current_cluster_indices.size() ; i++ ) {
          local_means(all,k) += X(all,current_cluster_indices[i]);
        }
        local_means(all,k) /= sum_gamma;
#endif
      }
    }

#if defined(USE_GPU)
    // We need to copy the matrix to the host in order to do the MPI 
    // Allreduce call. After that we copy it back to the device.
    HostMatrix<Scalar> tmp_local(Nc_,K_);
    HostMatrix<Scalar> tmp_global(Nc_, K_);
    tmp_local = local_means;
    MPI_Allreduce(tmp_local.pointer(), tmp_global.pointer(), Nc_*K_,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
    cluster_means_ = tmp_global;
#else
    MPI_Allreduce(GET_POINTER(local_means), GET_POINTER(cluster_means_), Nc_*K_,
        mpi_type_helper<Scalar>::value, MPI_SUM, MPI_COMM_WORLD);
#endif
  }

  /**
   * This function calculates the L-norm, which is a measure of how well the
   * original data can be represented in the subspace spanned by the
   * eigenvectors in EOFs.
   *
   * \param[in] X     The matrix that is to be compressed.
   * \param[in] EOFs  A vector with the eigenvectors for each cluster. Each
   *                  entry contains a matrix with the eigenvectors as
   *                  columns. We pass this explicitly instead of just using
   *                  the member variable as we use a different number of
   *                  vectors for the iterative clustering and for the final
   *                  compression.
   * \return          The scalar value of the L-norm.
   */
  Scalar L_norm(const DeviceMatrix<Scalar> &X, const std::vector< DeviceMatrix<Scalar> > &EOFs) {

    // This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    DeviceMatrix<Scalar> X_translated(Nc_, Nd_);
#if defined(USE_MINLIN)
    DeviceVector<Scalar> tmp_K(EOFs[0].cols()); // just 1 entry during clustering
    DeviceVector<Scalar> tmp_Nc(Nc_);
#endif
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);

      // Translate X columns with mean value and subtract projection
      // into subspace spanned by eigenvector(s)
      for (int i = 0; i < current_cluster_indices.size() ; i++ ) {
#if defined( USE_EIGEN )      
        X_translated.col(current_cluster_indices[i])  = X.col(current_cluster_indices[i]) - cluster_means_.col(k);
        X_translated.col(current_cluster_indices[i]) -=  EOFs[k] * (EOFs[k].transpose()
            * X_translated.col(current_cluster_indices[i]));
#elif defined( USE_MINLIN )
        X_translated(all,current_cluster_indices[i])  = X(all,current_cluster_indices[i]) - cluster_means_(all,k);
        tmp_K(all) = transpose(EOFs[k]) * X_translated(all,current_cluster_indices[i]);
        tmp_Nc(all) = EOFs[k] * tmp_K;
        X_translated(all,current_cluster_indices[i]) -= tmp_Nc;
#endif
      }
    }

    // Now X_translated contains the residuals of the column vectors,
    // the square norms just need to be summed
    Scalar local_norm = 0.0;
    for (int i = 0; i < Nd_; i++ ) {
      Scalar colnorm = GET_NORM(GET_COLUMN(X_translated, i));
      local_norm += colnorm * colnorm;
    }
    Scalar global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, mpi_type_helper<Scalar>::value,
        MPI_SUM, MPI_COMM_WORLD);
    return global_norm;
  }

  /**
   * This function goes through all columns of the original matrix and
   * calculates the difference between the original column and its projection
   * onto the largest eigenvectors for each cluster. The cluster index
   * resulting in the smallest difference is stored in cluster_indices_.
   *
   * \param[in] X   The matrix that is to be compressed.
   * \param[in] TT  A vector with the largest eigenvector for each cluster.
   */
  void update_clustering(const DeviceMatrix<Scalar> &X, const std::vector< DeviceMatrix<Scalar> > &TT) {

    std::vector<Scalar> smallest_norm(Nd_, std::numeric_limits<Scalar>::max());

    DeviceMatrix<Scalar> X_translated(Nc_, Nd_);
#if defined(USE_MINLIN)
    DeviceVector<Scalar> tmp_Nd(Nd_); // used in for loop
#endif

    // TODO: This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    for(int k = 0; k < K_; k++) {

      // Translate each column of X. TODO: This loop can be multithreaded!
      for(int i=0; i<Nd_; i++) GET_COLUMN(X_translated,i) = GET_COLUMN(X,i)
          - GET_COLUMN(cluster_means_,k);

#if defined(USE_EIGEN)
      X_translated -= TT[k] * (TT[k].transpose() * X_translated);
#elif defined(USE_MINLIN)
      tmp_Nd(all) = transpose(X_translated) * TT[k];
      geru_wrapper(X_translated, TT[k].pointer(), tmp_Nd.pointer(), (Scalar) -1);
#endif

      for(int i=0; i<Nd_; i++) {
        Scalar this_norm = GET_NORM(GET_COLUMN(X_translated,i));
        if(this_norm<smallest_norm[i]) {
          smallest_norm[i] = this_norm;
          cluster_indices_[i] = k;
        }
      }
    }
  }

  /**
   * This helper function goes through all columns of the original data and
   * returns a vector with the indices of the columns that belong to a given
   * cluster.
   *
   * \param[in] k   The index of the cluster that is to be searched.
   * \return        A vector with the indices of the matrix columns belonging
   *                to the cluster k.
   */
  std::vector<int> indices_for_cluster(const int k) {
    std::vector<int> found_items;
    for (int i=0; i<cluster_indices_.size(); i++) {
      if (cluster_indices_[i] == k) { found_items.push_back(i); }
    }
    return found_items;
  }

};
