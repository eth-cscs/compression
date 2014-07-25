#include "matrices.h"
#include "mpi_type_helper.h"

#if defined(USE_EIGEN)
#include "lanczos_correlation_eigen.h"
#elif defined(USE_MINLIN)
#include "lanczos_correlation_minlin.h"
#endif

template<class Scalar>
class CompressedMatrix
{

public:

  int original_size;
  int compressed_size;

  CompressedMatrix(const DeviceMatrix<Scalar> &X, const int K, const int M) {

    Nc_ = X.rows(); // number of entries along compressed direction
    Nd_ = X.cols(); // number of entries along distributed direction
    K_  = K;        // number of clusters
    M_  = M;        // number of eigenvectors per cluster

    initialize_data();
    do_iterative_clustering(X);
    do_final_pca(X);
    calculate_reduced_form(X);

  }

  DeviceMatrix<Scalar> reconstruct() {

    DeviceMatrix<Scalar> X_reconstructed(Nc_, Nd_);

    // This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);
      for (int m = 0; m < current_cluster_indices.size() ; m++ ) {
        GET_COLUMN(X_reconstructed, current_cluster_indices[m]) = eigenvectors_[k]
            * GET_COLUMN(X_reduced_[k], current_cluster_indices[m]);
        GET_COLUMN(X_reconstructed, current_cluster_indices[m]) += GET_COLUMN(cluster_means_, k);
      }
    }

    return X_reconstructed;
  }


private:

  // data structures for reduced representation
  std::vector< DeviceMatrix<Scalar> > X_reduced_;
  DeviceMatrix<Scalar> cluster_means_;
  std::vector< DeviceMatrix<Scalar> > eigenvectors_;
  std::vector<int> cluster_indices_;

  // data sizes
  int K_;
  int M_;
  int Nc_;
  int Nd_;
  int Nd_total_;

  // information about MPI
  int mpi_processes_;
  int my_rank_;

  void initialize_data() {
    
    // collect number of columns each process has
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
    int *Nd_global = new int[mpi_processes_];
    MPI_Allgather( &Nd_, 1, MPI_INT, Nd_global, 1, MPI_INT,
        MPI_COMM_WORLD);
    Nd_total_ = 0;
    for (int i=0; i<mpi_processes_; i++) {Nd_total_ += Nd_global[i];}

    // set up data
    cluster_indices_ = std::vector<int>(Nd_, 0);
    cluster_means_ = DeviceMatrix<Scalar>(Nc_, K_);
    eigenvectors_ = std::vector< DeviceMatrix<Scalar> >(K_, DeviceMatrix<Scalar>(Nc_, M_));
    X_reduced_ = std::vector< DeviceMatrix<Scalar> >(K_, DeviceMatrix<Scalar>(M_, Nd_));
    
    // initialize clusters independent from number of processes
    int cluster_start = 0;
    for (int i=0; i<my_rank_; i++) cluster_start += Nd_global[i];
    for (int i=0; i<Nd_; i++) cluster_indices_[i] = (cluster_start+i)%K_;

    // set up sizes
    original_size = Nc_ * Nd_total_;
    compressed_size = (K_ * (M_ + 1) * (Nc_ + Nd_total_));

    delete[] Nd_global;

  }

  void do_iterative_clustering(const DeviceMatrix<Scalar> &X) {

    // eigenvectors: 1 for each k
    std::vector< DeviceMatrix<Scalar> > TT(K_, DeviceMatrix<Scalar>(Nc_, 1));

    Scalar L_value_old = 1.0e19;   // Very big value
    Scalar L_value_new;

#if defined( USE_EIGEN )
    // initialize random seed used in lanczos algorithm
    srand(RANDOM_SEED);
#endif

    for (int iter = 0; iter < MAX_ITER; iter++) {

      // determine X column means for each active state denoted by gamma_ind
      update_cluster_means(X);
      
      // we calculate the L value here for output only
      // TODO: remove this for optimization later
      L_value_new =  L_norm(X, TT);
      if (!my_rank_) std::cout << "L value after Theta calc " << L_value_new << std::endl;
      // Not clear if there should be monotonic decrease here
      // new theta_s needs new TT, right?
      
      // Principle Component Analysis for every cluster
      for(int k = 0; k < K_; k++) {
        std::vector<int> current_cluster_indices = indices_for_cluster(k);
        DeviceMatrix<Scalar> X_translated(Nc_, current_cluster_indices.size()) ;
        for (int m = 0; m < current_cluster_indices.size() ; m++ ) {
          // Translate X columns with mean value at new origin
          GET_COLUMN(X_translated, m) = GET_COLUMN(X, current_cluster_indices[m])
              - GET_COLUMN(cluster_means_, k);
        }
        bool success = lanczos_correlation(X_translated, 1, (Scalar) 1.0e-11, 50, TT[k], true);
      }

      // we calculate the L value here for output only
      // TODO: remove this for optimization later
      L_value_new = L_norm(X, TT);
      if (!my_rank_) std::cout << "L value after PCA " << L_value_new << std::endl;
      
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
      lanczos_correlation(X_translated, M_, (Scalar) 1.0e-8, Nc_, eigenvectors_[k], true);
    }
    Scalar L_value_final = L_norm(X, eigenvectors_);
    if (!my_rank_) std::cout << "L value final " << L_value_final << std::endl;
  }

  void calculate_reduced_form(const DeviceMatrix<Scalar> &X) {

    // This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    DeviceMatrix<Scalar> X_translated(Nc_, Nd_);
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);

      // Translate X columns with mean values and project them into subspace
      // spanned by eigenvectors
      for (int i = 0; i < current_cluster_indices.size(); i++ ) {
#if defined( USE_EIGEN )      
        X_translated.col(current_cluster_indices[i]) = X.col(current_cluster_indices[i])
            - cluster_means_.col(k);
        X_reduced_[k].col(current_cluster_indices[i]) = eigenvectors_[k].transpose()
            * X_translated.col(current_cluster_indices[i]);
#elif defined( USE_MINLIN )
        X_translated(all, current_cluster_indices[i]) = X(all, current_cluster_indices[i])
            - cluster_means_(all,k);
        X_reduced_[k](all, current_cluster_indices[i]) = transpose(eigenvectors_[k])
            * X_translated(all,current_cluster_indices[i]);
#endif
      }
    }
  }

  void update_cluster_means(const DeviceMatrix<Scalar> &X) {


    // This loop is parallel: No dependencies between the columns
    // (local_vector needs to be private)
    std::vector<int> local_cluster_sizes(K_);
    std::vector<int> global_cluster_sizes(K_);
    for(int k = 0; k < K_; k++) {
      std::vector<int> current_cluster_indices = indices_for_cluster(k);
      local_cluster_sizes[k] = current_cluster_indices.size();
    }   
    MPI_Allreduce(local_cluster_sizes.data(), global_cluster_sizes.data(), K_,
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // This loop is parallel: No dependencies between the columns
    // (local_vector needs to be private)
    DeviceMatrix<Scalar> local_means(Nc_,K_);
    for(int k = 0; k < K_; k++) {
      
      // Could use a matrix for this to avoid 2nd load;
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

  void update_clustering(const DeviceMatrix<Scalar> &X, const std::vector< DeviceMatrix<Scalar> > &TT) {

    std::vector<Scalar> smallest_norm(Nd_, std::numeric_limits<Scalar>::max());

    DeviceMatrix<Scalar> X_translated(Nc_, Nd_);
#if defined(USE_MINLIN)
    DeviceVector<Scalar> tmp_Nd(Nd_); // used in for loop
#endif

    // This loop can be multithreaded, if all threads have a separate
    // copy of X_translated
    for(int k = 0; k < K_; k++) {

      // Translate each column of X. This loop can be multithreaded!
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

  std::vector<int> indices_for_cluster(const int k) {
    std::vector<int> found_items;
    for (int i=0; i<cluster_indices_.size(); i++) {
      if (cluster_indices_[i] == k) { found_items.push_back(i); }
    }
    return found_items;
  }

};
