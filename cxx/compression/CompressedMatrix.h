#include "matrices.h"

template<class Scalar>
class CompressedMatrix
{

public:

  int original_size;
  int compressed_size;

  CompressedMatrix(GenericMatrix &X, const int K, const int M) {

    Nc_ = X.rows(); // number of entries along compressed direction
    Nd_ = X.cols(); // number of entries along distributed direction
    K_  = K;        // number of clusters
    M_  = M;        // number of eigenvectors per cluster

    initialize_data();
    do_iterative_clustering();
    do_final_pca();
    calculate_reduced_form();

  }

  GenericMatrix reconstruct() {

    GenericMatrix X_reconstructed(Nc_, Nd_);

    // This loop can be multithreaded, if all threads have a separate
    // copy of Xtranslated
    for(int k = 0; k < K_; k++) {
      std::vector<int> Nonzeros = find(cluster_indices_, k);
      for (int m = 0; m < Nonzeros.size() ; m++ ) {       
        GET_COLUMN(X_reconstructed, Nonzeros[m]) = eigenvectors_[k]
            * GET_COLUMN(X_reduced_[k], Nonzeros[m]);
        GET_COLUMN(X_reconstructed, Nonzeros[m]) += GET_COLUMN(cluster_means_, k);
      }
    }

    return X_reconstructed;
  }


private:

  // data structures for reduced representation
  std::vector<GenericMatrix> X_reduced_;
  GenericMatrix cluster_means_;
  std::vector<GenericMatrix> eigenvectors_;
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
    int *Nd_global = new int[sizeof(int)*mpi_processes];
    MPI_Allgather( &Nd_, 1, MPI_INT, Nd_global, 1, MPI_INT,
        MPI_COMM_WORLD);
    Nd_total = 0;
    for (int i=0; i<mpi_processes; i++) {Nd_total_ += Nd_global[i];}

    // set up data
    cluster_means_ = GenericMatrix(Nc_, K_);
    eigenvectors_ = std::vector<GenericMatrix>(K_, GenericMatrix(Nc_, M_));
    X_reduced_ = std::vector<GenericMatrix>(K_, GenericMatrix(M_, Nd_));
    
    // initialize clusters independent from number of processes
    cluster_start = 0;
    for (int i=0; i<my_rank_; i++) cluster_start += Nd_global[i];
    for (int i=0; i<Nd_; i++) cluster_indices_[i] = (cluster_start+i)%K_;

    // set up sizes
    original_size = Nc_ * Nd_;
    int total_Nd = Nd_; // TODO: get global Nd
    compressed_size = (K_ * ( M_ + 1) * (Nc_ + total_Nd));

  }

  void do_iterative_clustering() {

    // eigenvectors: 1 for each k
    std::vector<GenericColMatrix> TT(K_,GenericColMatrix(Nc_,1) );

    ScalarType L_value_old = 1.0e19;   // Very big value
    ScalarType L_value_new;

#if defined( USE_EIGEN )
    // initialize random seed used in lanczos algorithm
    srand(RANDOM_SEED);
#endif

    for (int iter = 0; iter < MAX_ITER; iter++) {

      // determine X column means for each active state denoted by gamma_ind
      theta_s<ScalarType>(gamma_ind, X, theta);
      
      // we calculate the L value here for output only (TODO: remove this for optimization later)
      L_value_new =  L_value( gamma_ind, TT, X, theta );
      if (!my_rank) std::cout << "L value after Theta calc " << L_value_new << std::endl;
      // Not clear if there should be monotonic decrease here:  new theta_s needs new TT, right?
      
      // Principle Component Analysis for every cluster
      for(int k = 0; k < K_size; k++) {
        std::vector<int> Nonzeros = find( gamma_ind, k );
        GenericColMatrix Xtranslated( Ntl, Nonzeros.size() ) ;
        // if (!my_rank) std::cout << " For k = " << k << " nbr nonzeros " << Nonzeros.size() << std::endl;
        for (int m = 0; m < Nonzeros.size() ; m++ )        // Translate X columns with mean value at new origin
        {
          GET_COLUMN(Xtranslated,m) = GET_COLUMN(X,Nonzeros[m]) - GET_COLUMN(theta,k) ; 
        }
        bool success = lanczos_correlation(Xtranslated, 1, 1.0e-11, 50, TT[k], true);
      }

      // we calculate the L value here for output only (TODO: remove this for optimization later)
      L_value_new =  L_value( gamma_ind, TT, X, theta );
      if (!my_rank) std::cout << "L value after PCA " << L_value_new << std::endl;
      
      // find new optimal clustering
      gamma_s( X, theta, TT, gamma_ind );

      // calculate new L value and decide whether to continue
      L_value_new =  L_value( gamma_ind, TT, X, theta ); 
      if (!my_rank) std::cout << "L value after gamma minimization " << L_value_new << std::endl;
      if ( (L_value_old - L_value_new) < L_value_new*TOL ) {
        if (!my_rank) std::cout << " Converged: to tolerance " << TOL << " after " << iter << " iterations " << std::endl;
        break;
      }
      else if ( L_value_new > L_value_old ) { 
        if (!my_rank) std::cout << "New L_value " << L_value_new << " larger than old: " << L_value_old << " aborting " << std::endl;
        break;
      }
      else if ( iter+1 == MAX_ITER ) {
        if (!my_rank) std::cout << " Reached maximum number of iterations " << MAX_ITER << " without convergence " << std::endl;
      }
      L_value_old = L_value_new;
    }









    // TODO
    return;
  }

  void do_final_pca() {
    // TODO
    return;
  }

  void calculate_reduced_form() {
    // TODO
    return;
  }

  void update_clustering(const GenericMatrix &X, const GenericMatrix &TT) {

    std::vector<Scalar> smallest_norm(Nd_, std::numeric_limits<Scalar>::max());

    GenericMatrix X_translated(Nc_, Nd_);
#if defined(USE_MINLIN)
    GenericVector tmp_Nd(Nd_); // used in for loop
#endif

    // This loop can be multithreaded, if all threads have a separate
    // copy of Xtranslated
    for(int k = 0; k < K_; k++) {

      // Translate each column of X. This loop can be multithreaded!
      for(int i=0; i<Nd_; i++) GET_COLUMN(X_translated,i) = GET_COLUMN(X,i)
          - GET_COLUMN(cluster_means_,k);

#if defined(USE_EIGEN)
      X_translated -= TT[k] * (TT[k].transpose() * X_translated);
#elif defined(USE_MINLIN)
      tmp_Nd(all) = transpose(X_translated) * TT[k];
      geru_wrapper(X_translated, TT[k].pointer(), tmp_Nd.pointer(), -1.);
#endif

      for(int i=0; i<Nd_; i++) {
        Scalar this_norm = NORM(GET_COLUMN(X_translated,i));
        if(this_norm<smallest_norm[i]) {
          smallest_norm[i] = this_norm;
          cluster_indices[i] = k;
        }
      }
    }
  }

  std::vector<int> find(const std::vector<int> int_vector, const int k) {
    std::vector<int> found_items;
    for (int i=0; i<int_vector.size(); i++) {
      if (int_vector[i] == k) { found_items.push_back(i); }
    }
    return found_items;
  }



};
