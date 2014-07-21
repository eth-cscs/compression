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

  void initialize_data() {

    // set up data
    cluster_means_ = GenericMatrix(Nc_, K_);
    eigenvectors_ = std::vector<GenericMatrix>(K_, GenericMatrix(Nc_, M_));
    X_reduced_ = std::vector<GenericMatrix>(K_, GenericMatrix(M_, Nd_));
    cluster_indices_ = std::vector<int>(Nd_, 0);
    
    // set up sizes
    original_size = Nc_ * Nd_;
    int total_Nd = Nd_; // TODO: get global Nd
    compressed_size = (K_ * ( M_ + 1) * (Nc_ + total_Nd));

  }

  void do_iterative_clustering() {
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

  std::vector<int> find(const std::vector<int> int_vector, const int k) {
    std::vector<int> found_items;
    for (int i=0; i<int_vector.size(); i++) {
      if (int_vector[i] == k) { found_items.push_back(i); }
    }
    return found_items;
  }



};
