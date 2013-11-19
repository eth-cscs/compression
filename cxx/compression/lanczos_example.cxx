#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main()
{
  typedef Matrix<float, Dynamic, Dynamic> MatrixXX;
  typedef Matrix<float, Dynamic, 1> VectorX;

  MatrixXX A(6,6);
  A << 1,2,3,4,5,6, 2,4,5,6,7,8, 3,5,7,8,9,10, 4,6,8,11,12,13, 5,7,9,12,15,16, 6,8,10,13,16,19;

  float gamma, delta;
  float crap;
  float tol = 2.0e-5;

  int nrows = A.rows();
  int nbr_eig = 2;
  int max_iter = 5;
  MatrixXX EV = MatrixXX::Zero(nrows,nbr_eig);   // output
  MatrixXX V  = MatrixXX::Zero(nrows,max_iter);  // transformation
  MatrixXX Trid  = MatrixXX::Zero(max_iter,max_iter);  // Tridiagonal
  V.col(0).setRandom();   // Warning:  all nodes must generate same random vector in parallel (seed)
  V.col(0) /= V.col(0).norm();    // Unit vector
  MatrixXX normr  = MatrixXX::Zero(nbr_eig,1);  // norms

  MatrixXX AV = A*V.col(0);  // order important! evaluate right to left to save calculation!


  cout << "Here is the matrix A:\n" << A << endl;


  for(int iter=1; iter < max_iter; iter++) {  // main loop, will terminate earlier if tolerance is reached

    if ( iter == 1 ) { V.col(1) = AV; }
    else {  V.col(iter) = AV - gamma*V.col(iter-2); }

    delta = V.col(iter).transpose() * V.col(iter-1);
    V.col(iter) -= delta*V.col(iter-1);

    gamma  = V.col(iter).norm();
    V.col(iter) /= gamma;

    // reorthogonalize  -- this can be an argument-driven option
    for( int sub_iter = 0; sub_iter < iter; sub_iter++ )  {
      crap = V.col(iter).transpose()*V.col(sub_iter);
      V.col(iter) -= crap*V.col(sub_iter);
      //      V.col(iter) -= (V.col(iter).transpose()*V.col(sub_iter))*V.col(sub_iter);  // run-time crash... Why?
    }

    AV = A*V.col(iter);  
    Trid(iter-1,iter-1) = delta;  Trid(iter-1,iter) = gamma;  Trid(iter,iter-1) = gamma;
    SelfAdjointEigenSolver<MatrixXX> eigensolver(Trid.block(0,0,iter+1,iter+1));
    if (eigensolver.info() != Success) abort();

    VectorX  eigs = eigensolver.eigenvalues();  // Sorted: largest eigenvalues are last
    MatrixXX UT = eigensolver.eigenvectors();   // Corresponding eigenvectors

    if ( iter >= nbr_eig ) {
      EV = V.block(0,0,nrows,iter+1)*UT.block(0,iter-nbr_eig+1,iter+1,nbr_eig);
      VectorX r(nrows);
      float max_err = 0.0;
      for (int count = 0; count < nbr_eig; count++) { 
        float this_eig = eigs(count+iter-nbr_eig+1);
	max_err = max( abs(((A*EV.col(count) - EV.col(count)*this_eig).norm())/this_eig), max_err);
	cout << " Ritz value of interest " << this_eig << endl; 
      }
      cout << "The maximum error is : " << max_err << endl;
      if ( max_err < tol ) break; 
    }
  }

  // Now check whether the 
  SelfAdjointEigenSolver<MatrixXX> eigensolver(A);
  if (eigensolver.info() != Success) abort();
  cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << endl;
  /*
  cout << "Here's a matrix whose columns are eigenvectors of A \n"
       << "corresponding to these eigenvalues:\n"
       << eigensolver.eigenvectors() << endl;
  */
}
