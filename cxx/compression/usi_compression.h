#ifndef _USI_COMPRESSION_H
#define _USI_COMPRESSION_H

#include <mkl.h>

typedef double ScalarType;     // feel free to change this to 'double' if supported by your hardware

#if defined( USE_EIGEN )

#define GET_COLUMN( VARIABLE, COLUMN_NR )  VARIABLE.col(COLUMN_NR)
#define GET_POINTER( VARIABLE )            VARIABLE.data()
#define GET_NORM( VARIABLE )               VARIABLE.norm()
#define DOT_PRODUCT( VECTOR1, VECTOR2 )    VECTOR1.transpose() * VECTOR2
#define NORM( VECTOR )                     VECTOR.norm()
#include <Eigen/Dense>

typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXXrow;
typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXX;
typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorX;
// typedef Array<int, Dynamic, 1> ArrayX1i;

typedef MatrixXXrow GenericRowMatrix;
typedef MatrixXX    GenericColMatrix;
typedef VectorX     GenericVector;
// typedef ArrayX1i    GenericIntVector;

#elif defined( USE_MINLIN )

#if defined(DEBUG)
#define MINLIN_DEBUG 1
#endif

#define GET_COLUMN( VARIABLE, COLUMN_NR )  VARIABLE(all,COLUMN_NR)
#define GET_POINTER( VARIABLE )            VARIABLE.pointer()
#define GET_NORM( VARIABLE )               norm( VARIABLE )
#define DOT_PRODUCT( VECTOR1, VECTOR2 )    dot( VECTOR1, VECTOR2 )
#define NORM( VECTOR )                     norm(VECTOR)

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

#if defined( USE_GPU )
typedef DeviceMatrix<ScalarType>  GenericRowMatrix;
typedef DeviceMatrix<ScalarType>  GenericColMatrix;
typedef DeviceVector<ScalarType>  GenericVector;
#else
typedef HostMatrix<ScalarType>  GenericRowMatrix;
typedef HostMatrix<ScalarType>  GenericColMatrix;
typedef HostVector<ScalarType>  GenericVector;
#endif

// double precision gemv
// y <- alpha*A*x + beta*y
bool gemv_wrapper( double* y, const double* x, const GenericColMatrix &A, double alpha, double beta, char trans)
{
  const int inc = 1;
#ifdef USE_GPU
  cublasHandle_t handle = CublasState::instance()->handle();
  cublasOperation_t t = CUBLAS_OP_N;
  if(trans=='T') t = CUBLAS_OP_T;
  cublasStatus_t status = cublasDgemv(handle, t, A.rows(), A.cols(), &alpha, A.pointer(), A.rows(), x, inc, &beta, y, inc);
  return (status==CUBLAS_STATUS_SUCCESS);
#else
  const int m = A.rows();
  const int n = A.cols();
  const int lda = m;
  dgemv(&trans, &m, &n, &alpha, A.pointer(), &lda, const_cast<double*>(x), &inc, &beta, y, &inc);

  return true;
#endif
}

// double precision geru (rank-1 update)
// A <- A - alpha*x*y'
bool geru_wrapper( GenericColMatrix &A, const double* x, const double* y, double alpha )
{
  const int inc = 1;
#ifdef USE_GPU
  cublasHandle_t handle = CublasState::instance()->handle();
  cublasStatus_t status = cublasDger(handle, A.rows(), A.cols(), &alpha, x, inc, &beta, y, inc, A.pointer(), A.rows(), );
  return (status==CUBLAS_STATUS_SUCCESS);
#else
  const int m = A.rows();
  const int n = A.cols();
  const int lda = m;
  dger(&m, &n, &alpha, const_cast<double*>(x), &inc, const_cast<double*>(y), &inc,  A.pointer(), &lda);

  return true;
#endif
}

// meta programming to determine vector type (Device or Host) given the Matrix

template <typename matrix> struct vector_from_matrix{};
 
template <typename T>
struct vector_from_matrix<HostMatrix<T> > { typedef HostVector<T> type; };
 
template <typename T>
struct vector_from_matrix<DeviceMatrix<T> > { typedef DeviceVector<T> type; };


#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
MINLIN_INIT
#include <cublas_v2.h>
#endif

#else
ERROR!   MUST USE A LIBRARY, E.G. USE_EIGEN OR USE_MINLIN
#endif

#endif
