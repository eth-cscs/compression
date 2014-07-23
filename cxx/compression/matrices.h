#pragma once

#if defined( USE_EIGEN )

#define GET_COLUMN( VARIABLE, COLUMN_NR )  VARIABLE.col(COLUMN_NR)
#define GET_POINTER( VARIABLE )            VARIABLE.data()
#define GET_NORM( VARIABLE )               VARIABLE.norm()
#define DOT_PRODUCT( VECTOR1, VECTOR2 )    VECTOR1.transpose() * VECTOR2
#define NORM( VECTOR )                     VECTOR.norm()
#include <Eigen/Dense>

template<class Scalar> using HostMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template<class Scalar> using DeviceMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template<class Scalar> using HostVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<class Scalar> using DeviceVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
MINLIN_INIT
#include <cublas_v2.h>
#endif

// double precision geru (rank-1 update)
// A <- A - alpha*x*y'
template<typename Scalar>
bool geru_wrapper( DeviceMatrix<Scalar> &A, const double* x, const double* y, 
    double alpha )
{
  const int inc = 1;
#ifdef USE_GPU
  cublasHandle_t handle = CublasState::instance()->handle();
  cublasStatus_t status = cublasDger(handle, A.rows(), A.cols(), &alpha,
      x, inc, y, inc, A.pointer(), A.rows());
  return (status==CUBLAS_STATUS_SUCCESS);
#else
  const int m = A.rows();
  const int n = A.cols();
  const int lda = m;
  dger(&m, &n, &alpha, const_cast<double*>(x), &inc, 
      const_cast<double*>(y), &inc,  A.pointer(), &lda);

  return true;
#endif
}

#else
ERROR!   MUST USE A LIBRARY, E.G. USE_EIGEN OR USE_MINLIN
#endif
