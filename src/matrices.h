#pragma once

#if defined( USE_EIGEN )

#define GET_COLUMN( VARIABLE, COLUMN_NR )  VARIABLE.col(COLUMN_NR)
#define GET_POINTER( VARIABLE )            VARIABLE.data()
#define GET_NORM( VARIABLE )               VARIABLE.norm()
#define DOT_PRODUCT( VECTOR1, VECTOR2 )    VECTOR1.transpose() * VECTOR2
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
#define GET_NORM( VARIABLE )               minlin::norm( VARIABLE )
#define DOT_PRODUCT( VECTOR1, VECTOR2 )    minlin::dot( VECTOR1, VECTOR2 )

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
#include <mkl.h> // sger, dger
using minlin::threx::HostMatrix;
using minlin::threx::DeviceMatrix;
using minlin::threx::HostVector;
using minlin::threx::DeviceVector;

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
namespace minlin {
namespace threx {
MINLIN_INIT
}
}
#include <cublas_v2.h>
#endif


#if defined(USE_GPU)
cublasStatus_t cublasTger(cublasHandle_t handle, int m, int n,
    const double *alpha, const double *x, int incx, const double *y, int incy,
    double *A, int lda) {
  cublasStatus_t status = cublasDger(handle, m, n, alpha, x, incx, y, incy,
      A, lda);
  return status;
}
cublasStatus_t cublasTger(cublasHandle_t handle, int m, int n,
    const float *alpha, const float *x, int incx, const float *y, int incy,
    float *A, int lda) {
  cublasStatus_t status = cublasSger(handle, m, n, alpha, x, incx, y, incy,
      A, lda);
  return status;
}
#else
void tger(const int *m, const int *n, const double *alpha, const double *x, const
    int *incx, const double *y, const int *incy, double *A, const int *lda) {
  dger(m, n, alpha, x, incx, y, incy, A, lda);
}
void tger(const int *m, const int *n, const float *alpha, const float *x, const
    int *incx, const float *y, const int *incy, float *A, const int *lda) {
  sger(m, n, alpha, x, incx, y, incy, A, lda);
}
#endif

// geru (rank-1 update)
// A <- A - alpha*x*y'
template<typename Scalar>
bool geru_wrapper(DeviceMatrix<Scalar> &A, const Scalar* x, const Scalar* y,
    Scalar alpha)
{
  const int inc = 1;
#ifdef USE_GPU
  cublasHandle_t handle = minlin::threx::CublasState::instance()->handle();
  cublasStatus_t status = cublasTger(handle, A.rows(), A.cols(), &alpha,
      x, inc, y, inc, A.pointer(), A.rows());
  return (status==CUBLAS_STATUS_SUCCESS);
#else
  const int m = A.rows();
  const int n = A.cols();
  const int lda = m;
  tger(&m, &n, &alpha, x, &inc, y, &inc,  A.pointer(), &lda);
  return true;
#endif
}

#else
#error "ERROR! MUST USE A LIBRARY, E.G. USE_EIGEN OR USE_MINLIN"
#endif
