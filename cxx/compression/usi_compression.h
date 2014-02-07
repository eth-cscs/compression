#ifndef _USI_COMPRESSION_H
#define _USI_COMPRESSION_H

#if defined( USE_EIGEN )
#include <Eigen/Dense>
using namespace Eigen;

typedef double ScalarType;     // feel free to change this to 'double' if supported by your hardware

typedef Matrix<ScalarType, Dynamic, Dynamic, RowMajor> MatrixXXrow;
typedef Matrix<ScalarType, Dynamic, Dynamic, ColMajor> MatrixXX;
typedef Matrix<ScalarType, Dynamic, 1> VectorX;
typedef Array<int, Dynamic, 1> ArrayX1i;


#elif defined( USE_MINLIN )

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
MINLIN_INIT
#include <cublas_v2.h>
#endif

#else
ERROR!   MUST USE A LIBRARY, E.G. USE_EIGEN OR USE_MINLIN
#endif

#endif
