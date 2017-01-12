#include "cavs/backend/cublas_wrapper.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <>
void MatMulCublasWrapper<float>(
    const bool TransA, const bool TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B,
    const float beta, float* C) {
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  checkCublasError(cublasSgemm(CudaCommon::cublasHandle(),
      cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void MatMulCublasWrapper<double>(
    const bool TransA, const bool TransB, 
    const int M, const int N, const int K, 
    const double alpha, const double* A, const double* B,
    const double beta, double* C) {
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  checkCublasError(cublasDgemm(CudaCommon::cublasHandle(),
      cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void MatMulVecCublasWrapper<float>(
    const bool TransA, const bool TransB, 
    const int M, const int N,
    const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  int lda = (TransA == false) ? N : M;
  int incx = 1;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  checkCublasError(cublasSgemv(CudaCommon::cublasHandle(),
      cuTransA,
      N, M, &alpha, A, lda, x, incx, &beta, y, incx));
}

template <>
void MatMulVecCublasWrapper<double>(
    const bool TransA, const bool TransB, 
    const int M, const int N,
    const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  int lda = (TransA == false) ? N : M;
  int incx = 1;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  checkCublasError(cublasDgemv(CudaCommon::cublasHandle(),
      cuTransA,
      N, M, &alpha, A, lda, x, incx, &beta, y, incx));
}

} //namespace backend
