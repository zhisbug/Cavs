#include "cavs/backend/cublas_wrapper.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

#include <vector>
#include <algorithm>

using std::vector;

namespace backend {

template <>
void MatMulMatCublasWrapper<float>(
    cublasHandle_t handle,
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
  checkCublasError(cublasSgemm(handle,
      cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void MatMulMatCublasWrapper<double>(
    cublasHandle_t handle,
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
  checkCublasError(cublasDgemm(handle,
      cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void MatMulVecCublasWrapper<float>(
    const bool TransA,
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
    const bool TransA,
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

template<>
void AxpyCublasWrapper<float>(
    const int N, const float alpha,
    const float* x, float* y) {
  int incx = 1;
  checkCublasError(cublasSaxpy(CudaCommon::cublasHandle(),
      N, &alpha, x, incx, y, incx));
}

template<>
void AxpyCublasWrapper<double>(
    const int N, const double alpha,
    const double* x, double* y) {
  int incx = 1;
  checkCublasError(cublasDaxpy(CudaCommon::cublasHandle(),
      N, &alpha, x, incx, y, incx));
}

template <>
void ScalCublasWrapper<float>(
    const int N, const float alpha,
    float* x) {
  int incx = 1; 
  checkCublasError(cublasSscal(CudaCommon::cublasHandle(),
      N, &alpha, x, incx));
}

template <>
void ScalCublasWrapper<double>(
    const int N, const double alpha,
    double* x) {
  int incx = 1; 
  checkCublasError(cublasDscal(CudaCommon::cublasHandle(),
      N, &alpha, x, incx));
}

template <>
void AsumCublasWrapper<float>(
    const int N, const float* x,
    float* y) {
  //it is useful for chain rule
  //because all the data are located on GPU
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasSasum(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void AsumCublasWrapper<double>(
    const int N, const double* x,
    double* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasDasum(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void AsumCublasWrapperHost<float>(
    const int N, const float* x,
    float* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasSasum(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void AsumCublasWrapperHost<double>(
    const int N, const double* x,
    double* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasDasum(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void Nrm2CublasWrapper<float>(
    const int N, const float* x,
    float* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasSnrm2(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void Nrm2CublasWrapper<double>(
    const int N, const double* x,
    double* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasDnrm2(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void Nrm2CublasWrapperHost<float>(
    const int N, const float* x,
    float* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasSnrm2(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void Nrm2CublasWrapperHost<double>(
    const int N, const double* x,
    double* y) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasDnrm2(CudaCommon::cublasHandle(),
      N, x, incx, y)); 
}

template <>
void ArgminCublasWrapper<float>(
    const int N, const float* x,
    int* index) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasIsamin(CudaCommon::cublasHandle(),
      N, x, incx, index)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void ArgminCublasWrapper<double>(
    const int N, const double* x,
    int* index) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasIdamin(CudaCommon::cublasHandle(),
      N, x, incx, index)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void ArgmaxCublasWrapper<float>(
    const int N, const float* x,
    int* index) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasIsamax(CudaCommon::cublasHandle(),
      N, x, incx, index)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void ArgmaxCublasWrapper<double>(
    const int N, const double* x,
    int* index) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_DEVICE));
  int incx = 1;
  checkCublasError(cublasIdamax(CudaCommon::cublasHandle(),
      N, x, incx, index)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void ScalCublasWrapper<float>(
    int N, const float* alpha,
    float* x) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasSscal(CudaCommon::cublasHandle(),
      N, alpha, x, incx)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

template <>
void ScalCublasWrapper<double>(
    int N, const double* alpha,
    double* x) {
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
  int incx = 1;
  checkCublasError(cublasDscal(CudaCommon::cublasHandle(),
      N, alpha, x, incx)); 
  checkCublasError(cublasSetPointerMode(CudaCommon::cublasHandle(), 
        CUBLAS_POINTER_MODE_HOST));
}

} //namespace backend
